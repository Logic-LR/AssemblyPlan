#!/usr/bin/env python3
"""Train a pairwise connection classifier using only simplified SVG geometry."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


Pair = Tuple[str, str]


SHAPE_TYPES = ["elongated_bar", "plate_like", "irregular", "point_or_line"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simplified-root", default="experiments/svg_assembly/simplified_svg")
    parser.add_argument("--alignment-file", default="svg_features/instance_part_alignment.jsonl")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="experiments/svg_assembly/reports/simplified_connection_report.json")
    parser.add_argument("--model-output", default="experiments/svg_assembly/reports/simplified_connection_model.npz")
    return parser.parse_args()


def norm_pair(a: Any, b: Any) -> Pair:
    aa, bb = str(a), str(b)
    return (aa, bb) if aa <= bb else (bb, aa)


def gt_pairs(gt_connections: Sequence[Sequence[Any]]) -> set[Pair]:
    out: set[Pair] = set()
    for conn in gt_connections:
        if len(conn) == 2:
            out.add(norm_pair(conn[0], conn[1]))
    return out


def iter_simplified_files(root: Path) -> Iterable[Path]:
    yield from sorted(root.glob("*/*/step_*/simplified_instances.json"))


def load_mask_alignment(path: Path) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            out[record["feature_path"]] = {str(k): str(v) for k, v in record["stroke_to_part"].items()}
    return out


def safe(v: Any, default: float = 0.0) -> float:
    if v is None:
        return default
    try:
        value = float(v)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def polygon_area(points: Sequence[Sequence[float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for i, p in enumerate(points):
        q = points[(i + 1) % len(points)]
        area += safe(p[0]) * safe(q[1]) - safe(q[0]) * safe(p[1])
    return abs(area) / 2.0


def bbox_from_points(points: Sequence[Sequence[float]]) -> List[float]:
    if not points:
        return [0.0, 0.0, 0.0, 0.0]
    xs = [safe(p[0]) for p in points]
    ys = [safe(p[1]) for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def bbox_overlap(a: Sequence[float], b: Sequence[float]) -> float:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def bbox_distance(a: Sequence[float], b: Sequence[float]) -> float:
    dx = max(a[0] - b[2], b[0] - a[2], 0.0)
    dy = max(a[1] - b[3], b[1] - a[3], 0.0)
    return math.hypot(dx, dy)


def min_point_distance(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> float:
    if not a or not b:
        return 1e9
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    diff = aa[:, None, :] - bb[None, :, :]
    return float(np.sqrt((diff * diff).sum(axis=2)).min())


def instance_features(inst: Dict[str, Any], canvas_width: float = 793.701, canvas_height: float = 1122.52) -> np.ndarray:
    diag = math.hypot(canvas_width, canvas_height)
    poly = inst.get("simplified_polygon") or []
    hull = inst.get("convex_hull") or []
    obb = inst.get("oriented_bbox") or []
    box = inst.get("bbox") or bbox_from_points(poly or hull or obb)
    center = inst.get("center") or [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
    bw = max(0.0, safe(box[2]) - safe(box[0]))
    bh = max(0.0, safe(box[3]) - safe(box[1]))
    onehot = [1.0 if inst.get("shape_type") == shape else 0.0 for shape in SHAPE_TYPES]
    return np.asarray(
        [
            safe(center[0]) / canvas_width,
            safe(center[1]) / canvas_height,
            bw / canvas_width,
            bh / canvas_height,
            (bw * bh) / (canvas_width * canvas_height),
            polygon_area(poly) / (canvas_width * canvas_height),
            polygon_area(hull) / (canvas_width * canvas_height),
            safe(inst.get("axis_length")) / diag,
            safe(inst.get("axis_width")) / diag,
            math.log1p(safe(inst.get("elongation"))),
            len(poly) / 12.0,
            len(hull) / 32.0,
            *onehot,
        ],
        dtype=float,
    )


def pair_features(a: Dict[str, Any], b: Dict[str, Any]) -> np.ndarray:
    fa = instance_features(a)
    fb = instance_features(b)
    pts_a = (a.get("simplified_polygon") or []) + (a.get("principal_axis") or [])
    pts_b = (b.get("simplified_polygon") or []) + (b.get("principal_axis") or [])
    box_a = a.get("bbox") or bbox_from_points(pts_a)
    box_b = b.get("bbox") or bbox_from_points(pts_b)
    center_a = a.get("center") or [(box_a[0] + box_a[2]) / 2, (box_a[1] + box_a[3]) / 2]
    center_b = b.get("center") or [(box_b[0] + box_b[2]) / 2, (box_b[1] + box_b[3]) / 2]
    diag = math.hypot(793.701, 1122.52)
    relation = np.asarray(
        [
            bbox_overlap(box_a, box_b) / (793.701 * 1122.52),
            math.log1p(bbox_overlap(box_a, box_b)) / 10.0,
            bbox_distance(box_a, box_b) / diag,
            min_point_distance(pts_a, pts_b) / diag,
            (safe(center_b[0]) - safe(center_a[0])) / 793.701,
            (safe(center_b[1]) - safe(center_a[1])) / 1122.52,
            abs(safe(center_b[0]) - safe(center_a[0])) / 793.701,
            abs(safe(center_b[1]) - safe(center_a[1])) / 1122.52,
        ],
        dtype=float,
    )
    return np.concatenate([np.minimum(fa, fb), np.maximum(fa, fb), np.abs(fa - fb), relation])


def load_examples(root: Path, alignment: Dict[str, Dict[str, str]]) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    xs: List[np.ndarray] = []
    ys: List[int] = []
    metas: List[Dict[str, Any]] = []
    for path in iter_simplified_files(root):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        feature_path = data.get("source_feature")
        inst_to_part = alignment.get(feature_path, {})
        instances = data.get("instances") or []
        if set(inst_to_part) != {str(inst["id"]) for inst in instances}:
            continue
        gt = data.get("gt") or {}
        gold = gt_pairs(gt.get("connections") or [])
        split = gt.get("part_segmentation_split")
        step_key = f"{data.get('category')}/{data.get('name')}/step_{data.get('step_id')}"
        for a, b in combinations(instances, 2):
            part_pair = norm_pair(inst_to_part[str(a["id"])], inst_to_part[str(b["id"])])
            xs.append(pair_features(a, b))
            ys.append(1 if part_pair in gold else 0)
            metas.append({"split": split, "step_key": step_key, "part_pair": part_pair, "gold_pairs": sorted(gold), "num_gold": len(gold)})
    return np.vstack(xs), np.asarray(ys, dtype=float), metas


def standardize(train_x: np.ndarray, all_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std < 1e-8] = 1.0
    return (all_x - mean) / std, mean, std


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))


def train_logreg(x: np.ndarray, y: np.ndarray, epochs: int, lr: float, l2: float, seed: int) -> Tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 0.01, size=x.shape[1])
    b = 0.0
    pos = y.sum()
    neg = len(y) - pos
    pos_weight = neg / max(pos, 1.0)
    weights = np.where(y > 0.5, pos_weight, 1.0)
    weights = weights / weights.mean()
    for _ in range(epochs):
        pred = sigmoid(x @ w + b)
        err = (pred - y) * weights
        w -= lr * (x.T @ err / len(y) + l2 * w)
        b -= lr * float(err.mean())
    return w, b


def pair_metrics(probs: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    pred = probs >= 0.5
    gold = y > 0.5
    tp = int(np.logical_and(pred, gold).sum())
    fp = int(np.logical_and(pred, ~gold).sum())
    fn = int(np.logical_and(~pred, gold).sum())
    tn = int(np.logical_and(~pred, ~gold).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"accuracy": (tp + tn) / len(y), "precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def step_topk_metrics(probs: np.ndarray, metas: List[Dict[str, Any]], mask: np.ndarray) -> Dict[str, float]:
    groups: Dict[str, List[Tuple[float, Pair, set[Pair], int]]] = defaultdict(list)
    for idx in np.where(mask)[0]:
        m = metas[int(idx)]
        groups[m["step_key"]].append((float(probs[idx]), m["part_pair"], set(map(tuple, m["gold_pairs"])), int(m["num_gold"])))
    tp = fp = fn = exact = 0
    for items in groups.values():
        k = items[0][3]
        gold = items[0][2]
        pred = {pair for _, pair, _, _ in sorted(items, key=lambda item: item[0], reverse=True)[:k]}
        tp += len(pred & gold)
        fp += len(pred - gold)
        fn += len(gold - pred)
        exact += int(pred == gold)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"steps": len(groups), "precision": precision, "recall": recall, "f1": f1, "exact_match": exact / len(groups), "tp": tp, "fp": fp, "fn": fn}


def main() -> None:
    args = parse_args()
    alignment = load_mask_alignment(Path(args.alignment_file))
    x, y, metas = load_examples(Path(args.simplified_root), alignment)
    splits = np.asarray([m["split"] for m in metas])
    train_mask = splits == "train"
    test_mask = splits == "test"
    x_scaled, mean, std = standardize(x[train_mask], x)
    w, b = train_logreg(x_scaled[train_mask], y[train_mask], args.epochs, args.lr, args.l2, args.seed)
    probs = 1.0 / (1.0 + np.exp(-np.clip(x_scaled @ w + b, -40.0, 40.0)))
    report = {
        "feature_source": "simplified_svg",
        "num_pair_examples": int(len(y)),
        "num_train_pairs": int(train_mask.sum()),
        "num_test_pairs": int(test_mask.sum()),
        "positive_rate_train": float(y[train_mask].mean()),
        "positive_rate_test": float(y[test_mask].mean()),
        "pair_metrics_train": pair_metrics(probs[train_mask], y[train_mask]),
        "pair_metrics_test": pair_metrics(probs[test_mask], y[test_mask]),
        "step_topk_train": step_topk_metrics(probs, metas, train_mask),
        "step_topk_test": step_topk_metrics(probs, metas, test_mask),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez(args.model_output, weights=w, bias=b, mean=mean, std=std)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote {out}")
    print(f"Wrote {args.model_output}")


if __name__ == "__main__":
    main()
