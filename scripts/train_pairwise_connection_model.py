#!/usr/bin/env python3
"""Train a small pairwise connection classifier on SVG features.

This is a deliberately simple numpy baseline. It tests whether the structured
SVG geometry features contain learnable signal for step-level connections.
"""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", default="svg_features", help="Directory created by build_svg_features.py.")
    parser.add_argument("--align", default="mask", choices=["mask", "feature_order", "x_center", "y_center", "area_desc"], help="Visual-instance to gt.parts alignment.")
    parser.add_argument("--alignment-file", default="svg_features/instance_part_alignment.jsonl", help="Mask-derived alignment JSONL from analyze_instance_mask_alignment.py.")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="svg_features/pairwise_connection_report.json")
    parser.add_argument("--model-output", default="svg_features/pairwise_connection_model.npz")
    return parser.parse_args()


def iter_feature_files(root: Path) -> Iterable[Path]:
    yield from sorted(root.glob("*/*/step_*.json"))


def norm_pair(a: Any, b: Any) -> Pair:
    aa, bb = str(a), str(b)
    return (aa, bb) if aa <= bb else (bb, aa)


def gt_pairs(gt_connections: Sequence[Sequence[Any]]) -> set[Pair]:
    out: set[Pair] = set()
    for conn in gt_connections:
        if len(conn) == 2:
            out.add(norm_pair(conn[0], conn[1]))
    return out


def area(inst: Dict[str, Any]) -> float:
    box = inst.get("bbox") or [0, 0, 0, 0]
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def align_instances(instances: List[Dict[str, Any]], gt_parts: List[Any], mode: str) -> Dict[str, str]:
    if mode == "feature_order":
        ordered = instances
    elif mode == "x_center":
        ordered = sorted(instances, key=lambda inst: ((inst.get("center") or [0, 0])[0], (inst.get("center") or [0, 0])[1]))
    elif mode == "y_center":
        ordered = sorted(instances, key=lambda inst: ((inst.get("center") or [0, 0])[1], (inst.get("center") or [0, 0])[0]))
    elif mode == "area_desc":
        ordered = sorted(instances, key=area, reverse=True)
    else:
        raise ValueError(mode)
    return {str(inst["id"]): str(part) for inst, part in zip(ordered, gt_parts)}


def load_mask_alignment(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    out: Dict[str, Dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            out[record["feature_path"]] = {str(k): str(v) for k, v in record["stroke_to_part"].items()}
    return out


def relation_map(data: Dict[str, Any]) -> Dict[Pair, Dict[str, Any]]:
    out: Dict[Pair, Dict[str, Any]] = {}
    for rel in data.get("spatial_relations", []):
        out[norm_pair(rel["a"], rel["b"])] = rel
    return out


def safe(v: Any, default: float = 0.0) -> float:
    if v is None:
        return default
    try:
        value = float(v)
    except (TypeError, ValueError):
        return default
    if math.isnan(value) or math.isinf(value):
        return default
    return value


def instance_basic(inst: Dict[str, Any], canvas: Dict[str, Any]) -> np.ndarray:
    width = safe(canvas.get("width"), 1.0) or 1.0
    height = safe(canvas.get("height"), 1.0) or 1.0
    diag = math.hypot(width, height)
    box = inst.get("bbox") or [0.0, 0.0, 0.0, 0.0]
    center = inst.get("center") or [(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0]
    bw = max(0.0, safe(box[2]) - safe(box[0]))
    bh = max(0.0, safe(box[3]) - safe(box[1]))
    return np.asarray(
        [
            safe(center[0]) / width,
            safe(center[1]) / height,
            bw / width,
            bh / height,
            (bw * bh) / (width * height),
            safe(inst.get("axis_length")) / diag,
            safe(inst.get("axis_width")) / diag,
            math.log1p(safe(inst.get("elongation"))),
            math.log1p(safe(inst.get("num_paths"))),
            math.log1p(safe(inst.get("num_points"))),
            math.log1p(safe(inst.get("total_polyline_length"))) / 10.0,
        ],
        dtype=float,
    )


def pair_features(data: Dict[str, Any], a: Dict[str, Any], b: Dict[str, Any]) -> np.ndarray:
    canvas = data.get("canvas") or {}
    width = safe(canvas.get("width"), 1.0) or 1.0
    height = safe(canvas.get("height"), 1.0) or 1.0
    diag = math.hypot(width, height)
    fa = instance_basic(a, canvas)
    fb = instance_basic(b, canvas)
    rel = relation_map(data).get(norm_pair(a["id"], b["id"]), {})
    overlap = safe(rel.get("bbox_overlap_area"))
    bbox_dist = safe(rel.get("bbox_distance"))
    point_dist = safe(rel.get("min_sampled_point_distance"), diag)
    delta = rel.get("center_delta") or [0.0, 0.0]
    relation = np.asarray(
        [
            overlap / (width * height),
            math.log1p(overlap) / 10.0,
            bbox_dist / diag,
            point_dist / diag,
            safe(delta[0]) / width,
            safe(delta[1]) / height,
            abs(safe(delta[0])) / width,
            abs(safe(delta[1])) / height,
            1.0 if rel.get("likely_contact") else 0.0,
        ],
        dtype=float,
    )
    return np.concatenate([np.minimum(fa, fb), np.maximum(fa, fb), np.abs(fa - fb), relation])


def load_examples(root: Path, align: str, mask_alignment: Dict[str, Dict[str, str]]) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    xs: List[np.ndarray] = []
    ys: List[int] = []
    meta: List[Dict[str, Any]] = []
    for path in iter_feature_files(root):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        gt = data.get("gt") or {}
        gt_parts = list(gt.get("parts") or [])
        gold = gt_pairs(gt.get("connections") or [])
        instances = data.get("instances") or []
        if len(instances) != len(gt_parts) or len(instances) < 2:
            continue
        if align == "mask":
            inst_to_part = mask_alignment.get(str(path.as_posix()), {})
            if set(inst_to_part) != {str(inst["id"]) for inst in instances}:
                continue
        else:
            inst_to_part = align_instances(instances, gt_parts, align)
        for ia, ib in combinations(instances, 2):
            part_pair = norm_pair(inst_to_part[str(ia["id"])], inst_to_part[str(ib["id"])])
            xs.append(pair_features(data, ia, ib))
            ys.append(1 if part_pair in gold else 0)
            meta.append(
                {
                    "feature_path": str(path.as_posix()),
                    "split": gt.get("part_segmentation_split"),
                    "step_key": f"{data.get('category')}/{data.get('name')}/step_{data.get('step_id')}",
                    "part_pair": part_pair,
                    "gold_pairs": sorted(gold),
                    "num_gold": len(gold),
                }
            )
    return np.vstack(xs), np.asarray(ys, dtype=float), meta


def standardize(train_x: np.ndarray, all_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std < 1e-8] = 1.0
    return (all_x - mean) / std, mean, std


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


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
        grad_w = x.T @ err / len(y) + l2 * w
        grad_b = float(err.mean())
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def pair_metrics(probs: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    pred = probs >= threshold
    gold = y > 0.5
    tp = int(np.logical_and(pred, gold).sum())
    fp = int(np.logical_and(pred, ~gold).sum())
    fn = int(np.logical_and(~pred, gold).sum())
    tn = int(np.logical_and(~pred, ~gold).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    acc = (tp + tn) / len(y) if len(y) else 0.0
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def step_topk_metrics(probs: np.ndarray, metas: List[Dict[str, Any]], split_mask: np.ndarray) -> Dict[str, float]:
    groups: Dict[str, List[Tuple[float, Pair, set[Pair], int]]] = defaultdict(list)
    selected = np.where(split_mask)[0]
    for idx in selected:
        meta = metas[idx]
        groups[meta["step_key"]].append((float(probs[idx]), meta["part_pair"], set(map(tuple, meta["gold_pairs"])), int(meta["num_gold"])))
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
    return {
        "steps": len(groups),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": exact / len(groups) if groups else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def main() -> None:
    args = parse_args()
    mask_alignment = load_mask_alignment(Path(args.alignment_file)) if args.align == "mask" else {}
    x, y, meta = load_examples(Path(args.feature_root), args.align, mask_alignment)
    splits = np.asarray([m["split"] for m in meta])
    train_mask = splits == "train"
    test_mask = splits == "test"
    x_scaled, mean, std = standardize(x[train_mask], x)
    w, b = train_logreg(x_scaled[train_mask], y[train_mask], args.epochs, args.lr, args.l2, args.seed)
    probs = sigmoid(x_scaled @ w + b)
    report = {
        "align": args.align,
        "num_pair_examples": int(len(y)),
        "num_train_pairs": int(train_mask.sum()),
        "num_test_pairs": int(test_mask.sum()),
        "positive_rate_train": float(y[train_mask].mean()),
        "positive_rate_test": float(y[test_mask].mean()),
        "pair_metrics_train": pair_metrics(probs[train_mask], y[train_mask]),
        "pair_metrics_test": pair_metrics(probs[test_mask], y[test_mask]),
        "step_topk_train": step_topk_metrics(probs, meta, train_mask),
        "step_topk_test": step_topk_metrics(probs, meta, test_mask),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    np.savez(args.model_output, weights=w, bias=b, mean=mean, std=std)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote {output}")
    print(f"Wrote {args.model_output}")


if __name__ == "__main__":
    main()
