#!/usr/bin/env python3
"""Train a lightweight geometric part-to-SVG grounding baseline."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


SHAPE_TYPES = ["elongated_bar", "plate_like", "irregular", "point_or_line"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", default="experiments/svg_assembly/datasets/grounding_samples.jsonl")
    parser.add_argument("--primitive-only", action="store_true", help="Train/evaluate only primitive samples.")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="experiments/svg_assembly/reports/grounding_geometric_report.json")
    parser.add_argument("--model-output", default="experiments/svg_assembly/reports/grounding_geometric_model.npz")
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def safe(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def polygon_area(points: Sequence[Sequence[float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for i, p in enumerate(points):
        q = points[(i + 1) % len(points)]
        area += safe(p[0]) * safe(q[1]) - safe(q[0]) * safe(p[1])
    return abs(area) / 2.0


def part_sort_key(value: str) -> Tuple[int, Any]:
    return (0, int(value)) if value.isdigit() else (1, value)


def split_part_id(part_id: str) -> List[str]:
    return sorted([p.strip() for p in part_id.split(",") if p.strip()], key=part_sort_key)


def build_part_lookup(samples: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    lookup: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for sample in samples:
        for part in sample.get("positive_primitive_parts", []):
            lookup[(part["category"], part["name"], str(part["part_id"]))] = part
    return lookup


def aggregate_part_features(category: str, name: str, part_id: str, part_lookup: Dict[Tuple[str, str, str], Dict[str, Any]]) -> np.ndarray:
    primitive_ids = split_part_id(part_id)
    parts = [part_lookup[(category, name, pid)] for pid in primitive_ids if (category, name, pid) in part_lookup]
    if not parts:
        return np.zeros(30, dtype=float)
    extents = np.asarray([p.get("extent", [0.0, 0.0, 0.0]) for p in parts], dtype=float)
    extents = np.maximum(extents, 1e-9)
    sorted_ext = np.sort(extents, axis=1)[:, ::-1]
    ratios = np.stack(
        [
            sorted_ext[:, 0] / sorted_ext[:, 1],
            sorted_ext[:, 1] / sorted_ext[:, 2],
            sorted_ext[:, 0] / sorted_ext[:, 2],
        ],
        axis=1,
    )
    faces = np.asarray([safe(p.get("num_faces")) for p in parts], dtype=float)
    verts = np.asarray([safe(p.get("num_vertices")) for p in parts], dtype=float)
    mean_ext = extents.mean(axis=0)
    max_ext = extents.max(axis=0)
    sum_ext = extents.sum(axis=0)
    mean_sorted = sorted_ext.mean(axis=0)
    max_sorted = sorted_ext.max(axis=0)
    mean_ratios = ratios.mean(axis=0)
    max_ratios = ratios.max(axis=0)
    volume_proxy = np.prod(extents, axis=1)
    num_parts = len(parts)
    return np.asarray(
        [
            num_parts,
            math.log1p(num_parts),
            *mean_ext.tolist(),
            *max_ext.tolist(),
            *sum_ext.tolist(),
            *mean_sorted.tolist(),
            *max_sorted.tolist(),
            *mean_ratios.tolist(),
            *max_ratios.tolist(),
            float(volume_proxy.mean()),
            float(volume_proxy.max()),
            math.log1p(float(faces.mean())),
            math.log1p(float(faces.max())),
            math.log1p(float(verts.mean())),
            math.log1p(float(verts.max())),
            1.0 if mean_ratios[0] > 8 else 0.0,
            1.0 if mean_ratios[1] > 4 else 0.0,
        ],
        dtype=float,
    )


def svg_features(inst: Dict[str, Any]) -> np.ndarray:
    box = inst.get("bbox") or [0, 0, 0, 0]
    bw = max(0.0, safe(box[2]) - safe(box[0]))
    bh = max(0.0, safe(box[3]) - safe(box[1]))
    canvas_area = 793.701 * 1122.52
    diag = math.hypot(793.701, 1122.52)
    poly = inst.get("simplified_polygon") or []
    hull = inst.get("convex_hull") or []
    onehot = [1.0 if inst.get("shape_type") == s else 0.0 for s in SHAPE_TYPES]
    axis_length = safe(inst.get("axis_length"))
    axis_width = safe(inst.get("axis_width"))
    aspect = axis_length / max(axis_width, 1e-9)
    return np.asarray(
        [
            bw / 793.701,
            bh / 1122.52,
            (bw * bh) / canvas_area,
            polygon_area(poly) / canvas_area,
            polygon_area(hull) / canvas_area,
            axis_length / diag,
            axis_width / diag,
            math.log1p(safe(inst.get("elongation"))),
            math.log1p(aspect),
            len(poly) / 12.0,
            len(hull) / 32.0,
            *onehot,
        ],
        dtype=float,
    )


def pair_features(part_feat: np.ndarray, svg_feat: np.ndarray) -> np.ndarray:
    # Keep this simple and interpretable: raw features plus coarse comparisons
    # between 3D part ratios and 2D SVG ratios.
    comparisons = []
    # Indices in part feature for mean/max sorted extents and ratios.
    part_longness = part_feat[17] if len(part_feat) > 17 else 0.0
    part_flatness = part_feat[18] if len(part_feat) > 18 else 0.0
    svg_longness = svg_feat[8] if len(svg_feat) > 8 else 0.0
    comparisons.extend(
        [
            abs(part_longness - svg_longness),
            abs(math.log1p(part_flatness) - svg_longness),
            part_feat[0] * svg_feat[5],
            part_feat[-2] * svg_feat[-4],  # long-ish part with elongated_bar flag
            part_feat[-1] * svg_feat[-3],  # flat-ish part with plate_like flag
        ]
    )
    return np.concatenate([part_feat, svg_feat, np.asarray(comparisons, dtype=float)])


def step_key(sample: Dict[str, Any]) -> str:
    return f"{sample['category']}/{sample['name']}/step_{sample['step_id']}"


def build_pair_examples(samples: Sequence[Dict[str, Any]], primitive_only: bool) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    if primitive_only:
        samples = [s for s in samples if not s.get("is_composite")]
    part_lookup = build_part_lookup(samples)
    by_step: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_step[step_key(sample)].append(sample)

    xs: List[np.ndarray] = []
    ys: List[int] = []
    metas: List[Dict[str, Any]] = []
    for key, step_samples in by_step.items():
        candidate_ids = sorted({str(s["positive_part_id"]) for s in step_samples}, key=part_sort_key)
        for sample in step_samples:
            svg_feat = svg_features(sample["svg_simplified"])
            for candidate_id in candidate_ids:
                part_feat = aggregate_part_features(sample["category"], sample["name"], candidate_id, part_lookup)
                xs.append(pair_features(part_feat, svg_feat))
                ys.append(1 if candidate_id == str(sample["positive_part_id"]) else 0)
                metas.append(
                    {
                        "step_key": key,
                        "split": sample.get("split"),
                        "svg_instance_id": sample["svg_instance_id"],
                        "candidate_part_id": candidate_id,
                        "positive_part_id": str(sample["positive_part_id"]),
                    }
                )
    return np.vstack(xs), np.asarray(ys, dtype=float), metas


def standardize(train_x: np.ndarray, all_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std < 1e-8] = 1.0
    return (all_x - mean) / std, mean, std


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def train_logreg(x: np.ndarray, y: np.ndarray, epochs: int, lr: float, l2: float, seed: int) -> Tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.01, size=x.shape[1])
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


def solve_assignment(score_matrix: np.ndarray) -> List[int]:
    n, m = score_matrix.shape
    if n == 0 or m == 0:
        return []
    # DP over candidate masks. n and m are <= 11 in this dataset.
    dp: Dict[int, Tuple[float, List[int]]] = {0: (0.0, [])}
    for row in range(n):
        nxt: Dict[int, Tuple[float, List[int]]] = {}
        for mask, (score, assign) in dp.items():
            for col in range(m):
                if mask & (1 << col):
                    continue
                new_mask = mask | (1 << col)
                new_score = score + float(score_matrix[row, col])
                if new_mask not in nxt or new_score > nxt[new_mask][0]:
                    nxt[new_mask] = (new_score, assign + [col])
        dp = nxt
    return max(dp.values(), key=lambda item: item[0])[1]


def assignment_metrics(probs: np.ndarray, metas: List[Dict[str, Any]], mask: np.ndarray) -> Dict[str, float]:
    by_step: Dict[str, List[int]] = defaultdict(list)
    for idx in np.where(mask)[0]:
        by_step[metas[int(idx)]["step_key"]].append(int(idx))
    correct_instances = total_instances = exact = 0
    for indices in by_step.values():
        svg_ids = sorted({metas[i]["svg_instance_id"] for i in indices})
        cand_ids = sorted({metas[i]["candidate_part_id"] for i in indices}, key=part_sort_key)
        row_of = {sid: r for r, sid in enumerate(svg_ids)}
        col_of = {pid: c for c, pid in enumerate(cand_ids)}
        scores = np.full((len(svg_ids), len(cand_ids)), -1e9, dtype=float)
        gold: Dict[str, str] = {}
        for i in indices:
            r = row_of[metas[i]["svg_instance_id"]]
            c = col_of[metas[i]["candidate_part_id"]]
            scores[r, c] = probs[i]
            gold[metas[i]["svg_instance_id"]] = metas[i]["positive_part_id"]
        assignment = solve_assignment(scores)
        pred = {svg_ids[r]: cand_ids[c] for r, c in enumerate(assignment)}
        step_correct = sum(1 for sid, pid in pred.items() if gold.get(sid) == pid)
        correct_instances += step_correct
        total_instances += len(svg_ids)
        exact += int(step_correct == len(svg_ids))
    return {
        "steps": len(by_step),
        "instance_accuracy": correct_instances / total_instances if total_instances else 0.0,
        "exact_match": exact / len(by_step) if by_step else 0.0,
        "correct_instances": correct_instances,
        "total_instances": total_instances,
    }


def main() -> None:
    args = parse_args()
    samples = load_jsonl(Path(args.samples))
    x, y, metas = build_pair_examples(samples, args.primitive_only)
    splits = np.asarray([m["split"] for m in metas])
    train_mask = splits == "train"
    test_mask = splits == "test"
    x_scaled, mean, std = standardize(x[train_mask], x)
    w, b = train_logreg(x_scaled[train_mask], y[train_mask], args.epochs, args.lr, args.l2, args.seed)
    probs = sigmoid(x_scaled @ w + b)
    report = {
        "primitive_only": args.primitive_only,
        "num_pair_examples": int(len(y)),
        "num_train_pairs": int(train_mask.sum()),
        "num_test_pairs": int(test_mask.sum()),
        "positive_rate_train": float(y[train_mask].mean()),
        "positive_rate_test": float(y[test_mask].mean()),
        "pair_metrics_train": pair_metrics(probs[train_mask], y[train_mask]),
        "pair_metrics_test": pair_metrics(probs[test_mask], y[test_mask]),
        "assignment_train": assignment_metrics(probs, metas, train_mask),
        "assignment_test": assignment_metrics(probs, metas, test_mask),
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
