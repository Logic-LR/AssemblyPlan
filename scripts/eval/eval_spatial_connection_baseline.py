#!/usr/bin/env python3
"""Evaluate simple spatial baselines for step-level connection prediction.

This is a sanity check for SVG features. It assumes step-level visual instances
can be aligned to `gt.parts` by a deterministic ordering and predicts pairwise
connections from SVG spatial relations.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple


Pair = Tuple[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", default="svg_features", help="Directory created by build_svg_features.py.")
    parser.add_argument("--output", default="svg_features/spatial_baseline_report.json", help="Optional report JSON path.")
    parser.add_argument("--require-count-match", action="store_true", default=True, help="Skip steps where instance count != gt part count.")
    parser.add_argument("--include-count-mismatch", action="store_false", dest="require_count_match", help="Evaluate even when counts differ by truncating to the shorter list.")
    return parser.parse_args()


def iter_feature_files(root: Path) -> Iterable[Path]:
    yield from sorted(root.glob("*/*/step_*.json"))


def norm_pair(a: Any, b: Any) -> Pair:
    aa, bb = str(a), str(b)
    return (aa, bb) if aa <= bb else (bb, aa)


def gt_pairs(gt_connections: Sequence[Sequence[Any]]) -> set[Pair]:
    pairs: set[Pair] = set()
    for conn in gt_connections:
        if len(conn) != 2:
            continue
        pairs.add(norm_pair(conn[0], conn[1]))
    return pairs


def relation_key(a: str, b: str) -> Pair:
    return norm_pair(a, b)


def get_relation_map(data: Dict[str, Any]) -> Dict[Pair, Dict[str, Any]]:
    out: Dict[Pair, Dict[str, Any]] = {}
    for rel in data.get("spatial_relations", []):
        out[relation_key(rel["a"], rel["b"])] = rel
    return out


def align_instances(instances: List[Dict[str, Any]], gt_parts: List[Any], mode: str) -> Dict[str, str]:
    if mode == "feature_order":
        ordered = instances
    elif mode == "x_center":
        ordered = sorted(instances, key=lambda inst: ((inst.get("center") or [0, 0])[0], (inst.get("center") or [0, 0])[1]))
    elif mode == "y_center":
        ordered = sorted(instances, key=lambda inst: ((inst.get("center") or [0, 0])[1], (inst.get("center") or [0, 0])[0]))
    elif mode == "area_desc":
        def area(inst: Dict[str, Any]) -> float:
            box = inst.get("bbox") or [0, 0, 0, 0]
            return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
        ordered = sorted(instances, key=area, reverse=True)
    else:
        raise ValueError(f"Unknown align mode: {mode}")
    return {str(inst["id"]): str(part) for inst, part in zip(ordered, gt_parts)}


def all_candidate_pairs(instances: Sequence[Dict[str, Any]], inst_to_part: Dict[str, str]) -> List[Tuple[str, str, Pair]]:
    out: List[Tuple[str, str, Pair]] = []
    for a, b in combinations([str(inst["id"]) for inst in instances if str(inst["id"]) in inst_to_part], 2):
        out.append((a, b, norm_pair(inst_to_part[a], inst_to_part[b])))
    return out


def score_relation(rel: Dict[str, Any], method: str) -> float:
    overlap = float(rel.get("bbox_overlap_area") or 0.0)
    bbox_dist = float(rel.get("bbox_distance") or 0.0)
    point_dist = rel.get("min_sampled_point_distance")
    point_dist = float(point_dist) if point_dist is not None else 1e9
    if method == "min_point_distance":
        return -point_dist
    if method == "bbox_overlap":
        return overlap
    if method == "hybrid":
        return overlap - 25.0 * point_dist - 10.0 * bbox_dist
    raise ValueError(f"Unknown scoring method: {method}")


def predict_topk(
    data: Dict[str, Any],
    inst_to_part: Dict[str, str],
    k: int,
    method: str,
) -> set[Pair]:
    instances = data.get("instances", [])
    rel_map = get_relation_map(data)
    scored: List[Tuple[float, Pair]] = []
    for a, b, part_pair in all_candidate_pairs(instances, inst_to_part):
        rel = rel_map.get(relation_key(a, b))
        if rel is None:
            continue
        scored.append((score_relation(rel, method), part_pair))
    scored.sort(key=lambda item: item[0], reverse=True)
    return {pair for _, pair in scored[:k]}


def predict_hub(inst_to_part: Dict[str, str], gt_parts: List[Any]) -> set[Pair]:
    if len(gt_parts) < 2:
        return set()
    # Connect all other parts to the last part. This is intentionally simple:
    # many IKEA steps add several pieces onto a carried/central component.
    hub = str(gt_parts[-1])
    return {norm_pair(part, hub) for part in map(str, gt_parts) if str(part) != hub}


def update_counts(counts: Dict[str, int], pred: set[Pair], gold: set[Pair]) -> None:
    counts["tp"] += len(pred & gold)
    counts["fp"] += len(pred - gold)
    counts["fn"] += len(gold - pred)
    counts["exact"] += int(pred == gold)
    counts["steps"] += 1


def metrics(counts: Dict[str, int]) -> Dict[str, float]:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "steps": counts["steps"],
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": counts["exact"] / counts["steps"] if counts["steps"] else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def main() -> None:
    args = parse_args()
    root = Path(args.feature_root)
    align_modes = ["feature_order", "x_center", "y_center", "area_desc"]
    methods = ["min_point_distance", "bbox_overlap", "hybrid"]
    results: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    skipped_count_mismatch = 0
    skipped_no_gold = 0
    total = 0

    for path in iter_feature_files(root):
        total += 1
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        gt = data.get("gt") or {}
        gt_parts = list(gt.get("parts") or [])
        gold = gt_pairs(gt.get("connections") or [])
        if not gold:
            skipped_no_gold += 1
            continue
        instances = data.get("instances") or []
        if args.require_count_match and len(instances) != len(gt_parts):
            skipped_count_mismatch += 1
            continue
        if len(instances) < 2 or len(gt_parts) < 2:
            continue

        usable_gt_parts = gt_parts[: len(instances)]
        k = len(gold)
        for align in align_modes:
            inst_to_part = align_instances(instances, usable_gt_parts, align)
            hub_pred = predict_hub(inst_to_part, usable_gt_parts)
            update_counts(results[f"hub_last/{align}"], hub_pred, gold)
            for method in methods:
                pred = predict_topk(data, inst_to_part, k, method)
                update_counts(results[f"{method}/{align}"], pred, gold)

    report = {
        "total_feature_files": total,
        "skipped_no_gold_connections": skipped_no_gold,
        "skipped_count_mismatch": skipped_count_mismatch,
        "evaluated": {name: metrics(counts) for name, counts in sorted(results.items())},
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
