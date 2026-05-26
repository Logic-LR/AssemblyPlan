#!/usr/bin/env python3
"""Evaluate a manual/RAG-context decoder from composite SVG prototype tokens."""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

from eval.evaluate_paper_tree_metrics import (
    average_metrics,
    build_tree_from_list,
    eval_tree,
    step_tree_from_child_specs,
)
from export.export_tree_predictions_and_equivalence_report import tree_to_list
from train.train_tree_planner_baseline import cluster_token, split_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="experiments/svg_assembly/datasets/tree_generation_dataset.json")
    parser.add_argument("--output", default="experiments/svg_assembly/reports/tree_planner_composite_context_report.json")
    parser.add_argument("--pred-output-dir", default="experiments/svg_assembly/tree_planner_predictions_composite_context_test")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def token_part_set(token: Dict[str, Any], n_parts: int) -> frozenset[int]:
    parts = frozenset(int(part) for part in token.get("part_ids") or [])
    return frozenset(part for part in parts if 0 <= part < n_parts)


def maximal_children(parent: frozenset[int], known: Sequence[frozenset[int]]) -> List[frozenset[int]]:
    candidates = [cluster for cluster in known if cluster < parent]
    selected: List[frozenset[int]] = []
    used: set[int] = set()
    for cluster in sorted(candidates, key=lambda item: (len(item), tuple(sorted(item))), reverse=True):
        if cluster & used:
            continue
        selected.append(cluster)
        used.update(cluster)
    for part in sorted(parent - frozenset(used)):
        selected.append(frozenset([part]))
    return sorted(selected, key=lambda item: (len(item), tuple(sorted(item))))


def decode_from_composite_context(record: Dict[str, Any]) -> Any:
    n_parts = int(record["num_parts"])
    root = frozenset(range(n_parts))
    known: set[frozenset[int]] = {frozenset([part]) for part in range(n_parts)}
    child_specs: List[List[str]] = []
    composite_sets = {
        token_part_set(token, n_parts)
        for token in record.get("composite_tokens") or []
    }
    for part_set in sorted(composite_sets, key=lambda item: (len(item), tuple(sorted(item)))):
        if len(part_set) <= 1 or part_set in known or not part_set <= root:
            continue
        children = maximal_children(part_set, list(known))
        if len(children) < 2:
            continue
        child_specs.append([cluster_token(child) for child in children])
        known.add(part_set)
    if root not in known:
        children = maximal_children(root, list(known))
        if len(children) >= 2:
            child_specs.append([cluster_token(child) for child in children])
    return step_tree_from_child_specs(child_specs, n_parts)


def evaluate_records(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    rows = []
    examples = []
    for record in records:
        gt = build_tree_from_list(record["assembly_tree"])
        pred = decode_from_composite_context(record)
        metrics = eval_tree(gt, pred)
        rows.append(metrics)
        if len(examples) < 10:
            examples.append({"category": record["category"], "name": record["name"], "metrics": metrics})
    return {"objects": len(records), "metrics": average_metrics(rows), "examples": examples}


def export_predictions(records: Sequence[Dict[str, Any]], output_dir: Path) -> List[Dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    exported = []
    for record in records:
        gt = build_tree_from_list(record["assembly_tree"])
        pred = decode_from_composite_context(record)
        metrics = eval_tree(gt, pred)
        obj_dir = output_dir / record["category"] / record["name"]
        obj_dir.mkdir(parents=True, exist_ok=True)
        path = obj_dir / "assembly_tree_prediction.json"
        path.write_text(
            json.dumps(
                {
                    "category": record["category"],
                    "name": record["name"],
                    "split": record.get("split"),
                    "feature_mode": "composite_context",
                    "predicted_assembly_tree": tree_to_list(pred),
                    "ground_truth_assembly_tree": record["assembly_tree"],
                    "paper_tree_metrics": metrics,
                    "part_tokens": record["part_tokens"],
                    "composite_tokens": record.get("composite_tokens") or [],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        exported.append(
            {
                "category": record["category"],
                "name": record["name"],
                "split": record.get("split"),
                "path": str(path.as_posix()),
                "paper_tree_metrics": metrics,
            }
        )
    (output_dir / "index.json").write_text(json.dumps(exported, ensure_ascii=False, indent=2), encoding="utf-8")
    return exported


def main() -> None:
    args = parse_args()
    records = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    fit_records, val_records, test_records = split_records(records, args.val_fraction, args.seed)
    exported = export_predictions(test_records, Path(args.pred_output_dir))
    report = {
        "model": "composite_context_decoder",
        "feature_mode": "composite_context",
        "dataset": args.dataset,
        "splits": {
            "fit_objects": len(fit_records),
            "val_objects": len(val_records),
            "test_objects": len(test_records),
            "all_objects": len(records),
        },
        "tree_metrics": {
            "train": evaluate_records(fit_records),
            "val": evaluate_records(val_records) if val_records else None,
            "test": evaluate_records(test_records),
            "all": evaluate_records(records),
        },
        "exported_test_predictions": {
            "output_dir": args.pred_output_dir,
            "num_objects": len(exported),
        },
        "notes": [
            "This is a manual/RAG-context upper-bound decoder.",
            "It uses composite SVG prototype part sets from manual middle steps, so it is not pure observed-part inference.",
            "Use it to estimate how useful explicit subassembly context could be if a retrieval or prediction module supplies it.",
        ],
        "config": vars(args),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote {out}")
    print(f"Wrote {Path(args.pred_output_dir) / 'index.json'}")


if __name__ == "__main__":
    main()
