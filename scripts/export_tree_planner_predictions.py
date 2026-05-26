#!/usr/bin/env python3
"""Export predicted assembly trees from a trained tree-planner baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from evaluate_paper_tree_metrics import average_metrics, build_tree_from_list, eval_tree
from export_tree_predictions_and_equivalence_report import tree_to_list
from train_tree_planner_baseline import plan_tree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="experiments/svg_assembly/datasets/tree_generation_dataset.json")
    parser.add_argument("--model", default="experiments/svg_assembly/reports/tree_planner_svg_model.npz")
    parser.add_argument("--split", choices=["train", "test", "all"], default="test")
    parser.add_argument("--output-dir", default="experiments/svg_assembly/tree_planner_predictions_svg")
    parser.add_argument("--output-report", default="experiments/svg_assembly/reports/tree_planner_svg_predictions_report.json")
    return parser.parse_args()


def model_value(model: Dict[str, Any], key: str) -> Any:
    value = model[key]
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def selected_records(records: Sequence[Dict[str, Any]], split: str) -> List[Dict[str, Any]]:
    if split == "all":
        return list(records)
    return [record for record in records if record.get("split") == split]


def main() -> None:
    args = parse_args()
    records = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    model_np = dict(np.load(args.model, allow_pickle=True))
    feature_mode = str(model_value(model_np, "feature_mode"))
    weights = np.asarray(model_np["weights"], dtype=np.float32)
    bias = float(model_value(model_np, "bias"))
    mean = np.asarray(model_np["mean"], dtype=np.float32)
    std = np.asarray(model_np["std"], dtype=np.float32)
    threshold = float(model_value(model_np, "threshold"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    exported = []
    for record in selected_records(records, args.split):
        gt_tree = build_tree_from_list(record["assembly_tree"])
        pred_tree = plan_tree(record, feature_mode, weights, bias, mean, std, threshold)
        metrics = eval_tree(gt_tree, pred_tree)
        rows.append(metrics)
        obj_dir = output_dir / record["category"] / record["name"]
        obj_dir.mkdir(parents=True, exist_ok=True)
        path = obj_dir / "assembly_tree_prediction.json"
        payload = {
            "category": record["category"],
            "name": record["name"],
            "split": record.get("split"),
            "feature_mode": feature_mode,
            "threshold": threshold,
            "predicted_assembly_tree": tree_to_list(pred_tree),
            "ground_truth_assembly_tree": record["assembly_tree"],
            "paper_tree_metrics": metrics,
            "part_tokens": record["part_tokens"],
            "composite_tokens": record.get("composite_tokens") or [],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        exported.append(
            {
                "category": record["category"],
                "name": record["name"],
                "split": record.get("split"),
                "path": str(path.as_posix()),
                "paper_tree_metrics": metrics,
            }
        )

    report = {
        "dataset": args.dataset,
        "model": args.model,
        "split": args.split,
        "feature_mode": feature_mode,
        "threshold": threshold,
        "num_objects": len(exported),
        "paper_tree_metrics": average_metrics(rows),
        "exported_predictions": exported,
        "notes": [
            "This planner consumes object-level primitive part tokens, not per-step SVGs.",
            "The predicted tree is a baseline greedy merge tree.",
        ],
    }
    report_path = Path(args.output_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "index.json").write_text(json.dumps(exported, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote {report_path}")
    print(f"Wrote {output_dir / 'index.json'}")


if __name__ == "__main__":
    main()
