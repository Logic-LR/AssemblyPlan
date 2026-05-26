#!/usr/bin/env python3
"""Summarize low-label / weak-supervision tree-planner reports."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reports",
        nargs="*",
        default=None,
        help="Report JSON files. Defaults to experiments/svg_assembly/reports/label_ratio_*.json",
    )
    parser.add_argument(
        "--glob",
        default="experiments/svg_assembly/reports/label_ratio_*.json",
        help="Glob used when --reports is omitted.",
    )
    parser.add_argument(
        "--output-json",
        default="experiments/svg_assembly/reports/label_ratio_summary.json",
    )
    parser.add_argument(
        "--output-md",
        default="experiments/svg_assembly/reports/label_ratio_summary.md",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def nested(report: Dict[str, Any], *keys: str) -> Any:
    cur: Any = report
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def infer_label_ratio(report: Dict[str, Any]) -> Optional[float]:
    return (
        nested(report, "splits", "label_ratio_requested")
        or nested(report, "splits", "gt_label_ratio_requested")
        or nested(report, "config", "label_ratio")
        or nested(report, "config", "gt_label_ratio")
    )


def infer_method(report: Dict[str, Any], path: Path) -> str:
    model = report.get("model") or path.stem
    if model == "context_aware_merge_mlp":
        entropy = float(nested(report, "config", "entropy_weight") or 0.0)
        return f"BCE context MLP entropy={entropy:g}"
    if model == "grpo_context_mlp":
        weights = nested(report, "grpo_config", "reward_weights") or {}
        if not weights:
            weights = {
                "svg_coherence": nested(report, "config", "reward_svg_coherence"),
                "spatial_svg": nested(report, "config", "reward_spatial_svg"),
                "gt_f1": nested(report, "config", "reward_gt_f1"),
            }
        return (
            "GRPO "
            f"gt={fmt(weights.get('gt_f1'), 2)} "
            f"svg={fmt(weights.get('svg_coherence'), 2)} "
            f"spatial={fmt(weights.get('spatial_svg'), 2)}"
        )
    return str(model)


def row_from_report(path: Path) -> Dict[str, Any]:
    report = load_json(path)
    return {
        "report": str(path.as_posix()),
        "label_ratio": infer_label_ratio(report),
        "method": infer_method(report, path),
        "feature_mode": report.get("feature_mode"),
        "fit_objects": nested(report, "splits", "fit_objects"),
        "labeled_fit_objects": nested(report, "splits", "labeled_fit_objects"),
        "gt_reward_objects": nested(report, "splits", "gt_reward_objects"),
        "val_simple_f1": nested(report, "tree_metrics", "val", "metrics", "simple", "f1"),
        "val_hard_f1": nested(report, "tree_metrics", "val", "metrics", "hard", "f1"),
        "test_simple_f1": nested(report, "tree_metrics", "test", "metrics", "simple", "f1"),
        "test_hard_f1": nested(report, "tree_metrics", "test", "metrics", "hard", "f1"),
        "all_hard_f1": nested(report, "tree_metrics", "all", "metrics", "hard", "f1"),
        "train_entropy": nested(report, "probability_diagnostics", "train", "mean_binary_entropy"),
        "train_confident_0_95": nested(report, "probability_diagnostics", "train", "frac_confident_0_05"),
    }


def markdown_table(rows: List[Dict[str, Any]]) -> str:
    columns = [
        "label_ratio",
        "method",
        "feature_mode",
        "labeled_fit_objects",
        "gt_reward_objects",
        "val_hard_f1",
        "test_simple_f1",
        "test_hard_f1",
        "all_hard_f1",
        "train_entropy",
        "train_confident_0_95",
    ]
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(fmt(row.get(col)) for col in columns) + " |")
    return "\n".join(out)


def main() -> None:
    args = parse_args()
    report_paths = [Path(p) for p in (args.reports or sorted(glob.glob(args.glob)))]
    rows = [row_from_report(path) for path in report_paths if path.exists()]
    rows.sort(key=lambda row: (row.get("label_ratio") is None, row.get("label_ratio") or 0, row.get("method") or ""))

    output = {
        "num_reports": len(rows),
        "reports": [str(path.as_posix()) for path in report_paths],
        "rows": rows,
    }
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text("# Label-Ratio Tree Planner Summary\n\n" + markdown_table(rows) + "\n", encoding="utf-8")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
