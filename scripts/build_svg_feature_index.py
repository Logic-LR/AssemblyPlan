#!/usr/bin/env python3
"""Build a JSONL index for generated SVG feature files."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", default="svg_features", help="Directory created by build_svg_features.py.")
    parser.add_argument("--output", default="svg_features/index.jsonl", help="Output JSONL path.")
    parser.add_argument("--summary", default="svg_features/index_summary.json", help="Output summary JSON path.")
    return parser.parse_args()


def iter_feature_files(root: Path) -> Iterable[Path]:
    yield from sorted(root.glob("*/*/step_*.json"))


def make_record(path: Path, root: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    gt = data.get("gt") or {}
    instances = data.get("instances", [])
    parts = gt.get("parts") or []
    connections = gt.get("connections") or []
    return {
        "category": data.get("category"),
        "name": data.get("name"),
        "step_id": data.get("step_id"),
        "feature_path": str(path.relative_to(root.parent).as_posix()),
        "source_svg": data.get("source_svg"),
        "num_instances": len(instances),
        "instance_ids": [inst.get("id") for inst in instances],
        "num_gt_parts": len(parts),
        "gt_parts": parts,
        "num_gt_connections": len(connections),
        "gt_connections": connections,
        "instance_part_count_match": len(instances) == len(parts),
        "split": gt.get("part_segmentation_split"),
        "step_id_global": gt.get("step_id_global"),
    }


def main() -> None:
    args = parse_args()
    root = Path(args.feature_root)
    output = Path(args.output)
    summary_path = Path(args.summary)
    output.parent.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = [make_record(path, root) for path in iter_feature_files(root)]

    with output.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    matches = sum(1 for r in records if r["instance_part_count_match"])
    splits = Counter(r.get("split") for r in records)
    by_category = Counter(r.get("category") for r in records)
    instance_counts = Counter(r["num_instances"] for r in records)
    gt_part_counts = Counter(r["num_gt_parts"] for r in records)
    mismatch_examples = [
        {
            "feature_path": r["feature_path"],
            "num_instances": r["num_instances"],
            "instance_ids": r["instance_ids"],
            "num_gt_parts": r["num_gt_parts"],
            "gt_parts": r["gt_parts"],
        }
        for r in records
        if not r["instance_part_count_match"]
    ][:25]

    summary = {
        "num_records": len(records),
        "instance_part_count_matches": matches,
        "instance_part_count_match_rate": matches / len(records) if records else 0.0,
        "splits": dict(splits),
        "categories": dict(by_category),
        "num_instances_histogram": dict(sorted(instance_counts.items())),
        "num_gt_parts_histogram": dict(sorted(gt_part_counts.items())),
        "mismatch_examples": mismatch_examples,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Wrote {output}")
    print(f"Wrote {summary_path}")
    print(f"records={len(records)} match_rate={summary['instance_part_count_match_rate']:.4f}")


if __name__ == "__main__":
    main()
