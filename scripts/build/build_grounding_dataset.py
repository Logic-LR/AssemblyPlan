#!/usr/bin/env python3
"""Build part-to-SVG-instance grounding samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simplified-root", default="experiments/svg_assembly/simplified_svg")
    parser.add_argument("--alignment-file", default="svg_features/instance_part_alignment.jsonl")
    parser.add_argument("--part-index", default="part_images/part_index.jsonl")
    parser.add_argument("--output", default="experiments/svg_assembly/datasets/grounding_samples.jsonl")
    parser.add_argument("--summary", default="experiments/svg_assembly/datasets/grounding_summary.json")
    return parser.parse_args()


def iter_simplified_files(root: Path) -> Iterable[Path]:
    yield from sorted(root.glob("*/*/step_*/simplified_instances.json"))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_alignment(path: Path) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for record in load_jsonl(path):
        out[record["feature_path"]] = {str(k): str(v) for k, v in record["stroke_to_part"].items()}
    return out


def load_part_index(path: Path) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    out: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for record in load_jsonl(path):
        out[(record["category"], record["name"], str(record["part_id"]))] = record
    return out


def part_sort_key(value: str) -> Tuple[int, Any]:
    return (0, int(value)) if value.isdigit() else (1, value)


def primitive_ids(part_id: str) -> List[str]:
    return sorted([p.strip() for p in part_id.split(",") if p.strip()], key=part_sort_key)


def main() -> None:
    args = parse_args()
    simplified_root = Path(args.simplified_root)
    alignment = load_alignment(Path(args.alignment_file))
    part_index = load_part_index(Path(args.part_index))
    output = Path(args.output)
    summary_path = Path(args.summary)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    samples: List[Dict[str, Any]] = []
    missing_parts: List[Dict[str, Any]] = []
    for path in iter_simplified_files(simplified_root):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        category = data["category"]
        name = data["name"]
        step_id = data["step_id"]
        feature_path = data["source_feature"]
        inst_to_part = alignment.get(feature_path, {})
        all_part_ids = sorted({str(v) for v in inst_to_part.values()}, key=part_sort_key)
        for inst in data.get("instances", []):
            stroke = str(inst["id"])
            part_id = inst_to_part.get(stroke)
            if part_id is None:
                continue
            primitive_part_ids = primitive_ids(str(part_id))
            primitive_parts = [
                part_index[(category, name, pid)]
                for pid in primitive_part_ids
                if (category, name, pid) in part_index
            ]
            is_composite = len(primitive_part_ids) > 1
            part = primitive_parts[0] if len(primitive_parts) == 1 else None
            if len(primitive_parts) != len(primitive_part_ids):
                missing_parts.append({"category": category, "name": name, "step_id": step_id, "part_id": part_id})
                continue
            negative_part_ids = [pid for pid in all_part_ids if pid != str(part_id)]
            samples.append(
                {
                    "category": category,
                    "name": name,
                    "step_id": step_id,
                    "split": (data.get("gt") or {}).get("part_segmentation_split"),
                    "source_feature": feature_path,
                    "source_svg": data.get("source_svg"),
                    "svg_instance_id": stroke,
                    "svg_shape_type": inst.get("shape_type"),
                    "svg_simplified": inst,
                    "positive_part_id": str(part_id),
                    "is_composite": is_composite,
                    "positive_part": part,
                    "positive_primitive_part_ids": primitive_part_ids,
                    "positive_primitive_parts": primitive_parts,
                    "negative_part_ids_in_step": negative_part_ids,
                    "gt_connections": (data.get("gt") or {}).get("connections"),
                    "object_assembly_tree": None,
                }
            )

    with output.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    summary = {
        "num_samples": len(samples),
        "num_missing_parts": len(missing_parts),
        "missing_parts": missing_parts[:50],
        "primitive_samples": sum(1 for s in samples if not s.get("is_composite")),
        "composite_samples": sum(1 for s in samples if s.get("is_composite")),
        "splits": {
            "train": sum(1 for s in samples if s.get("split") == "train"),
            "test": sum(1 for s in samples if s.get("split") == "test"),
        },
        "example": samples[0] if samples else None,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {output} samples={len(samples)}")
    print(f"Wrote {summary_path}")
    if missing_parts:
        print(f"missing_parts={len(missing_parts)}")


if __name__ == "__main__":
    main()
