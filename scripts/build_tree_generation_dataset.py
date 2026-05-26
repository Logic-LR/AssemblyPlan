#!/usr/bin/env python3
"""Build object-level data for SVG-enhanced assembly-tree generation."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from train_grounding_cnn import SHAPE_TYPES, primitive_geometry, svg_feature


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-data", default="main_data.json")
    parser.add_argument("--grounding-samples", default="experiments/svg_assembly/datasets/grounding_samples.jsonl")
    parser.add_argument("--part-index", default="part_images/part_index.jsonl")
    parser.add_argument("--output", default="experiments/svg_assembly/datasets/tree_generation_dataset.json")
    parser.add_argument("--summary", default="experiments/svg_assembly/datasets/tree_generation_summary.json")
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def part_sort_key(value: str) -> Tuple[int, Any]:
    return (0, int(value)) if str(value).isdigit() else (1, str(value))


def part_set_from_tree(value: Any) -> List[int]:
    if isinstance(value, int):
        return [value]
    out: List[int] = []
    for child in value:
        out.extend(part_set_from_tree(child))
    return sorted(set(out))


def postorder_actions(value: Any) -> List[Dict[str, Any]]:
    if isinstance(value, int):
        return []
    actions: List[Dict[str, Any]] = []
    child_sets = []
    for child in value:
        actions.extend(postorder_actions(child))
        child_sets.append(part_set_from_tree(child))
    actions.append(
        {
            "parent": part_set_from_tree(value),
            "children": child_sets,
        }
    )
    return actions


def load_part_index(path: Path) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    out = {}
    for row in load_jsonl(path):
        out[(row["category"], row["name"], str(row["part_id"]))] = row
    return out


def aggregate_svg_features(samples: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    features: Dict[Tuple[str, str, str], List[np.ndarray]] = defaultdict(list)
    shape_counts: Dict[Tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    examples: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)

    for sample in samples:
        primitive_ids = [str(pid) for pid in sample.get("positive_primitive_part_ids") or []]
        positive_id = str(sample.get("positive_part_id"))
        if sample.get("is_composite") or len(primitive_ids) != 1 or primitive_ids[0] != positive_id:
            continue
        key = (sample["category"], sample["name"], positive_id)
        inst = sample.get("svg_simplified") or {}
        features[key].append(svg_feature(inst))
        shape_counts[key][str(inst.get("shape_type") or "unknown")] += 1
        examples[key].append(
            {
                "step_id": sample.get("step_id"),
                "svg_instance_id": sample.get("svg_instance_id"),
                "source_svg": sample.get("source_svg"),
            }
        )

    out = {}
    for key, rows in features.items():
        arr = np.vstack(rows).astype(np.float32)
        shape_total = sum(shape_counts[key].values()) or 1
        out[key] = {
            "svg_feature_mean": arr.mean(axis=0).tolist(),
            "svg_feature_std": arr.std(axis=0).tolist(),
            "svg_feature_count": int(len(rows)),
            "shape_histogram": {shape: int(shape_counts[key][shape]) for shape in sorted(shape_counts[key])},
            "shape_distribution": {
                shape: float(shape_counts[key][shape] / shape_total) for shape in sorted(shape_counts[key])
            },
            "examples": examples[key][:5],
        }
    return out


def aggregate_composite_svg_features(samples: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, str, Tuple[int, ...]], Dict[str, Any]]:
    features: Dict[Tuple[str, str, Tuple[int, ...]], List[np.ndarray]] = defaultdict(list)
    shape_counts: Dict[Tuple[str, str, Tuple[int, ...]], Counter[str]] = defaultdict(Counter)
    examples: Dict[Tuple[str, str, Tuple[int, ...]], List[Dict[str, Any]]] = defaultdict(list)

    for sample in samples:
        primitive_ids = tuple(sorted(int(pid) for pid in sample.get("positive_primitive_part_ids") or []))
        if not sample.get("is_composite") or len(primitive_ids) <= 1:
            continue
        key = (sample["category"], sample["name"], primitive_ids)
        inst = sample.get("svg_simplified") or {}
        features[key].append(svg_feature(inst))
        shape_counts[key][str(inst.get("shape_type") or "unknown")] += 1
        examples[key].append(
            {
                "step_id": sample.get("step_id"),
                "svg_instance_id": sample.get("svg_instance_id"),
                "source_svg": sample.get("source_svg"),
            }
        )

    out = {}
    for key, rows in features.items():
        arr = np.vstack(rows).astype(np.float32)
        shape_total = sum(shape_counts[key].values()) or 1
        out[key] = {
            "part_ids": [str(part_id) for part_id in key[2]],
            "svg_feature_mean": arr.mean(axis=0).tolist(),
            "svg_feature_std": arr.std(axis=0).tolist(),
            "svg_feature_count": int(len(rows)),
            "shape_histogram": {shape: int(shape_counts[key][shape]) for shape in sorted(shape_counts[key])},
            "shape_distribution": {
                shape: float(shape_counts[key][shape] / shape_total) for shape in sorted(shape_counts[key])
            },
            "examples": examples[key][:5],
        }
    return out


def object_split(category: str, name: str, samples_by_object: Dict[Tuple[str, str], List[Dict[str, Any]]]) -> str:
    splits = {sample.get("split") for sample in samples_by_object.get((category, name), [])}
    return "test" if "test" in splits else "train"


def make_part_token(
    category: str,
    name: str,
    part_id: int,
    part_index: Dict[Tuple[str, str, str], Dict[str, Any]],
    svg_index: Dict[Tuple[str, str, str], Dict[str, Any]],
) -> Dict[str, Any]:
    key = (category, name, str(part_id))
    part = part_index.get(key)
    if part is None:
        geom = np.zeros(14, dtype=np.float32)
        part_payload: Dict[str, Any] = {"missing_part_index": True}
    else:
        geom = primitive_geometry(part)
        part_payload = {
            "obj_path": part.get("obj_path"),
            "image_paths": part.get("image_paths") or [],
            "num_vertices": part.get("num_vertices"),
            "num_faces": part.get("num_faces"),
            "bbox_min": part.get("bbox_min"),
            "bbox_max": part.get("bbox_max"),
            "extent": part.get("extent"),
            "center": part.get("center"),
        }

    svg = svg_index.get(key)
    if svg is None:
        svg_payload = {
            "svg_feature_mean": [0.0] * 17,
            "svg_feature_std": [0.0] * 17,
            "svg_feature_count": 0,
            "shape_histogram": {},
            "shape_distribution": {},
            "examples": [],
        }
    else:
        svg_payload = svg

    shape_vec = [float(svg_payload["shape_distribution"].get(shape, 0.0)) for shape in SHAPE_TYPES]
    return {
        "part_id": str(part_id),
        "geometry_feature": geom.astype(float).tolist(),
        "svg_feature_mean": svg_payload["svg_feature_mean"],
        "svg_feature_std": svg_payload["svg_feature_std"],
        "svg_feature_count": svg_payload["svg_feature_count"],
        "shape_distribution": shape_vec,
        "shape_histogram": svg_payload["shape_histogram"],
        "svg_examples": svg_payload["examples"],
        **part_payload,
    }


def make_composite_tokens(
    category: str,
    name: str,
    composite_index: Dict[Tuple[str, str, Tuple[int, ...]], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    out = []
    for key, payload in composite_index.items():
        if key[0] != category or key[1] != name:
            continue
        shape_vec = [float(payload["shape_distribution"].get(shape, 0.0)) for shape in SHAPE_TYPES]
        out.append(
            {
                "part_ids": payload["part_ids"],
                "part_token": ",".join(payload["part_ids"]),
                "svg_feature_mean": payload["svg_feature_mean"],
                "svg_feature_std": payload["svg_feature_std"],
                "svg_feature_count": payload["svg_feature_count"],
                "shape_distribution": shape_vec,
                "shape_histogram": payload["shape_histogram"],
                "svg_examples": payload["examples"],
            }
        )
    return sorted(out, key=lambda item: (len(item["part_ids"]), [int(part) for part in item["part_ids"]]))


def main() -> None:
    args = parse_args()
    main_data = json.loads(Path(args.main_data).read_text(encoding="utf-8"))
    grounding_samples = load_jsonl(Path(args.grounding_samples))
    part_index = load_part_index(Path(args.part_index))
    svg_index = aggregate_svg_features(grounding_samples)
    composite_index = aggregate_composite_svg_features(grounding_samples)
    samples_by_object: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for sample in grounding_samples:
        samples_by_object[(sample["category"], sample["name"])].append(sample)

    records = []
    missing_parts = []
    for obj in main_data:
        category = obj["category"]
        name = obj["name"]
        n_parts = int(obj.get("parts_ct") or len(part_set_from_tree(obj["assembly_tree"])))
        part_tokens = []
        for part_id in range(n_parts):
            token = make_part_token(category, name, part_id, part_index, svg_index)
            part_tokens.append(token)
            if token.get("missing_part_index"):
                missing_parts.append({"category": category, "name": name, "part_id": part_id})
        composite_tokens = make_composite_tokens(category, name, composite_index)
        actions = postorder_actions(obj["assembly_tree"])
        step_groups = [
            {
                "step_id": step.get("step_id"),
                "parts": [str(part) for part in step.get("parts", [])],
                "connections": step.get("connections") or [],
            }
            for step in obj.get("steps", [])
        ]
        records.append(
            {
                "category": category,
                "name": name,
                "split": object_split(category, name, samples_by_object),
                "num_parts": n_parts,
                "part_tokens": part_tokens,
                "composite_tokens": composite_tokens,
                "assembly_tree": obj["assembly_tree"],
                "tree_actions_postorder": actions,
                "manual_step_groups": step_groups,
                "connection_relation": obj.get("connection_relation"),
                "geometric_equivalence_relation": obj.get("geometric_equivalence_relation") or {},
            }
        )

    split_counts = Counter(record["split"] for record in records)
    action_counts = [len(record["tree_actions_postorder"]) for record in records]
    part_counts = [record["num_parts"] for record in records]
    svg_counts = [
        sum(1 for token in record["part_tokens"] if int(token.get("svg_feature_count") or 0) > 0)
        for record in records
    ]
    composite_counts = [len(record.get("composite_tokens") or []) for record in records]
    tree_parent_sets = [
        tuple(str(part) for part in action["parent"])
        for record in records
        for action in record.get("tree_actions_postorder") or []
        if len(action.get("parent") or []) > 1
    ]
    composite_sets = {
        tuple(token["part_ids"])
        for record in records
        for token in record.get("composite_tokens") or []
    }
    composite_tree_hits = sum(1 for part_set in tree_parent_sets if part_set in composite_sets)
    summary = {
        "num_objects": len(records),
        "splits": dict(split_counts),
        "num_missing_parts": len(missing_parts),
        "missing_parts": missing_parts[:20],
        "total_parts": int(sum(part_counts)),
        "total_tree_actions": int(sum(action_counts)),
        "avg_parts_per_object": float(sum(part_counts) / len(part_counts)) if part_counts else 0.0,
        "avg_tree_actions_per_object": float(sum(action_counts) / len(action_counts)) if action_counts else 0.0,
        "parts_with_svg_prototype": int(sum(svg_counts)),
        "parts_with_svg_prototype_rate": float(sum(svg_counts) / max(sum(part_counts), 1)),
        "composite_svg_prototypes": int(sum(composite_counts)),
        "objects_with_composite_svg_prototype": int(sum(1 for count in composite_counts if count > 0)),
        "tree_action_parents_with_composite_svg_prototype": int(composite_tree_hits),
        "tree_action_parent_composite_svg_coverage": float(composite_tree_hits / max(len(tree_parent_sets), 1)),
        "notes": [
            "The split is object-level: any object with at least one existing step-level test sample is assigned to test.",
            "Primitive SVG prototypes are aggregated from primitive, non-composite manual instances.",
            "Composite SVG prototypes are aggregated from manual instances whose positive part id is a multi-primitive subassembly.",
            "Stroke color is retained only as a within-step source key in examples; it is not a semantic identity.",
        ],
    }

    out = Path(args.output)
    summary_path = Path(args.summary)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {out}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
