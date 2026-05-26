#!/usr/bin/env python3
"""Derive and evaluate SVG-color to primitive-part alignment from masks.

The line-seg SVG uses stroke colors as step-level visual instances. This script
uses the pixel masks and `main_data.json` RLE masks to recover which stroke
color corresponds to which `gt.parts` entry.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


RGB = Tuple[int, int, int]

CSS_COLORS: Dict[str, RGB] = {
    "red": (255, 0, 0),
    "black": (0, 0, 0),
    "none": (0, 0, 0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", default="svg_features")
    parser.add_argument("--mask-root", default="mask")
    parser.add_argument("--main-data", default="main_data.json")
    parser.add_argument("--alignment-output", default="svg_features/instance_part_alignment.jsonl")
    parser.add_argument("--report-output", default="experiments/svg_assembly/reports/instance_mask_alignment_report.json")
    return parser.parse_args()


def iter_feature_files(root: Path) -> Iterable[Path]:
    yield from sorted(root.glob("*/*/step_*.json"))


def load_main_data(path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {(item["category"], item["name"]): item for item in data}


def decode_counts(counts: str) -> List[int]:
    out: List[int] = []
    p = 0
    while p < len(counts):
        x = 0
        k = 0
        more = True
        while more:
            c = ord(counts[p]) - 48
            p += 1
            x |= (c & 0x1F) << (5 * k)
            more = bool(c & 0x20)
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if len(out) > 2:
            x += out[-2]
        out.append(x)
    return out


def decode_rle(rle: Dict[str, Any]) -> np.ndarray:
    h, w = rle["size"]
    counts = decode_counts(rle["counts"]) if isinstance(rle["counts"], str) else rle["counts"]
    flat = np.zeros(h * w, dtype=np.uint8)
    idx = 0
    value = 0
    for count in counts:
        if value:
            flat[idx : idx + count] = 1
        idx += count
        value = 1 - value
    return flat.reshape((h, w), order="F").astype(bool)


def color_from_stroke(stroke: str) -> Optional[RGB]:
    stroke = stroke.strip()
    if stroke in CSS_COLORS:
        return CSS_COLORS[stroke]
    if stroke.startswith("#"):
        value = stroke[1:]
        if len(value) == 3:
            value = "".join(ch * 2 for ch in value)
        if len(value) == 6:
            return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))
    return None


def color_distance(a: RGB, b: RGB) -> float:
    return math.sqrt(sum((int(x) - int(y)) ** 2 for x, y in zip(a, b)))


def mask_colors(mask_path: Path) -> Dict[RGB, np.ndarray]:
    arr = np.asarray(Image.open(mask_path).convert("RGB"))
    flat = arr.reshape(-1, 3)
    colors = [tuple(map(int, c)) for c in np.unique(flat, axis=0)]
    colors = [c for c in colors if c != (0, 0, 0)]
    return {c: np.all(arr == np.asarray(c, dtype=np.uint8), axis=2) for c in colors}


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def step_data(obj: Dict[str, Any], step_id: int) -> Optional[Dict[str, Any]]:
    return next((s for s in obj.get("steps", []) if int(s.get("step_id", -1)) == step_id), None)


def deterministic_mapping(instances: List[Dict[str, Any]], parts: List[str], mode: str) -> Dict[str, str]:
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
        raise ValueError(mode)
    return {str(inst["id"]): str(part) for inst, part in zip(ordered, parts)}


def main() -> None:
    args = parse_args()
    feature_root = Path(args.feature_root)
    mask_root = Path(args.mask_root)
    main_data = load_main_data(Path(args.main_data))
    alignment_path = Path(args.alignment_output)
    report_path = Path(args.report_output)
    alignment_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    mode_correct = Counter()
    mode_total = Counter()
    perfect_iou_steps = 0
    low_iou_examples: List[Dict[str, Any]] = []
    color_distance_values: List[float] = []

    for feature_path in iter_feature_files(feature_root):
        with feature_path.open("r", encoding="utf-8") as f:
            feat = json.load(f)
        category = feat["category"]
        name = feat["name"]
        step_id = int(feat["step_id"])
        obj = main_data[(category, name)]
        step = step_data(obj, step_id)
        if step is None:
            continue
        parts = [str(p) for p in step.get("parts", [])]
        rle_masks = [decode_rle(rle) for rle in step.get("masks", [])]
        png_path = mask_root / category / name / f"step_{step_id}_mask.png"
        png_masks = mask_colors(png_path)

        png_color_to_part: Dict[RGB, str] = {}
        png_color_iou: Dict[str, float] = {}
        for color, png_mask in png_masks.items():
            scores = [iou(png_mask, rle_mask) for rle_mask in rle_masks]
            best_idx = int(np.argmax(scores))
            png_color_to_part[color] = parts[best_idx]
            png_color_iou[str(color)] = float(scores[best_idx])

        stroke_to_part: Dict[str, str] = {}
        stroke_to_png_color: Dict[str, List[int]] = {}
        stroke_color_distance: Dict[str, float] = {}
        stroke_iou: Dict[str, float] = {}
        for inst in feat.get("instances", []):
            stroke = str(inst["id"])
            rgb = color_from_stroke(stroke)
            if rgb is None:
                continue
            best_color = min(png_masks.keys(), key=lambda c: color_distance(rgb, c))
            dist = color_distance(rgb, best_color)
            color_distance_values.append(dist)
            stroke_to_part[stroke] = png_color_to_part[best_color]
            stroke_to_png_color[stroke] = list(best_color)
            stroke_color_distance[stroke] = dist
            stroke_iou[stroke] = png_color_iou[str(best_color)]

        if stroke_iou and all(v >= 0.999 for v in stroke_iou.values()):
            perfect_iou_steps += 1
        else:
            low_iou_examples.append(
                {
                    "feature_path": str(feature_path.as_posix()),
                    "stroke_iou": stroke_iou,
                    "stroke_to_part": stroke_to_part,
                }
            )

        for mode in ["feature_order", "x_center", "y_center", "area_desc"]:
            guess = deterministic_mapping(feat.get("instances", []), parts, mode)
            comparable = set(guess) & set(stroke_to_part)
            if comparable:
                mode_total[mode] += 1
                mode_correct[mode] += int(all(guess[k] == stroke_to_part[k] for k in comparable))

        records.append(
            {
                "category": category,
                "name": name,
                "step_id": step_id,
                "feature_path": str(feature_path.as_posix()),
                "mask_path": str(png_path.as_posix()),
                "gt_parts": parts,
                "stroke_to_part": stroke_to_part,
                "stroke_to_png_color": stroke_to_png_color,
                "stroke_color_distance": stroke_color_distance,
                "stroke_iou": stroke_iou,
            }
        )

    with alignment_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    report = {
        "num_steps": len(records),
        "perfect_iou_steps": perfect_iou_steps,
        "perfect_iou_step_rate": perfect_iou_steps / len(records) if records else 0.0,
        "max_stroke_to_png_color_distance": max(color_distance_values) if color_distance_values else None,
        "deterministic_alignment_exact_rates": {
            mode: mode_correct[mode] / mode_total[mode] if mode_total[mode] else 0.0
            for mode in ["feature_order", "x_center", "y_center", "area_desc"]
        },
        "low_iou_examples": low_iou_examples[:20],
    }
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote {alignment_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
