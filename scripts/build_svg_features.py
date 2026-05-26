#!/usr/bin/env python3
"""Build structured SVG features for IKEA-Manual line segmentation files.

The script intentionally does not infer assembly trees. It converts each
step-level SVG into objective vector features and attaches the ground-truth
step annotations from main_data.json when available.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


Point = Tuple[float, float]


COMMAND_RE = re.compile(
    r"[AaCcHhLlMmQqSsTtVvZz]|[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?"
)


@dataclass
class PathMeta:
    d: str
    stroke: str
    fill: str
    stroke_width: Optional[float]
    stroke_dasharray: str
    points: List[Point]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default="line_seg", help="Root directory containing step SVG files.")
    parser.add_argument("--main-data", default="main_data.json", help="IKEA-Manual main_data.json path.")
    parser.add_argument("--output-root", default="svg_features", help="Directory to write feature JSON files.")
    parser.add_argument("--category", help="Optional category filter, e.g. Bench.")
    parser.add_argument("--name", help="Optional object-name filter, e.g. applaro.")
    parser.add_argument("--limit", type=int, help="Optional maximum number of SVGs to process.")
    parser.add_argument("--samples-per-segment", type=int, default=8, help="Samples for curves/long segments.")
    parser.add_argument("--max-points-per-instance", type=int, default=256, help="Downsampled points stored per instance.")
    parser.add_argument("--include-raw-d", action="store_true", help="Include raw path d strings in JSON output.")
    return parser.parse_args()


def as_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    match = re.search(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?", value)
    if not match:
        return None
    return float(match.group(0))


def parse_style(style: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for chunk in style.split(";"):
        if ":" not in chunk:
            continue
        key, value = chunk.split(":", 1)
        out[key.strip()] = value.strip()
    return out


def get_svg_attr(elem: ET.Element, key: str) -> Optional[str]:
    return elem.attrib.get(key)


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_viewbox(value: Optional[str]) -> Optional[List[float]]:
    if not value:
        return None
    nums = [float(x) for x in re.findall(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?", value)]
    return nums if len(nums) == 4 else None


def tokenize_path(d: str) -> List[str]:
    return [m.group(0) for m in COMMAND_RE.finditer(d.replace(",", " "))]


def is_command(token: str) -> bool:
    return len(token) == 1 and token.isalpha()


def cubic(p0: Point, p1: Point, p2: Point, p3: Point, t: float) -> Point:
    mt = 1.0 - t
    x = mt**3 * p0[0] + 3 * mt**2 * t * p1[0] + 3 * mt * t**2 * p2[0] + t**3 * p3[0]
    y = mt**3 * p0[1] + 3 * mt**2 * t * p1[1] + 3 * mt * t**2 * p2[1] + t**3 * p3[1]
    return (x, y)


def quad(p0: Point, p1: Point, p2: Point, t: float) -> Point:
    mt = 1.0 - t
    x = mt**2 * p0[0] + 2 * mt * t * p1[0] + t**2 * p2[0]
    y = mt**2 * p0[1] + 2 * mt * t * p1[1] + t**2 * p2[1]
    return (x, y)


def lerp(p0: Point, p1: Point, t: float) -> Point:
    return (p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t)


def append_line(points: List[Point], p0: Point, p1: Point, samples: int) -> None:
    steps = max(2, samples)
    for idx in range(1, steps + 1):
        points.append(lerp(p0, p1, idx / steps))


def parse_path_points(d: str, samples_per_segment: int) -> List[Point]:
    """Sample points from an SVG path.

    This parser is designed for feature extraction, not exact rendering.
    Elliptical arcs are approximated by endpoint interpolation.
    """
    tokens = tokenize_path(d)
    points: List[Point] = []
    i = 0
    cmd = ""
    current = (0.0, 0.0)
    start = (0.0, 0.0)
    last_cubic_ctrl: Optional[Point] = None
    last_quad_ctrl: Optional[Point] = None

    def has_number(offset: int = 0) -> bool:
        return i + offset < len(tokens) and not is_command(tokens[i + offset])

    def read_float() -> float:
        nonlocal i
        value = float(tokens[i])
        i += 1
        return value

    def absolute_xy(x: float, y: float, relative: bool) -> Point:
        return (current[0] + x, current[1] + y) if relative else (x, y)

    while i < len(tokens):
        if is_command(tokens[i]):
            cmd = tokens[i]
            i += 1
        if not cmd:
            break

        relative = cmd.islower()
        upper = cmd.upper()

        if upper == "M":
            if not (has_number() and has_number(1)):
                continue
            x, y = read_float(), read_float()
            current = absolute_xy(x, y, relative)
            start = current
            points.append(current)
            cmd = "l" if relative else "L"
            last_cubic_ctrl = None
            last_quad_ctrl = None
            continue

        if upper == "L":
            while has_number() and has_number(1):
                p0 = current
                current = absolute_xy(read_float(), read_float(), relative)
                append_line(points, p0, current, samples_per_segment)
            last_cubic_ctrl = None
            last_quad_ctrl = None
            continue

        if upper == "H":
            while has_number():
                p0 = current
                x = read_float()
                current = (current[0] + x, current[1]) if relative else (x, current[1])
                append_line(points, p0, current, samples_per_segment)
            last_cubic_ctrl = None
            last_quad_ctrl = None
            continue

        if upper == "V":
            while has_number():
                p0 = current
                y = read_float()
                current = (current[0], current[1] + y) if relative else (current[0], y)
                append_line(points, p0, current, samples_per_segment)
            last_cubic_ctrl = None
            last_quad_ctrl = None
            continue

        if upper == "C":
            while all(has_number(k) for k in range(6)):
                p0 = current
                c1 = absolute_xy(read_float(), read_float(), relative)
                c2 = absolute_xy(read_float(), read_float(), relative)
                current = absolute_xy(read_float(), read_float(), relative)
                for idx in range(1, samples_per_segment + 1):
                    points.append(cubic(p0, c1, c2, current, idx / samples_per_segment))
                last_cubic_ctrl = c2
                last_quad_ctrl = None
            continue

        if upper == "S":
            while all(has_number(k) for k in range(4)):
                p0 = current
                if last_cubic_ctrl is None:
                    c1 = current
                else:
                    c1 = (2 * current[0] - last_cubic_ctrl[0], 2 * current[1] - last_cubic_ctrl[1])
                c2 = absolute_xy(read_float(), read_float(), relative)
                current = absolute_xy(read_float(), read_float(), relative)
                for idx in range(1, samples_per_segment + 1):
                    points.append(cubic(p0, c1, c2, current, idx / samples_per_segment))
                last_cubic_ctrl = c2
                last_quad_ctrl = None
            continue

        if upper == "Q":
            while all(has_number(k) for k in range(4)):
                p0 = current
                c1 = absolute_xy(read_float(), read_float(), relative)
                current = absolute_xy(read_float(), read_float(), relative)
                for idx in range(1, samples_per_segment + 1):
                    points.append(quad(p0, c1, current, idx / samples_per_segment))
                last_quad_ctrl = c1
                last_cubic_ctrl = None
            continue

        if upper == "T":
            while all(has_number(k) for k in range(2)):
                p0 = current
                if last_quad_ctrl is None:
                    c1 = current
                else:
                    c1 = (2 * current[0] - last_quad_ctrl[0], 2 * current[1] - last_quad_ctrl[1])
                current = absolute_xy(read_float(), read_float(), relative)
                for idx in range(1, samples_per_segment + 1):
                    points.append(quad(p0, c1, current, idx / samples_per_segment))
                last_quad_ctrl = c1
                last_cubic_ctrl = None
            continue

        if upper == "A":
            while all(has_number(k) for k in range(7)):
                p0 = current
                # rx, ry, x-axis-rotation, large-arc-flag, sweep-flag
                _ = [read_float() for _ in range(5)]
                current = absolute_xy(read_float(), read_float(), relative)
                append_line(points, p0, current, samples_per_segment)
            last_cubic_ctrl = None
            last_quad_ctrl = None
            continue

        if upper == "Z":
            append_line(points, current, start, samples_per_segment)
            current = start
            last_cubic_ctrl = None
            last_quad_ctrl = None
            cmd = ""
            continue

        # Unknown command: stop parsing this path rather than silently drifting.
        break

    return points


def bbox(points: Sequence[Point]) -> Optional[List[float]]:
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def center_from_bbox(box: Optional[Sequence[float]]) -> Optional[List[float]]:
    if not box:
        return None
    return [(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0]


def downsample_points(points: Sequence[Point], max_points: int) -> List[List[float]]:
    if len(points) <= max_points:
        return [[round(x, 4), round(y, 4)] for x, y in points]
    idxs = np.linspace(0, len(points) - 1, max_points).astype(int)
    return [[round(points[i][0], 4), round(points[i][1], 4)] for i in idxs]


def polyline_length(points: Sequence[Point]) -> float:
    if len(points) < 2:
        return 0.0
    return sum(math.dist(points[i - 1], points[i]) for i in range(1, len(points)))


def pca_features(points: Sequence[Point]) -> Dict[str, Any]:
    if len(points) < 2:
        return {}
    arr = np.asarray(points, dtype=float)
    mean = arr.mean(axis=0)
    centered = arr - mean
    cov = np.cov(centered.T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    axis = vecs[:, 0]
    perp = np.array([-axis[1], axis[0]])
    proj = centered @ axis
    proj_perp = centered @ perp
    start = mean + axis * proj.min()
    end = mean + axis * proj.max()
    return {
        "principal_axis": [[round(float(start[0]), 4), round(float(start[1]), 4)], [round(float(end[0]), 4), round(float(end[1]), 4)]],
        "axis_length": round(float(proj.max() - proj.min()), 4),
        "axis_width": round(float(proj_perp.max() - proj_perp.min()), 4),
        "elongation": round(float((vals[0] + 1e-9) / (vals[1] + 1e-9)), 4) if len(vals) > 1 else None,
    }


def bbox_overlap(a: Sequence[float], b: Sequence[float]) -> float:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)


def bbox_distance(a: Sequence[float], b: Sequence[float]) -> float:
    dx = max(a[0] - b[2], b[0] - a[2], 0.0)
    dy = max(a[1] - b[3], b[1] - a[3], 0.0)
    return math.hypot(dx, dy)


def min_point_distance(a: Sequence[Point], b: Sequence[Point], cap: int = 128) -> Optional[float]:
    if not a or not b:
        return None
    aa = np.asarray(downsample_points(a, cap), dtype=float)
    bb = np.asarray(downsample_points(b, cap), dtype=float)
    diff = aa[:, None, :] - bb[None, :, :]
    dist = np.sqrt((diff * diff).sum(axis=2))
    return float(dist.min())


def parse_svg(svg_path: Path, samples_per_segment: int, max_points: int, include_raw_d: bool) -> Dict[str, Any]:
    tree = ET.parse(svg_path)
    root = tree.getroot()
    width = as_float(get_svg_attr(root, "width"))
    height = as_float(get_svg_attr(root, "height"))
    viewbox = parse_viewbox(get_svg_attr(root, "viewBox"))

    paths: List[PathMeta] = []
    for elem in root.iter():
        if local_name(elem.tag) != "path":
            continue
        d = elem.attrib.get("d", "")
        if not d:
            continue
        style = parse_style(elem.attrib.get("style", ""))
        stroke = elem.attrib.get("stroke") or style.get("stroke") or "none"
        fill = elem.attrib.get("fill") or style.get("fill") or "none"
        stroke_width = as_float(elem.attrib.get("stroke-width") or style.get("stroke-width"))
        stroke_dasharray = elem.attrib.get("stroke-dasharray") or style.get("stroke-dasharray") or "none"
        points = parse_path_points(d, samples_per_segment)
        paths.append(PathMeta(d, stroke, fill, stroke_width, stroke_dasharray, points))

    grouped: Dict[str, List[PathMeta]] = {}
    for path_meta in paths:
        grouped.setdefault(path_meta.stroke, []).append(path_meta)

    instances: List[Dict[str, Any]] = []
    point_cache: Dict[str, List[Point]] = {}
    for stroke, items in sorted(grouped.items(), key=lambda kv: kv[0]):
        all_points = [pt for item in items for pt in item.points]
        box = bbox(all_points)
        point_cache[stroke] = all_points
        instance: Dict[str, Any] = {
            "id": stroke,
            "stroke": stroke,
            "num_paths": len(items),
            "num_points": len(all_points),
            "bbox": [round(float(v), 4) for v in box] if box else None,
            "center": [round(float(v), 4) for v in center_from_bbox(box)] if box else None,
            "total_polyline_length": round(sum(polyline_length(item.points) for item in items), 4),
            "sampled_points": downsample_points(all_points, max_points),
            "stroke_widths": sorted({item.stroke_width for item in items if item.stroke_width is not None}),
            "fills": sorted({item.fill for item in items}),
            "dasharrays": sorted({item.stroke_dasharray for item in items}),
        }
        instance.update(pca_features(all_points))
        if include_raw_d:
            instance["raw_paths"] = [{"d": item.d, "fill": item.fill, "stroke_width": item.stroke_width} for item in items]
        instances.append(instance)

    spatial_relations: List[Dict[str, Any]] = []
    for i in range(len(instances)):
        for j in range(i + 1, len(instances)):
            a = instances[i]
            b = instances[j]
            if not a["bbox"] or not b["bbox"]:
                continue
            center_a = a["center"] or [0.0, 0.0]
            center_b = b["center"] or [0.0, 0.0]
            min_dist = min_point_distance(point_cache[a["id"]], point_cache[b["id"]])
            spatial_relations.append(
                {
                    "a": a["id"],
                    "b": b["id"],
                    "bbox_distance": round(bbox_distance(a["bbox"], b["bbox"]), 4),
                    "bbox_overlap_area": round(bbox_overlap(a["bbox"], b["bbox"]), 4),
                    "center_delta": [round(center_b[0] - center_a[0], 4), round(center_b[1] - center_a[1], 4)],
                    "min_sampled_point_distance": round(min_dist, 4) if min_dist is not None else None,
                    "likely_contact": bool(min_dist is not None and min_dist <= 5.0),
                }
            )

    return {
        "source_svg": str(svg_path.as_posix()),
        "canvas": {"width": width, "height": height, "viewBox": viewbox},
        "num_paths": len(paths),
        "instances": instances,
        "spatial_relations": spatial_relations,
    }


def load_main_data(path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {(item["category"], item["name"]): item for item in data}


def find_svgs(input_root: Path, category: Optional[str], name: Optional[str]) -> List[Path]:
    pattern = "step_*.svg"
    paths = []
    for path in input_root.glob(f"*/*/{pattern}"):
        rel = path.relative_to(input_root)
        if category and rel.parts[0] != category:
            continue
        if name and rel.parts[1] != name:
            continue
        paths.append(path)
    return sorted(paths)


def step_id_from_path(path: Path) -> Optional[int]:
    match = re.search(r"step_(\d+)\.svg$", path.name)
    return int(match.group(1)) if match else None


def attach_gt(features: Dict[str, Any], object_data: Optional[Dict[str, Any]], step_id: int) -> None:
    if object_data is None:
        features["gt"] = None
        return
    step = next((s for s in object_data.get("steps", []) if int(s.get("step_id", -1)) == step_id), None)
    features["object_gt"] = {
        "parts_ct": object_data.get("parts_ct"),
        "assembly_tree": object_data.get("assembly_tree"),
        "connection_relation": object_data.get("connection_relation"),
        "geometric_equivalence_relation": object_data.get("geometric_equivalence_relation"),
    }
    if step is None:
        features["gt"] = None
        return
    features["gt"] = {
        "step_id": step.get("step_id"),
        "step_id_global": step.get("step_id_global"),
        "page_id": step.get("page_id"),
        "parts": step.get("parts"),
        "connections": step.get("connections"),
        "part_segmentation_split": step.get("part_segmentation_split"),
        "num_masks": len(step.get("masks", [])),
        "num_poses": len(step.get("extrinsics", [])),
    }


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    main_data = load_main_data(Path(args.main_data))

    svg_paths = find_svgs(input_root, args.category, args.name)
    if args.limit is not None:
        svg_paths = svg_paths[: args.limit]

    for svg_path in svg_paths:
        rel = svg_path.relative_to(input_root)
        category, name = rel.parts[0], rel.parts[1]
        step_id = step_id_from_path(svg_path)
        if step_id is None:
            continue
        features = parse_svg(svg_path, args.samples_per_segment, args.max_points_per_instance, args.include_raw_d)
        features.update({"category": category, "name": name, "step_id": step_id})
        attach_gt(features, main_data.get((category, name)), step_id)

        out_path = output_root / category / name / f"step_{step_id}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(features, f, ensure_ascii=False, indent=2)
        print(out_path)


if __name__ == "__main__":
    main()
