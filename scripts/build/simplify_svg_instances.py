#!/usr/bin/env python3
"""Create simplified SVG geometry for parsed SVG visual instances.

The simplification is intentionally multi-representation:

- convex hull: preserves a coarse filled silhouette;
- oriented bbox: useful for plate/bar-like parts;
- principal axis with width: useful for rod/bar-like parts.

It does not replace raw SVG features. It creates an additional, inspectable
layer under experiments/svg_assembly/simplified_svg by default.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


Point = Tuple[float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", default="svg_features")
    parser.add_argument("--output-root", default="experiments/svg_assembly/simplified_svg")
    parser.add_argument("--category", help="Optional category filter.")
    parser.add_argument("--name", help="Optional object-name filter.")
    parser.add_argument("--limit", type=int, help="Optional max number of step files.")
    parser.add_argument("--max-hull-points", type=int, default=32, help="Downsample hull if it has many vertices.")
    return parser.parse_args()


def iter_feature_files(root: Path, category: str | None, name: str | None) -> Iterable[Path]:
    paths = sorted(root.glob("*/*/step_*.json"))
    for path in paths:
        rel = path.relative_to(root)
        if category and rel.parts[0] != category:
            continue
        if name and rel.parts[1] != name:
            continue
        yield path


def clean_points(points: Sequence[Sequence[float]]) -> List[Point]:
    out: List[Point] = []
    for p in points:
        if len(p) < 2:
            continue
        x, y = float(p[0]), float(p[1])
        if math.isfinite(x) and math.isfinite(y):
            out.append((x, y))
    return out


def cross(o: Point, a: Point, b: Point) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def convex_hull(points: Sequence[Point]) -> List[Point]:
    unique = sorted(set(points))
    if len(unique) <= 1:
        return list(unique)
    lower: List[Point] = []
    for p in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: List[Point] = []
    for p in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def point_line_distance(p: Point, a: Point, b: Point) -> float:
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    if abs(dx) < 1e-12 and abs(dy) < 1e-12:
        return math.hypot(p[0] - a[0], p[1] - a[1])
    t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    x = a[0] + t * dx
    y = a[1] + t * dy
    return math.hypot(p[0] - x, p[1] - y)


def rdp_open(points: Sequence[Point], epsilon: float) -> List[Point]:
    if len(points) <= 2:
        return list(points)
    a, b = points[0], points[-1]
    distances = [point_line_distance(p, a, b) for p in points[1:-1]]
    if not distances:
        return [a, b]
    max_dist = max(distances)
    if max_dist <= epsilon:
        return [a, b]
    split = distances.index(max_dist) + 1
    return rdp_open(points[: split + 1], epsilon)[:-1] + rdp_open(points[split:], epsilon)


def rdp_closed(points: Sequence[Point], epsilon: float) -> List[Point]:
    if len(points) <= 3:
        return list(points)
    return rdp_open(list(points) + [points[0]], epsilon)[:-1]


def downsample_closed(points: Sequence[Point], max_points: int) -> List[Point]:
    if len(points) <= max_points:
        return list(points)
    idxs = np.linspace(0, len(points) - 1, max_points).astype(int)
    return [points[int(i)] for i in idxs]


def pca_axes(points: Sequence[Point]) -> Dict[str, Any]:
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
    min_a, max_a = float(proj.min()), float(proj.max())
    min_p, max_p = float(proj_perp.min()), float(proj_perp.max())
    corners = [
        mean + axis * min_a + perp * min_p,
        mean + axis * max_a + perp * min_p,
        mean + axis * max_a + perp * max_p,
        mean + axis * min_a + perp * max_p,
    ]
    start = mean + axis * min_a
    end = mean + axis * max_a
    return {
        "center": mean.tolist(),
        "axis": axis.tolist(),
        "perp": perp.tolist(),
        "oriented_bbox": [c.tolist() for c in corners],
        "principal_axis": [start.tolist(), end.tolist()],
        "axis_length": max_a - min_a,
        "axis_width": max_p - min_p,
        "elongation": float((vals[0] + 1e-9) / (vals[1] + 1e-9)) if len(vals) > 1 else None,
    }


def rounded_point(p: Sequence[float]) -> List[float]:
    return [round(float(p[0]), 4), round(float(p[1]), 4)]


def point_str(points: Sequence[Sequence[float]]) -> str:
    return " ".join(f"{float(p[0]):.3f},{float(p[1]):.3f}" for p in points)


def polygon_path(points: Sequence[Sequence[float]]) -> str:
    if not points:
        return ""
    first = points[0]
    rest = points[1:]
    return "M" + f"{float(first[0]):.3f} {float(first[1]):.3f} " + " ".join(
        f"L{float(p[0]):.3f} {float(p[1]):.3f}" for p in rest
    ) + " Z"


def css_color(stroke: str) -> str:
    return stroke if stroke else "black"


def simplify_instance(instance: Dict[str, Any], max_hull_points: int) -> Dict[str, Any]:
    points = clean_points(instance.get("sampled_points", []))
    hull = downsample_closed(convex_hull(points), max_hull_points)
    axes = pca_axes(points)
    bbox = instance.get("bbox") or [0.0, 0.0, 0.0, 0.0]
    bbox_diag = math.hypot(float(bbox[2]) - float(bbox[0]), float(bbox[3]) - float(bbox[1]))
    simplified_polygon = rdp_closed(hull, max(2.0, 0.05 * bbox_diag))
    axis_length = float(axes.get("axis_length") or 0.0)
    axis_width = float(axes.get("axis_width") or 0.0)
    elongation = axes.get("elongation")
    if axis_width <= 1e-6:
        shape_type = "point_or_line"
    elif elongation is not None and elongation >= 12:
        shape_type = "elongated_bar"
    elif len(simplified_polygon) <= 6:
        shape_type = "plate_like"
    else:
        shape_type = "irregular"
    return {
        "id": instance["id"],
        "stroke": instance.get("stroke", instance["id"]),
        "shape_type": shape_type,
        "num_source_paths": instance.get("num_paths"),
        "num_sampled_points": len(points),
        "bbox": instance.get("bbox"),
        "center": instance.get("center"),
        "convex_hull": [rounded_point(p) for p in hull],
        "simplified_polygon": [rounded_point(p) for p in simplified_polygon],
        "oriented_bbox": [rounded_point(p) for p in axes.get("oriented_bbox", [])],
        "principal_axis": [rounded_point(p) for p in axes.get("principal_axis", [])],
        "axis_length": round(axis_length, 4),
        "axis_width": round(axis_width, 4),
        "elongation": round(float(elongation), 4) if elongation is not None else None,
        "connection_candidates": {
            "hull_vertices": [rounded_point(p) for p in hull],
            "simplified_polygon_vertices": [rounded_point(p) for p in simplified_polygon],
            "axis_endpoints": [rounded_point(p) for p in axes.get("principal_axis", [])],
            "bbox_corners": [rounded_point(p) for p in axes.get("oriented_bbox", [])],
        },
    }


def write_overlay_svg(data: Dict[str, Any], simplified: List[Dict[str, Any]], out_path: Path) -> None:
    canvas = data.get("canvas") or {}
    width = canvas.get("width") or 793.701
    height = canvas.get("height") or 1122.52
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="white"/>',
    ]
    for inst in simplified:
        stroke = css_color(inst["stroke"])
        hull = inst.get("convex_hull") or []
        simplified_polygon = inst.get("simplified_polygon") or []
        obb = inst.get("oriented_bbox") or []
        axis = inst.get("principal_axis") or []
        if hull:
            lines.append(
                f'<path d="{polygon_path(hull)}" fill="{stroke}" fill-opacity="0.10" stroke="{stroke}" stroke-width="2"/>'
            )
        if simplified_polygon:
            lines.append(
                f'<path d="{polygon_path(simplified_polygon)}" fill="none" stroke="{stroke}" stroke-width="4" stroke-opacity="0.75"/>'
            )
        if obb:
            lines.append(
                f'<path d="{polygon_path(obb)}" fill="none" stroke="{stroke}" stroke-width="1.5" stroke-dasharray="8 5"/>'
            )
        if len(axis) == 2:
            lines.append(
                f'<line x1="{axis[0][0]}" y1="{axis[0][1]}" x2="{axis[1][0]}" y2="{axis[1][1]}" stroke="{stroke}" stroke-width="3" stroke-linecap="round"/>'
            )
        center = inst.get("center")
        if center:
            lines.append(f'<circle cx="{center[0]}" cy="{center[1]}" r="4" fill="{stroke}"/>')
    lines.append("</svg>")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    feature_root = Path(args.feature_root)
    output_root = Path(args.output_root)
    paths = list(iter_feature_files(feature_root, args.category, args.name))
    if args.limit is not None:
        paths = paths[: args.limit]
    for feature_path in paths:
        with feature_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        rel = feature_path.relative_to(feature_root)
        out_dir = output_root / rel.parent / feature_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        simplified = [simplify_instance(inst, args.max_hull_points) for inst in data.get("instances", [])]
        summary = {
            "source_feature": str(feature_path.as_posix()),
            "source_svg": data.get("source_svg"),
            "category": data.get("category"),
            "name": data.get("name"),
            "step_id": data.get("step_id"),
            "instances": simplified,
            "gt": data.get("gt"),
        }
        summary_path = out_dir / "simplified_instances.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        overlay_path = out_dir / "overlay.svg"
        write_overlay_svg(data, simplified, overlay_path)
        for inst in simplified:
            single = {**summary, "instances": [inst]}
            instance_id = str(inst["id"]).replace("#", "hex_").replace("/", "_")
            (out_dir / f"{instance_id}.json").write_text(json.dumps(single, ensure_ascii=False, indent=2), encoding="utf-8")
        print(overlay_path)


if __name__ == "__main__":
    main()
