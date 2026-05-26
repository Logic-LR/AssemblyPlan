#!/usr/bin/env python3
"""Render lightweight orthographic images from IKEA-Manual part OBJ files.

This renderer is intentionally dependency-light: it uses only numpy and PIL.
It is meant for dataset bootstrapping and visual grounding prototypes, not
physically accurate rendering.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts-root", default="parts")
    parser.add_argument("--output-root", default="part_images")
    parser.add_argument("--index-output", default="part_images/part_index.jsonl")
    parser.add_argument("--category", help="Optional category filter.")
    parser.add_argument("--name", help="Optional object-name filter.")
    parser.add_argument("--limit", type=int, help="Optional max number of OBJ files.")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--views", type=int, default=8, help="Number of azimuth views.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip rendering views that already exist.")
    parser.add_argument("--quiet", action="store_true", help="Reduce per-part logging.")
    return parser.parse_args()


def iter_obj_files(root: Path, category: str | None, name: str | None) -> Iterable[Path]:
    paths = sorted(root.glob("*/*/*.obj"))
    for path in paths:
        rel = path.relative_to(root)
        if category and rel.parts[0] != category:
            continue
        if name and rel.parts[1] != name:
            continue
        yield path


def parse_face_index(token: str) -> int:
    return int(token.split("/")[0]) - 1


def load_obj(path: Path) -> Tuple[np.ndarray, List[List[int]]]:
    vertices: List[Tuple[float, float, float]] = []
    faces: List[List[int]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("f "):
                idxs = [parse_face_index(tok) for tok in line.split()[1:]]
                if len(idxs) >= 3:
                    faces.append(idxs)
    return np.asarray(vertices, dtype=float), faces


def rotation_matrix(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    rz = np.asarray(
        [
            [math.cos(az), -math.sin(az), 0.0],
            [math.sin(az), math.cos(az), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rx = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(el), -math.sin(el)],
            [0.0, math.sin(el), math.cos(el)],
        ]
    )
    return rx @ rz


def normalize_vertices(vertices: np.ndarray) -> Tuple[np.ndarray, Dict[str, List[float]]]:
    if len(vertices) == 0:
        return vertices, {"bbox_min": [0, 0, 0], "bbox_max": [0, 0, 0], "extent": [0, 0, 0], "center": [0, 0, 0]}
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    extent = bbox_max - bbox_min
    scale = float(extent.max()) if float(extent.max()) > 1e-9 else 1.0
    normalized = (vertices - center) / scale
    meta = {
        "bbox_min": bbox_min.round(6).tolist(),
        "bbox_max": bbox_max.round(6).tolist(),
        "extent": extent.round(6).tolist(),
        "center": center.round(6).tolist(),
    }
    return normalized, meta


def project(vertices: np.ndarray, azimuth: float, elevation: float, size: int) -> Tuple[np.ndarray, np.ndarray]:
    rot = rotation_matrix(azimuth, elevation)
    pts = vertices @ rot.T
    xy = pts[:, :2]
    z = pts[:, 2]
    max_abs = max(float(np.abs(xy).max()), 1e-6)
    xy = xy / max_abs * (size * 0.38)
    xy[:, 0] += size / 2.0
    xy[:, 1] = size / 2.0 - xy[:, 1]
    return xy, z


def shade_color(normal_z: float) -> Tuple[int, int, int]:
    intensity = 150 + int(75 * max(0.0, min(1.0, normal_z)))
    return (intensity, intensity, intensity)


def render_view(vertices: np.ndarray, faces: List[List[int]], azimuth: float, elevation: float, size: int) -> Image.Image:
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    if len(vertices) == 0:
        return image
    xy, z = project(vertices, azimuth, elevation, size)
    face_items = []
    for face in faces:
        valid = [idx for idx in face if 0 <= idx < len(vertices)]
        if len(valid) < 3:
            continue
        pts3 = vertices[valid]
        v1 = pts3[1] - pts3[0]
        v2 = pts3[2] - pts3[0]
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        normal_z = 0.0 if norm < 1e-9 else float((normal / norm)[2])
        poly = [(float(xy[idx, 0]), float(xy[idx, 1])) for idx in valid]
        face_items.append((float(z[valid].mean()), poly, normal_z))
    for _, poly, normal_z in sorted(face_items, key=lambda item: item[0]):
        draw.polygon(poly, fill=shade_color(abs(normal_z)), outline=(60, 60, 60))
    return image


def part_id_from_path(path: Path) -> str:
    return path.stem.lstrip("0") or "0"


def main() -> None:
    args = parse_args()
    parts_root = Path(args.parts_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    obj_paths = list(iter_obj_files(parts_root, args.category, args.name))
    if args.limit is not None:
        obj_paths = obj_paths[: args.limit]

    records = []
    for obj_path in obj_paths:
        rel = obj_path.relative_to(parts_root)
        category, name = rel.parts[0], rel.parts[1]
        part_id = part_id_from_path(obj_path)
        vertices, faces = load_obj(obj_path)
        vertices_norm, meta = normalize_vertices(vertices)
        part_dir = output_root / category / name / part_id
        part_dir.mkdir(parents=True, exist_ok=True)
        image_paths: List[str] = []
        for view_id in range(args.views):
            azimuth = 360.0 * view_id / args.views
            elevation = 25.0
            image = render_view(vertices_norm, faces, azimuth, elevation, args.size)
            image_path = part_dir / f"view_{view_id:02d}.png"
            if not (args.skip_existing and image_path.exists()):
                image.save(image_path)
            image_paths.append(str(image_path.as_posix()))
        record = {
            "category": category,
            "name": name,
            "part_id": part_id,
            "obj_path": str(obj_path.as_posix()),
            "image_paths": image_paths,
            "num_vertices": int(len(vertices)),
            "num_faces": int(len(faces)),
            **meta,
        }
        (part_dir / "metadata.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        records.append(record)
        if not args.quiet:
            print(part_dir)

    index_path = Path(args.index_output)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {index_path} records={len(records)}")


if __name__ == "__main__":
    main()
