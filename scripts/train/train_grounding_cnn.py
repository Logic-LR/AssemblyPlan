#!/usr/bin/env python3
"""Train a small CNN + geometry model for part-to-SVG grounding."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

SHAPE_TYPES = ["elongated_bar", "plate_like", "irregular", "point_or_line"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", default="experiments/svg_assembly/datasets/grounding_samples.jsonl")
    parser.add_argument("--data-json", default="main_data.json")
    parser.add_argument("--primitive-only", action="store_true")
    parser.add_argument("--equivalence-labels", action="store_true")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--max-images", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--augment-translate", type=int, default=4)
    parser.add_argument("--augment-scale", type=float, default=0.12)
    parser.add_argument("--augment-noise", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="experiments/svg_assembly/reports/grounding_cnn_report.json")
    parser.add_argument("--model-output", default="experiments/svg_assembly/reports/grounding_cnn_model.pt")
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def safe(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def part_sort_key(value: str) -> Tuple[int, Any]:
    return (0, int(value)) if value.isdigit() else (1, value)


def split_part_id(part_id: str) -> List[str]:
    return sorted([p.strip() for p in str(part_id).split(",") if p.strip()], key=part_sort_key)


def parse_token(value: Any) -> Tuple[int, ...]:
    return tuple(int(piece) for piece in split_part_id(str(value)) if str(piece).isdigit())


class DisjointSet:
    def __init__(self, n_items: int) -> None:
        self.parent = list(range(n_items))

    def find(self, item: int) -> int:
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


class EquivalenceLookup:
    def __init__(self, n_parts: int, relation: Dict[str, List[str]]) -> None:
        self.n_parts = n_parts
        self.dsu = DisjointSet(n_parts)
        for key, values in (relation or {}).items():
            key_parts = parse_token(key)
            for value in values:
                value_parts = parse_token(value)
                if len(key_parts) == 1 and len(value_parts) == 1:
                    self.dsu.union(key_parts[0], value_parts[0])

    def signature(self, token: Any) -> Tuple[int, ...]:
        parts = parse_token(token)
        return tuple(sorted(self.dsu.find(part) for part in parts if 0 <= part < self.n_parts))

    def equivalent(self, a: Any, b: Any) -> bool:
        return str(a) == str(b) or self.signature(a) == self.signature(b)


def load_equivalences(path: Path) -> Dict[Tuple[str, str], EquivalenceLookup]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    out: Dict[Tuple[str, str], EquivalenceLookup] = {}
    for obj in data:
        out[(obj["category"], obj["name"])] = EquivalenceLookup(
            int(obj.get("parts_ct") or 0),
            obj.get("geometric_equivalence_relation") or {},
        )
    return out


def polygon_area(points: Sequence[Sequence[float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for i, p in enumerate(points):
        q = points[(i + 1) % len(points)]
        area += safe(p[0]) * safe(q[1]) - safe(q[0]) * safe(p[1])
    return abs(area) / 2.0


def svg_feature(inst: Dict[str, Any]) -> np.ndarray:
    box = inst.get("bbox") or [0, 0, 0, 0]
    bw = max(0.0, safe(box[2]) - safe(box[0]))
    bh = max(0.0, safe(box[3]) - safe(box[1]))
    canvas_area = 793.701 * 1122.52
    diag = math.hypot(793.701, 1122.52)
    poly = inst.get("simplified_polygon") or []
    hull = inst.get("convex_hull") or []
    axis_length = safe(inst.get("axis_length"))
    axis_width = safe(inst.get("axis_width"))
    aspect = axis_length / max(axis_width, 1e-9)
    center = inst.get("center") or [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
    onehot = [1.0 if inst.get("shape_type") == shape else 0.0 for shape in SHAPE_TYPES]
    return np.asarray(
        [
            safe(center[0]) / 793.701,
            safe(center[1]) / 1122.52,
            bw / 793.701,
            bh / 1122.52,
            (bw * bh) / canvas_area,
            polygon_area(poly) / canvas_area,
            polygon_area(hull) / canvas_area,
            axis_length / diag,
            axis_width / diag,
            math.log1p(safe(inst.get("elongation"))),
            math.log1p(aspect),
            len(poly) / 12.0,
            len(hull) / 32.0,
            *onehot,
        ],
        dtype=np.float32,
    )


def primitive_geometry(part: Dict[str, Any]) -> np.ndarray:
    ext = np.asarray(part.get("extent", [0.0, 0.0, 0.0]), dtype=np.float32)
    ext = np.maximum(ext, 1e-9)
    sorted_ext = np.sort(ext)[::-1]
    ratios = np.asarray(
        [
            sorted_ext[0] / sorted_ext[1],
            sorted_ext[1] / sorted_ext[2],
            sorted_ext[0] / sorted_ext[2],
        ],
        dtype=np.float32,
    )
    return np.asarray(
        [
            *ext.tolist(),
            *sorted_ext.tolist(),
            *ratios.tolist(),
            float(np.prod(ext)),
            math.log1p(safe(part.get("num_faces"))),
            math.log1p(safe(part.get("num_vertices"))),
            1.0 if ratios[0] > 8 else 0.0,
            1.0 if ratios[1] > 4 else 0.0,
        ],
        dtype=np.float32,
    )


def aggregate_geometry(parts: Sequence[Dict[str, Any]]) -> np.ndarray:
    if not parts:
        return np.zeros(15 * 3 + 4, dtype=np.float32)
    arr = np.vstack([primitive_geometry(part) for part in parts])
    return np.concatenate(
        [
            arr.mean(axis=0),
            arr.max(axis=0),
            arr.min(axis=0),
            np.asarray([len(parts), math.log1p(len(parts)), 1.0 if len(parts) > 1 else 0.0, 1.0], dtype=np.float32),
        ]
    ).astype(np.float32)


def build_part_lookup(samples: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    lookup = {}
    for sample in samples:
        for part in sample.get("positive_primitive_parts", []):
            lookup[(part["category"], part["name"], str(part["part_id"]))] = part
    return lookup


def select_evenly(items: Sequence[str], max_count: int) -> List[str]:
    if len(items) <= max_count:
        return list(items)
    idx = np.linspace(0, len(items) - 1, max_count).astype(int)
    return [items[int(i)] for i in idx]


def candidate_parts_and_images(
    category: str,
    name: str,
    part_id: str,
    part_lookup: Dict[Tuple[str, str, str], Dict[str, Any]],
    max_images: int,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    parts = []
    image_paths = []
    for pid in split_part_id(part_id):
        part = part_lookup.get((category, name, pid))
        if part is None:
            continue
        parts.append(part)
        image_paths.extend(part.get("image_paths", []))
    return parts, select_evenly(image_paths, max_images)


def make_pair_examples(
    samples: Sequence[Dict[str, Any]],
    primitive_only: bool,
    max_images: int,
    equivalences: Dict[Tuple[str, str], EquivalenceLookup] | None = None,
    equivalence_labels: bool = False,
) -> List[Dict[str, Any]]:
    if primitive_only:
        samples = [s for s in samples if not s.get("is_composite")]
    part_lookup = build_part_lookup(samples)
    by_step = defaultdict(list)
    for sample in samples:
        by_step[f"{sample['category']}/{sample['name']}/step_{sample['step_id']}"].append(sample)

    examples = []
    for step_key, step_samples in by_step.items():
        candidate_ids = sorted({str(s["positive_part_id"]) for s in step_samples}, key=part_sort_key)
        for sample in step_samples:
            sf = svg_feature(sample["svg_simplified"])
            for candidate_id in candidate_ids:
                parts, image_paths = candidate_parts_and_images(
                    sample["category"], sample["name"], candidate_id, part_lookup, max_images
                )
                if not parts or not image_paths:
                    continue
                exact_label = 1.0 if candidate_id == str(sample["positive_part_id"]) else 0.0
                equiv = (equivalences or {}).get((sample["category"], sample["name"]))
                is_equivalent = bool(equiv and equiv.equivalent(candidate_id, sample["positive_part_id"]))
                examples.append(
                    {
                        "category": sample["category"],
                        "name": sample["name"],
                        "step_key": step_key,
                        "split": sample.get("split"),
                        "svg_instance_id": sample["svg_instance_id"],
                        "candidate_part_id": candidate_id,
                        "positive_part_id": str(sample["positive_part_id"]),
                        "exact_label": exact_label,
                        "equivalence_label": 1.0 if is_equivalent or exact_label > 0.5 else 0.0,
                        "label": 1.0 if equivalence_labels and (is_equivalent or exact_label > 0.5) else exact_label,
                        "svg_feature": sf,
                        "part_geometry": aggregate_geometry(parts),
                        "image_paths": image_paths,
                    }
                )
    return examples


def split_examples(
    examples: Sequence[Dict[str, Any]], val_fraction: float, seed: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    train_candidates = [ex for ex in examples if ex["split"] == "train"]
    test_examples = [ex for ex in examples if ex["split"] == "test"]
    step_keys = sorted({ex["step_key"] for ex in train_candidates})
    val_keys: set[str] = set()
    if 0.0 < val_fraction < 1.0 and len(step_keys) > 1:
        rng = random.Random(seed)
        rng.shuffle(step_keys)
        val_count = max(1, min(len(step_keys) - 1, round(len(step_keys) * val_fraction)))
        val_keys = set(step_keys[:val_count])
    fit_examples = [ex for ex in train_candidates if ex["step_key"] not in val_keys]
    val_examples = [ex for ex in train_candidates if ex["step_key"] in val_keys]
    return fit_examples, val_examples, test_examples, sorted(val_keys)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        return F.relu(x + r, inplace=True)


class BetterImageCNN(nn.Module):
    """Residual CNN for 64x64 grayscale silhouette images → 256-dim features."""

    def __init__(self, out_dim: int = 256) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        # stage 1: 32ch @ 64x64 → 64ch @ 32x32
        self.stage1 = nn.Sequential(
            ResidualBlock(32),
            ResidualBlock(32),
        )
        self.proj1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # stage 2: 64ch @ 32x32 → 128ch @ 16x16
        self.stage2 = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
        )
        self.proj2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        # stage 3: 128ch @ 16x16 → 256ch @ 8x8
        self.stage3 = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
        )
        self.proj3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.proj1(x)
        x = self.stage2(x)
        x = self.proj2(x)
        x = self.stage3(x)
        x = self.proj3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def _preload_images(examples: Sequence[Dict[str, Any]], image_size: int, max_images: int) -> Dict[str, torch.Tensor]:
    """Preload all unique images into memory to eliminate per-epoch disk I/O."""
    all_paths: set[str] = set()
    for ex in examples:
        for p in ex.get("image_paths", [])[:max_images]:
            all_paths.add(p)
    cache: Dict[str, torch.Tensor] = {}
    for path in sorted(all_paths):
        image = Image.open(path).convert("L").resize((image_size, image_size))
        arr = np.asarray(image, dtype=np.float32) / 255.0
        arr = 1.0 - arr  # invert: part=white, background=black
        cache[path] = torch.from_numpy(arr[None, :, :])
    return cache


class GroundingDataset(Dataset):
    def __init__(
        self,
        examples: Sequence[Dict[str, Any]],
        image_size: int,
        max_images: int,
        svg_mean: np.ndarray,
        svg_std: np.ndarray,
        geom_mean: np.ndarray,
        geom_std: np.ndarray,
        augment: bool = False,
        translate: int = 4,
        scale_jitter: float = 0.12,
        noise_std: float = 0.02,
    ) -> None:
        self.examples = list(examples)
        self.image_size = image_size
        self.max_images = max_images
        self.svg_mean = svg_mean.astype(np.float32)
        self.svg_std = svg_std.astype(np.float32)
        self.geom_mean = geom_mean.astype(np.float32)
        self.geom_std = geom_std.astype(np.float32)
        self.augment = augment
        self.translate = translate
        self.scale_jitter = scale_jitter
        self.noise_std = noise_std
        self.image_cache = _preload_images(examples, image_size, max_images)

    def __len__(self) -> int:
        return len(self.examples)

    @staticmethod
    def match_feature_dim(feature: np.ndarray, mean: np.ndarray) -> np.ndarray:
        if feature.shape[0] == mean.shape[0]:
            return feature
        if feature.shape[0] == mean.shape[0] + 2:
            # New SVG features prepend center_x/center_y. Legacy checkpoints were
            # trained on the remaining 15 dimensions.
            return feature[2:]
        raise ValueError(f"Feature dimension {feature.shape[0]} does not match checkpoint dimension {mean.shape[0]}")

    def augment_image(self, image: torch.Tensor) -> torch.Tensor:
        if not self.augment:
            return image
        out = image.clone()
        if self.scale_jitter > 0:
            scale = random.uniform(1.0 - self.scale_jitter, 1.0 + self.scale_jitter)
            size = max(4, int(round(self.image_size * scale)))
            scaled = F.interpolate(out.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False).squeeze(0)
            canvas = torch.zeros_like(out)
            if size >= self.image_size:
                start = (size - self.image_size) // 2
                out = scaled[:, start : start + self.image_size, start : start + self.image_size]
            else:
                start = (self.image_size - size) // 2
                canvas[:, start : start + size, start : start + size] = scaled
                out = canvas
        if self.translate > 0:
            dx = random.randint(-self.translate, self.translate)
            dy = random.randint(-self.translate, self.translate)
            shifted = torch.zeros_like(out)
            src_x0 = max(0, -dx)
            src_y0 = max(0, -dy)
            dst_x0 = max(0, dx)
            dst_y0 = max(0, dy)
            width = self.image_size - abs(dx)
            height = self.image_size - abs(dy)
            if width > 0 and height > 0:
                shifted[:, dst_y0 : dst_y0 + height, dst_x0 : dst_x0 + width] = out[
                    :, src_y0 : src_y0 + height, src_x0 : src_x0 + width
                ]
            out = shifted
        if self.noise_std > 0:
            out = out + torch.randn_like(out) * self.noise_std
        return out.clamp(0.0, 1.0)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[index]
        paths = ex["image_paths"][: self.max_images]
        images = torch.zeros((self.max_images, 1, self.image_size, self.image_size), dtype=torch.float32)
        mask = torch.zeros((self.max_images,), dtype=torch.float32)
        for i, path in enumerate(paths):
            if path in self.image_cache:
                images[i] = self.augment_image(self.image_cache[path])
                mask[i] = 1.0
        svg_feature = self.match_feature_dim(ex["svg_feature"], self.svg_mean)
        geom_feature = self.match_feature_dim(ex["part_geometry"], self.geom_mean)
        svg = (svg_feature - self.svg_mean) / self.svg_std
        geom = (geom_feature - self.geom_mean) / self.geom_std
        return {
            "images": images,
            "image_mask": mask,
            "svg": torch.from_numpy(svg.astype(np.float32)),
            "geom": torch.from_numpy(geom.astype(np.float32)),
            "label": torch.tensor(ex["label"], dtype=torch.float32),
        }


class LegacyGroundingCNN(nn.Module):
    """Original shallow CNN kept for loading historical checkpoints."""

    def __init__(self, svg_dim: int, geom_dim: int, dropout: float) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.view_fc = nn.Sequential(nn.Linear(32, 48), nn.ReLU(inplace=True))
        self.svg_mlp = nn.Sequential(nn.Linear(svg_dim, 32), nn.ReLU(inplace=True))
        self.geom_mlp = nn.Sequential(nn.Linear(geom_dim, 32), nn.ReLU(inplace=True))
        self.head = nn.Sequential(
            nn.Linear(48 * 2 + 32 + 32, 80),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(80, 1),
        )

    def forward(self, images: torch.Tensor, image_mask: torch.Tensor, svg: torch.Tensor, geom: torch.Tensor) -> torch.Tensor:
        b, n, c, h, w = images.shape
        img_feats = self.cnn(images.view(b * n, c, h, w)).flatten(1)
        img_feats = self.view_fc(img_feats).view(b, n, -1)
        mask = image_mask.unsqueeze(-1)
        mean_pool = (img_feats * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        max_pool = img_feats.masked_fill(mask == 0, float("-inf")).max(dim=1).values
        max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))
        fused = torch.cat(
            [
                mean_pool,
                max_pool,
                self.svg_mlp(svg),
                self.geom_mlp(geom),
            ],
            dim=1,
        )
        return self.head(fused).squeeze(1)


class TinyGroundingCNN(nn.Module):
    def __init__(self, svg_dim: int, geom_dim: int, dropout: float, img_out_dim: int = 256) -> None:
        super().__init__()
        self.cnn = BetterImageCNN(out_dim=img_out_dim)
        # Attention over multi-view features
        self.view_attn = nn.Sequential(nn.Linear(img_out_dim, img_out_dim // 4), nn.ReLU(inplace=True), nn.Linear(img_out_dim // 4, 1))
        self.svg_mlp = nn.Sequential(nn.Linear(svg_dim, 48), nn.ReLU(inplace=True))
        self.geom_mlp = nn.Sequential(nn.Linear(geom_dim, 48), nn.ReLU(inplace=True))
        fused_dim = img_out_dim * 2 + 48 + 48
        self.head = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fused_dim // 2, 1),
        )

    def forward(self, images: torch.Tensor, image_mask: torch.Tensor, svg: torch.Tensor, geom: torch.Tensor) -> torch.Tensor:
        b, n, c, h, w = images.shape
        img_feats = self.cnn(images.view(b * n, c, h, w)).view(b, n, -1)  # (B, N, D)
        # Attention-based view pooling
        raw_attn = self.view_attn(img_feats)  # (B, N, 1)
        raw_attn = raw_attn.masked_fill(image_mask.unsqueeze(-1) == 0, float("-inf"))
        attn_w = torch.softmax(raw_attn, dim=1)
        mask = image_mask.unsqueeze(-1)
        denom = (mask * attn_w).sum(dim=1).clamp_min(1e-6)
        weighted_pool = (img_feats * attn_w * mask).sum(dim=1) / denom
        # Max pool (mask-aware)
        max_pool = img_feats.masked_fill(mask == 0, float("-inf")).max(dim=1).values
        max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))
        fused = torch.cat([weighted_pool, max_pool, self.svg_mlp(svg), self.geom_mlp(geom)], dim=1)
        return self.head(fused).squeeze(1)


def feature_stats(examples: Sequence[Dict[str, Any]], key: str) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.vstack([ex[key] for ex in examples]).astype(np.float32)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean, std


def pair_metrics(probs: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    pred = probs >= 0.5
    gold = labels > 0.5
    tp = int(np.logical_and(pred, gold).sum())
    fp = int(np.logical_and(pred, ~gold).sum())
    fn = int(np.logical_and(~pred, gold).sum())
    tn = int(np.logical_and(~pred, ~gold).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"accuracy": (tp + tn) / len(labels), "precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def solve_assignment(score_matrix: np.ndarray) -> List[int]:
    n, m = score_matrix.shape
    dp: Dict[int, Tuple[float, List[int]]] = {0: (0.0, [])}
    for row in range(n):
        nxt = {}
        for mask, (score, assign) in dp.items():
            for col in range(m):
                if mask & (1 << col):
                    continue
                new_mask = mask | (1 << col)
                new_score = score + float(score_matrix[row, col])
                if new_mask not in nxt or new_score > nxt[new_mask][0]:
                    nxt[new_mask] = (new_score, assign + [col])
        dp = nxt
    return max(dp.values(), key=lambda item: item[0])[1] if dp else []


def assignment_metrics(probs: np.ndarray, examples: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    by_step = defaultdict(list)
    for i, ex in enumerate(examples):
        by_step[ex["step_key"]].append(i)
    correct = total = exact = 0
    for indices in by_step.values():
        svg_ids = sorted({examples[i]["svg_instance_id"] for i in indices})
        cand_ids = sorted({examples[i]["candidate_part_id"] for i in indices}, key=part_sort_key)
        row_of = {sid: r for r, sid in enumerate(svg_ids)}
        col_of = {pid: c for c, pid in enumerate(cand_ids)}
        scores = np.full((len(svg_ids), len(cand_ids)), -1e9, dtype=np.float32)
        gold = {}
        for i in indices:
            scores[row_of[examples[i]["svg_instance_id"]], col_of[examples[i]["candidate_part_id"]]] = probs[i]
            gold[examples[i]["svg_instance_id"]] = examples[i]["positive_part_id"]
        assign = solve_assignment(scores)
        pred = {svg_ids[r]: cand_ids[c] for r, c in enumerate(assign)}
        step_correct = sum(1 for sid, pid in pred.items() if gold.get(sid) == pid)
        correct += step_correct
        total += len(svg_ids)
        exact += int(step_correct == len(svg_ids))
    return {
        "steps": len(by_step),
        "instance_accuracy": correct / total if total else 0.0,
        "exact_match": exact / len(by_step) if by_step else 0.0,
        "correct_instances": correct,
        "total_instances": total,
    }


def assignment_metrics_equiv(
    probs: np.ndarray,
    examples: Sequence[Dict[str, Any]],
    equivalences: Dict[Tuple[str, str], EquivalenceLookup] | None,
) -> Dict[str, float] | None:
    if not equivalences:
        return None
    by_step = defaultdict(list)
    for i, ex in enumerate(examples):
        by_step[ex["step_key"]].append(i)
    correct = total = exact = 0
    for indices in by_step.values():
        svg_ids = sorted({examples[i]["svg_instance_id"] for i in indices})
        cand_ids = sorted({examples[i]["candidate_part_id"] for i in indices}, key=part_sort_key)
        row_of = {sid: r for r, sid in enumerate(svg_ids)}
        col_of = {pid: c for c, pid in enumerate(cand_ids)}
        scores = np.full((len(svg_ids), len(cand_ids)), -1e9, dtype=np.float32)
        gold = {}
        object_key = None
        for i in indices:
            ex = examples[i]
            scores[row_of[ex["svg_instance_id"]], col_of[ex["candidate_part_id"]]] = probs[i]
            gold[ex["svg_instance_id"]] = ex["positive_part_id"]
            object_key = (ex["category"], ex["name"])
        assign = solve_assignment(scores)
        pred = {svg_ids[r]: cand_ids[c] for r, c in enumerate(assign)}
        equiv = equivalences.get(object_key) if object_key else None
        if equiv is None:
            continue
        step_correct = sum(1 for sid, pid in pred.items() if equiv.equivalent(pid, gold.get(sid)))
        correct += step_correct
        total += len(svg_ids)
        exact += int(step_correct == len(svg_ids))
    return {
        "steps": len(by_step),
        "instance_accuracy": correct / total if total else 0.0,
        "exact_match": exact / len(by_step) if by_step else 0.0,
        "correct_instances": correct,
        "total_instances": total,
    }


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    out = []
    for batch in loader:
        logits = model(
            batch["images"].to(device),
            batch["image_mask"].to(device),
            batch["svg"].to(device),
            batch["geom"].to(device),
        )
        out.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(out) if out else np.asarray([])


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    total = 0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        logits = model(
            batch["images"].to(device),
            batch["image_mask"].to(device),
            batch["svg"].to(device),
            batch["geom"].to(device),
        )
        labels = batch["label"].to(device)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * len(labels)
        total += len(labels)
    return total_loss / max(total, 1)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = load_jsonl(Path(args.samples))
    equivalences = load_equivalences(Path(args.data_json))
    examples = make_pair_examples(samples, args.primitive_only, args.max_images, equivalences, args.equivalence_labels)
    train_examples, val_examples, test_examples, val_step_keys = split_examples(examples, args.val_fraction, args.seed)
    svg_mean, svg_std = feature_stats(train_examples, "svg_feature")
    geom_mean, geom_std = feature_stats(train_examples, "part_geometry")

    train_ds = GroundingDataset(
        train_examples,
        args.image_size,
        args.max_images,
        svg_mean,
        svg_std,
        geom_mean,
        geom_std,
        augment=args.augment,
        translate=args.augment_translate,
        scale_jitter=args.augment_scale,
        noise_std=args.augment_noise,
    )
    val_ds = GroundingDataset(val_examples, args.image_size, args.max_images, svg_mean, svg_std, geom_mean, geom_std)
    test_ds = GroundingDataset(test_examples, args.image_size, args.max_images, svg_mean, svg_std, geom_mean, geom_std)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    train_eval_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = TinyGroundingCNN(svg_dim=len(svg_mean), geom_dim=len(geom_mean), dropout=args.dropout).to(device)
    labels = np.asarray([ex["label"] for ex in train_examples], dtype=np.float32)
    pos_weight = torch.tensor([(len(labels) - labels.sum()) / max(labels.sum(), 1.0)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    best_state = None
    best_score = -1.0
    for epoch in range(1, args.epochs + 1):
        t_start = __import__("time").time()
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        elapsed = __import__("time").time() - t_start
        if epoch == args.epochs or epoch % max(args.eval_every, 1) == 0:
            val_probs = predict(model, val_loader, device) if val_examples else np.asarray([])
            assign = assignment_metrics(val_probs, val_examples) if val_examples else {"instance_accuracy": 0.0, "exact_match": 0.0}
            assign_equiv = assignment_metrics_equiv(val_probs, val_examples, equivalences) if val_examples else None
            score = (assign_equiv if args.equivalence_labels and assign_equiv is not None else assign)["instance_accuracy"]
            history.append(
                {
                    "epoch": epoch,
                    "loss": loss,
                    "val_assignment_instance_accuracy": assign["instance_accuracy"],
                    "val_exact": assign["exact_match"],
                    "val_equivalence_assignment_instance_accuracy": assign_equiv["instance_accuracy"] if assign_equiv else None,
                    "val_equivalence_exact": assign_equiv["exact_match"] if assign_equiv else None,
                    "selection_score": score,
                }
            )
            if score > best_score:
                best_score = score
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  epoch {epoch:3d} | loss={loss:.4f} | time={elapsed:.1f}s | val_acc={assign['instance_accuracy']:.4f} | best={best_score:.4f}", flush=True)
        else:
            print(f"  epoch {epoch:3d} | loss={loss:.4f} | time={elapsed:.1f}s", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    train_probs = predict(model, train_eval_loader, device)
    val_probs = predict(model, val_loader, device) if val_examples else np.asarray([])
    test_probs = predict(model, test_loader, device)
    train_labels = np.asarray([ex["label"] for ex in train_examples], dtype=np.float32)
    val_labels = np.asarray([ex["label"] for ex in val_examples], dtype=np.float32)
    test_labels = np.asarray([ex["label"] for ex in test_examples], dtype=np.float32)
    report = {
        "model": "tiny_cnn_geometry",
        "primitive_only": args.primitive_only,
        "num_train_pairs": len(train_examples),
        "num_val_pairs": len(val_examples),
        "num_test_pairs": len(test_examples),
        "positive_rate_train": float(train_labels.mean()),
        "positive_rate_val": float(val_labels.mean()) if len(val_labels) else None,
        "positive_rate_test": float(test_labels.mean()),
        "pair_metrics_train": pair_metrics(train_probs, train_labels),
        "pair_metrics_val": pair_metrics(val_probs, val_labels) if len(val_labels) else None,
        "pair_metrics_test": pair_metrics(test_probs, test_labels),
        "assignment_train": assignment_metrics(train_probs, train_examples),
        "assignment_val": assignment_metrics(val_probs, val_examples) if val_examples else None,
        "assignment_test": assignment_metrics(test_probs, test_examples),
        "assignment_equivalence_train": assignment_metrics_equiv(train_probs, train_examples, equivalences),
        "assignment_equivalence_val": assignment_metrics_equiv(val_probs, val_examples, equivalences) if val_examples else None,
        "assignment_equivalence_test": assignment_metrics_equiv(test_probs, test_examples, equivalences),
        "history": history,
        "selection": {
            "metric": "assignment_equivalence_val.instance_accuracy" if args.equivalence_labels else "assignment_val.instance_accuracy",
            "best_score": best_score,
            "val_fraction": args.val_fraction,
            "num_val_steps": len(val_step_keys),
        },
        "config": vars(args),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    torch.save(
        {
            "model_state": model.state_dict(),
            "svg_mean": svg_mean,
            "svg_std": svg_std,
            "geom_mean": geom_mean,
            "geom_std": geom_std,
            "config": vars(args),
        },
        args.model_output,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote {out}")
    print(f"Wrote {args.model_output}")


if __name__ == "__main__":
    main()
