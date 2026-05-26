#!/usr/bin/env python3
"""Run an end-to-end diagnostic from grounding scores to step connections."""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from train.train_grounding_cnn import (
    GroundingDataset,
    LegacyGroundingCNN,
    TinyGroundingCNN,
    assignment_metrics,
    load_jsonl,
    make_pair_examples,
    predict,
    solve_assignment,
)
from train.train_simplified_connection_model import norm_pair, pair_features


Pair = Tuple[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", default="experiments/svg_assembly/datasets/grounding_samples.jsonl")
    parser.add_argument("--simplified-root", default="experiments/svg_assembly/simplified_svg")
    parser.add_argument("--grounding-model", default="experiments/svg_assembly/reports/grounding_cnn_improved_val_model.pt")
    parser.add_argument("--connection-model", default="experiments/svg_assembly/reports/simplified_connection_model.npz")
    parser.add_argument("--split", default="test", choices=["train", "test", "all"])
    parser.add_argument("--output", default="experiments/svg_assembly/reports/end_to_end_diagnostic_report.json")
    return parser.parse_args()


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))


def gt_pairs(connections: Sequence[Sequence[Any]]) -> set[Pair]:
    out: set[Pair] = set()
    for conn in connections:
        if len(conn) == 2:
            out.add(norm_pair(conn[0], conn[1]))
    return out


def group_indices_by_step(examples: Sequence[Dict[str, Any]]) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = defaultdict(list)
    for i, ex in enumerate(examples):
        groups[ex["step_key"]].append(i)
    return groups


def assignment_maps(
    probs: np.ndarray,
    examples: Sequence[Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, str]]]:
    pred_maps: Dict[str, Dict[str, str]] = {}
    gold_maps: Dict[str, Dict[str, str]] = {}
    by_step = group_indices_by_step(examples)
    for step_key in sorted(by_step):
        indices = by_step[step_key]
        svg_ids = sorted({examples[i]["svg_instance_id"] for i in indices})
        cand_ids = sorted({examples[i]["candidate_part_id"] for i in indices}, key=lambda value: (0, int(value)) if value.isdigit() else (1, value))
        row_of = {sid: r for r, sid in enumerate(svg_ids)}
        col_of = {pid: c for c, pid in enumerate(cand_ids)}
        scores = np.full((len(svg_ids), len(cand_ids)), -1e9, dtype=np.float32)
        gold: Dict[str, str] = {}
        for i in indices:
            ex = examples[i]
            scores[row_of[ex["svg_instance_id"]], col_of[ex["candidate_part_id"]]] = probs[i]
            gold[ex["svg_instance_id"]] = ex["positive_part_id"]
        assign = solve_assignment(scores)
        pred_maps[step_key] = {svg_ids[row]: cand_ids[col] for row, col in enumerate(assign)}
        gold_maps[step_key] = gold
    return pred_maps, gold_maps


def step_path(root: Path, step_key: str) -> Path:
    category, name, step = step_key.split("/")
    return root / category / name / step / "simplified_instances.json"


def score_connection_pairs(path: Path, model: Dict[str, np.ndarray]) -> List[Tuple[float, str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    instances = data.get("instances") or []
    out = []
    for a, b in combinations(instances, 2):
        feat = pair_features(a, b)
        scaled = (feat - model["mean"]) / model["std"]
        prob = float(sigmoid(scaled @ model["weights"] + model["bias"]))
        out.append((prob, str(a["id"]), str(b["id"])))
    return sorted(out, key=lambda item: item[0], reverse=True)


def connection_metrics(records: Sequence[Dict[str, Any]], key: str) -> Dict[str, Any]:
    tp = fp = fn = exact = 0
    for record in records:
        pred = set(map(tuple, record[key]))
        gold = set(map(tuple, record["gold_connections"]))
        tp += len(pred & gold)
        fp += len(pred - gold)
        fn += len(gold - pred)
        exact += int(pred == gold)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "steps": len(records),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": exact / len(records) if records else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def load_grounding_model(path: Path) -> Tuple[torch.nn.Module, Dict[str, Any], Dict[str, np.ndarray]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    stats = {
        "svg_mean": checkpoint["svg_mean"],
        "svg_std": checkpoint["svg_std"],
        "geom_mean": checkpoint["geom_mean"],
        "geom_std": checkpoint["geom_std"],
    }
    config = checkpoint["config"]
    state = checkpoint["model_state"]
    if any(key.startswith("cnn.stem.") for key in state):
        model: torch.nn.Module = TinyGroundingCNN(
            svg_dim=len(stats["svg_mean"]),
            geom_dim=len(stats["geom_mean"]),
            dropout=float(config.get("dropout", 0.0)),
        )
    elif "view_fc.0.weight" in state:
        model = LegacyGroundingCNN(
            svg_dim=len(stats["svg_mean"]),
            geom_dim=len(stats["geom_mean"]),
            dropout=float(config.get("dropout", 0.0)),
        )
    else:
        raise ValueError(f"Unrecognized grounding checkpoint architecture in {path}")
    model.load_state_dict(state)
    model.eval()
    return model, config, stats


def main() -> None:
    args = parse_args()
    grounding_model, grounding_config, stats = load_grounding_model(Path(args.grounding_model))
    samples = load_jsonl(Path(args.samples))
    examples = make_pair_examples(
        samples,
        primitive_only=bool(grounding_config.get("primitive_only", False)),
        max_images=int(grounding_config.get("max_images", 16)),
    )
    split_examples = examples if args.split == "all" else [ex for ex in examples if ex["split"] == args.split]
    ds = GroundingDataset(
        split_examples,
        int(grounding_config.get("image_size", 64)),
        int(grounding_config.get("max_images", 16)),
        stats["svg_mean"],
        stats["svg_std"],
        stats["geom_mean"],
        stats["geom_std"],
    )
    loader = DataLoader(ds, batch_size=int(grounding_config.get("batch_size", 64)), shuffle=False)
    grounding_probs = predict(grounding_model, loader, torch.device("cpu"))
    pred_maps, gold_maps = assignment_maps(grounding_probs, split_examples)

    connection_model = dict(np.load(args.connection_model))
    simplified_root = Path(args.simplified_root)
    by_step_samples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        step_key = f"{sample['category']}/{sample['name']}/step_{sample['step_id']}"
        if args.split == "all" or sample.get("split") == args.split:
            by_step_samples[step_key].append(sample)

    records: List[Dict[str, Any]] = []
    for step_key in sorted(pred_maps):
        step_samples = by_step_samples.get(step_key) or []
        if not step_samples:
            continue
        gold = gt_pairs(step_samples[0].get("gt_connections") or [])
        ranked_instance_pairs = score_connection_pairs(step_path(simplified_root, step_key), connection_model)
        k = min(len(gold), len(ranked_instance_pairs))
        pred_grounding = pred_maps[step_key]
        gold_grounding = gold_maps[step_key]
        pred_connections = []
        oracle_connections = []
        selected_connections = []
        for _, inst_a, inst_b in ranked_instance_pairs[:k]:
            item = {"instance_pair": [inst_a, inst_b], "score": _}
            if inst_a in pred_grounding and inst_b in pred_grounding:
                pred_pair = norm_pair(pred_grounding[inst_a], pred_grounding[inst_b])
                pred_connections.append(pred_pair)
                item["pred_part_pair"] = list(pred_pair)
            if inst_a in gold_grounding and inst_b in gold_grounding:
                oracle_pair = norm_pair(gold_grounding[inst_a], gold_grounding[inst_b])
                oracle_connections.append(oracle_pair)
                item["oracle_part_pair"] = list(oracle_pair)
            selected_connections.append(item)
        records.append(
            {
                "step_key": step_key,
                "num_instances": len(gold_grounding),
                "gold_connections": sorted(gold),
                "pred_connections": sorted(set(pred_connections)),
                "oracle_grounding_connections": sorted(set(oracle_connections)),
                "selected_connections": selected_connections,
                "connection_confidence_mean": float(np.mean([item["score"] for item in selected_connections])) if selected_connections else None,
                "connection_confidence_min": float(np.min([item["score"] for item in selected_connections])) if selected_connections else None,
                "grounding_correct": pred_grounding == gold_grounding,
                "pred_grounding": pred_grounding,
                "gold_grounding": gold_grounding,
            }
        )

    labels = np.asarray([ex["label"] for ex in split_examples], dtype=np.float32)
    report = {
        "split": args.split,
        "grounding_model": args.grounding_model,
        "connection_model": args.connection_model,
        "num_pair_examples": len(split_examples),
        "grounding_assignment": assignment_metrics(grounding_probs, split_examples),
        "grounding_pair_positive_rate": float(labels.mean()) if len(labels) else math.nan,
        "connection_with_predicted_grounding": connection_metrics(records, "pred_connections"),
        "connection_with_oracle_grounding": connection_metrics(records, "oracle_grounding_connections"),
        "num_steps": len(records),
        "records": records,
        "error_examples": [
            record
            for record in records
            if record["pred_connections"] != record["gold_connections"] or not record["grounding_correct"]
        ][:20],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
