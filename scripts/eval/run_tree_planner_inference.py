#!/usr/bin/env python3
"""Run object-level assembly-tree planner inference from observed part tokens."""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from eval.evaluate_paper_tree_metrics import Node
from train.train_tree_planner_baseline import plan_tree, sigmoid
from train.train_tree_planner_nn import MergeMLP, plan_tree_nn


SHAPE_TYPES = ["elongated_bar", "plate_like", "irregular", "point_or_line"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="JSON with part_tokens or observed_part_tokens.")
    parser.add_argument("--output", required=True, help="Where to write predicted_assembly_tree.json.")
    parser.add_argument(
        "--model",
        default="experiments/svg_assembly/reports/tree_planner_nn_svg_geometry_model.pt",
        help="Planner checkpoint: greedy .npz or neural .pt.",
    )
    parser.add_argument("--planner-type", choices=["auto", "greedy", "neural"], default="auto")
    parser.add_argument(
        "--feature-mode",
        choices=["geometry", "svg", "svg_geometry", "svg_composite", "svg_geometry_composite"],
        default=None,
    )
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--include-part-tokens", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def planner_type_from_path(path: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    if path.suffix.lower() == ".npz":
        return "greedy"
    if path.suffix.lower() == ".pt":
        return "neural"
    raise ValueError(f"Cannot infer planner type from {path}; pass --planner-type.")


def shape_distribution(value: Any) -> List[float]:
    if isinstance(value, list):
        out = [float(item) for item in value[: len(SHAPE_TYPES)]]
        return out + [0.0] * (len(SHAPE_TYPES) - len(out))
    if isinstance(value, dict):
        return [float(value.get(shape, 0.0)) for shape in SHAPE_TYPES]
    return [0.0] * len(SHAPE_TYPES)


def fixed_float_list(value: Any, length: int) -> List[float]:
    if isinstance(value, list):
        out = [float(item) for item in value[:length]]
        return out + [0.0] * (length - len(out))
    return [0.0] * length


def extract_tokens(payload: Any) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if isinstance(payload, list):
        return {}, payload
    if not isinstance(payload, dict):
        raise ValueError("Input JSON must be a list of tokens or an object containing part_tokens.")
    if isinstance(payload.get("observed_part_tokens"), list):
        return payload, payload["observed_part_tokens"]
    if isinstance(payload.get("part_tokens"), list):
        return payload, payload["part_tokens"]
    raise ValueError("Input JSON must contain part_tokens or observed_part_tokens.")


def normalize_record(payload: Any) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    meta, tokens = extract_tokens(payload)
    normalized: List[Dict[str, Any]] = []
    mapping: List[Dict[str, Any]] = []
    for internal_id, token in enumerate(tokens):
        if not isinstance(token, dict):
            raise ValueError(f"Part token {internal_id} is not an object.")
        original_id = token.get("part_id", token.get("observed_part_id", internal_id))
        normalized.append(
            {
                **token,
                "part_id": str(internal_id),
                "geometry_feature": fixed_float_list(token.get("geometry_feature"), 14),
                "svg_feature_mean": fixed_float_list(token.get("svg_feature_mean"), 17),
                "svg_feature_std": fixed_float_list(token.get("svg_feature_std"), 17),
                "svg_feature_count": int(token.get("svg_feature_count") or 0),
                "shape_distribution": shape_distribution(token.get("shape_distribution")),
                "shape_histogram": token.get("shape_histogram") or {},
            }
        )
        mapping.append(
            {
                "internal_id": internal_id,
                "part_id": original_id,
                "source_part_id": token.get("source_part_id"),
                "grounding_score": token.get("grounding_score"),
            }
        )
    composite_tokens = []
    for token in meta.get("composite_tokens") or []:
        if not isinstance(token, dict):
            continue
        composite_tokens.append(
            {
                **token,
                "part_ids": [str(part) for part in token.get("part_ids", [])],
                "svg_feature_mean": fixed_float_list(token.get("svg_feature_mean"), 17),
                "svg_feature_std": fixed_float_list(token.get("svg_feature_std"), 17),
                "svg_feature_count": int(token.get("svg_feature_count") or 0),
                "shape_distribution": shape_distribution(token.get("shape_distribution")),
                "shape_histogram": token.get("shape_histogram") or {},
            }
        )
    record = {
        "category": meta.get("category", "observed"),
        "name": meta.get("name", "parts"),
        "split": meta.get("split", "inference"),
        "num_parts": len(normalized),
        "part_tokens": normalized,
        "composite_tokens": composite_tokens,
    }
    return record, mapping


def load_greedy_model(path: Path, feature_mode: str | None, threshold: float | None) -> Dict[str, Any]:
    model = np.load(path, allow_pickle=True)
    stored_mode = str(model["feature_mode"].item() if hasattr(model["feature_mode"], "item") else model["feature_mode"])
    return {
        "planner_type": "greedy",
        "feature_mode": feature_mode or stored_mode,
        "weights": model["weights"].astype(np.float32),
        "bias": float(model["bias"]),
        "mean": model["mean"].astype(np.float32),
        "std": model["std"].astype(np.float32),
        "threshold": float(threshold if threshold is not None else model["threshold"]),
    }


def torch_load(path: Path, device: torch.device) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_neural_model(
    path: Path,
    feature_mode: str | None,
    threshold: float | None,
    requested_device: str,
) -> Dict[str, Any]:
    if requested_device == "cuda":
        device = torch.device("cuda")
    elif requested_device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch_load(path, device)
    model = MergeMLP(int(checkpoint["input_dim"]), int(checkpoint["hidden_dim"]), float(checkpoint["dropout"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return {
        "planner_type": "neural",
        "feature_mode": feature_mode or checkpoint["feature_mode"],
        "model": model,
        "mean": np.asarray(checkpoint["mean"], dtype=np.float32),
        "std": np.asarray(checkpoint["std"], dtype=np.float32),
        "threshold": float(threshold if threshold is not None else checkpoint["threshold"]),
        "device": device,
    }


def tree_to_list_internal(tree: Node) -> Any:
    if not tree.children:
        return int(next(iter(tree.parts)))
    return [tree_to_list_internal(child) for child in tree.children]


def tree_to_external(tree: Node, id_mapping: Sequence[Dict[str, Any]]) -> Any:
    if not tree.children:
        internal_id = int(next(iter(tree.parts)))
        return id_mapping[internal_id]["part_id"]
    return [tree_to_external(child, id_mapping) for child in tree.children]


def run_planner(record: Dict[str, Any], model_info: Dict[str, Any]) -> Node:
    if model_info["planner_type"] == "greedy":
        return plan_tree(
            record,
            model_info["feature_mode"],
            model_info["weights"],
            model_info["bias"],
            model_info["mean"],
            model_info["std"],
            model_info["threshold"],
        )
    return plan_tree_nn(
        record,
        model_info["feature_mode"],
        model_info["model"],
        model_info["mean"],
        model_info["std"],
        model_info["threshold"],
        model_info["device"],
    )


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    planner_kind = planner_type_from_path(model_path, args.planner_type)
    payload = load_json(Path(args.input))
    record, mapping = normalize_record(payload)
    if record["num_parts"] == 0:
        raise ValueError("No observed part tokens were provided.")

    if planner_kind == "greedy":
        model_info = load_greedy_model(model_path, args.feature_mode, args.threshold)
    else:
        model_info = load_neural_model(model_path, args.feature_mode, args.threshold, args.device)

    pred_tree = run_planner(record, model_info)
    report = {
        "input": str(Path(args.input).as_posix()),
        "model": str(model_path.as_posix()),
        "planner_type": model_info["planner_type"],
        "feature_mode": model_info["feature_mode"],
        "threshold": model_info["threshold"],
        "num_parts": record["num_parts"],
        "part_id_mapping": mapping,
        "predicted_assembly_tree": tree_to_external(pred_tree, mapping),
        "predicted_assembly_tree_internal": tree_to_list_internal(pred_tree),
    }
    if args.include_part_tokens:
        report["part_tokens"] = record["part_tokens"]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
