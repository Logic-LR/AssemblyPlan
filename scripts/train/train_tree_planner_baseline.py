#!/usr/bin/env python3
"""Train a small SVG/geometry baseline for object-level assembly-tree generation."""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from eval.evaluate_paper_tree_metrics import (
    Node,
    PartSet,
    average_metrics,
    build_tree_from_list,
    eval_tree,
    part_count,
    step_tree_from_child_specs,
)


Cluster = frozenset[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="experiments/svg_assembly/datasets/tree_generation_dataset.json")
    parser.add_argument(
        "--feature-mode",
        choices=["geometry", "svg", "svg_geometry", "svg_composite", "svg_geometry_composite"],
        default="svg_geometry",
    )
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="experiments/svg_assembly/reports/tree_planner_svg_geometry_report.json")
    parser.add_argument("--model-output", default="experiments/svg_assembly/reports/tree_planner_svg_geometry_model.npz")
    return parser.parse_args()


def part_sort_key(value: Any) -> Tuple[int, Any]:
    text = str(value)
    return (0, int(text)) if text.isdigit() else (1, text)


def parse_part_set(values: Iterable[Any]) -> Cluster:
    return frozenset(int(value) for value in values)


def cluster_token(cluster: Cluster) -> str:
    return ",".join(str(part) for part in sorted(cluster))


def node_actions(node: Node) -> List[Dict[str, Any]]:
    if not node.children:
        return []
    actions: List[Dict[str, Any]] = []
    for child in node.children:
        actions.extend(node_actions(child))
    actions.append({"parent": node.parts, "children": [child.parts for child in node.children]})
    return actions


def split_records(records: Sequence[Dict[str, Any]], val_fraction: float, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    train_records = [record for record in records if record.get("split") == "train"]
    test_records = [record for record in records if record.get("split") == "test"]
    rng = random.Random(seed)
    rng.shuffle(train_records)
    val_count = max(1, round(len(train_records) * val_fraction)) if len(train_records) > 1 and val_fraction > 0 else 0
    val_records = train_records[:val_count]
    fit_records = train_records[val_count:]
    return fit_records, val_records, test_records


def part_feature(token: Dict[str, Any], mode: str) -> np.ndarray:
    geom = np.asarray(token.get("geometry_feature") or [], dtype=np.float32)
    svg = np.asarray(token.get("svg_feature_mean") or [], dtype=np.float32)
    shape = np.asarray(token.get("shape_distribution") or [], dtype=np.float32)
    svg_count = np.asarray([math.log1p(float(token.get("svg_feature_count") or 0))], dtype=np.float32)
    if mode == "geometry":
        return geom
    if mode in {"svg", "svg_composite"}:
        return np.concatenate([svg, shape, svg_count]).astype(np.float32)
    return np.concatenate([geom, svg, shape, svg_count]).astype(np.float32)


def part_feature_map(record: Dict[str, Any], mode: str) -> Dict[int, np.ndarray]:
    return {int(token["part_id"]): part_feature(token, mode) for token in record["part_tokens"]}


def uses_composite_features(mode: str) -> bool:
    return mode.endswith("_composite")


def composite_token_feature(token: Dict[str, Any]) -> np.ndarray:
    svg = np.asarray(token.get("svg_feature_mean") or [0.0] * 17, dtype=np.float32)
    shape = np.asarray(token.get("shape_distribution") or [0.0] * 4, dtype=np.float32)
    svg_count = np.asarray([math.log1p(float(token.get("svg_feature_count") or 0))], dtype=np.float32)
    has_prototype = np.asarray([1.0 if int(token.get("svg_feature_count") or 0) > 0 else 0.0], dtype=np.float32)
    return np.concatenate([svg, shape, svg_count, has_prototype]).astype(np.float32)


def composite_feature_map(record: Dict[str, Any], mode: str) -> Dict[Cluster, np.ndarray]:
    if not uses_composite_features(mode):
        return {}
    out: Dict[Cluster, np.ndarray] = {}
    for token in record.get("composite_tokens") or []:
        cluster = frozenset(int(part) for part in token.get("part_ids") or [])
        if len(cluster) > 1:
            out[cluster] = composite_token_feature(token)
    return out


def empty_composite_feature() -> np.ndarray:
    return np.zeros(23, dtype=np.float32)


def cluster_repr(cluster: Cluster, features: Dict[int, np.ndarray], composites: Dict[Cluster, np.ndarray] | None = None) -> np.ndarray:
    arr = np.vstack([features[part] for part in sorted(cluster)]).astype(np.float32)
    base = np.concatenate(
        [
            arr.mean(axis=0),
            arr.max(axis=0),
            arr.min(axis=0),
            np.asarray([len(cluster), math.log1p(len(cluster))], dtype=np.float32),
        ]
    )
    if composites is None:
        return base
    return np.concatenate([base, composites.get(cluster, empty_composite_feature())]).astype(np.float32)


def pair_feature(
    a: Cluster,
    b: Cluster,
    features: Dict[int, np.ndarray],
    composites: Dict[Cluster, np.ndarray] | None = None,
) -> np.ndarray:
    if min(a) > min(b):
        a, b = b, a
    ra = cluster_repr(a, features, composites)
    rb = cluster_repr(b, features, composites)
    union = cluster_repr(a | b, features, composites)
    return np.concatenate([ra, rb, np.abs(ra - rb), ra * rb, union]).astype(np.float32)


def pair_key(a: Cluster, b: Cluster) -> Tuple[Cluster, Cluster]:
    return (a, b) if tuple(sorted(a)) <= tuple(sorted(b)) else (b, a)


def training_examples_for_record(record: Dict[str, Any], mode: str) -> Tuple[List[np.ndarray], List[float]]:
    tree = build_tree_from_list(record["assembly_tree"])
    features = part_feature_map(record, mode)
    composites = composite_feature_map(record, mode)
    current: set[Cluster] = {frozenset([part]) for part in range(part_count(tree))}
    xs: List[np.ndarray] = []
    ys: List[float] = []
    for action in node_actions(tree):
        children = [frozenset(child) for child in action["children"]]
        positives = {pair_key(a, b) for i, a in enumerate(children) for b in children[i + 1 :]}
        clusters = sorted(current, key=lambda item: (len(item), tuple(sorted(item))))
        for i, a in enumerate(clusters):
            for b in clusters[i + 1 :]:
                xs.append(pair_feature(a, b, features, composites))
                ys.append(1.0 if pair_key(a, b) in positives else 0.0)
        for child in children:
            current.discard(child)
        current.add(frozenset(action["parent"]))
    return xs, ys


def build_pair_dataset(records: Sequence[Dict[str, Any]], mode: str) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[np.ndarray] = []
    ys: List[float] = []
    for record in records:
        rec_x, rec_y = training_examples_for_record(record, mode)
        xs.extend(rec_x)
        ys.extend(rec_y)
    return np.vstack(xs).astype(np.float32), np.asarray(ys, dtype=np.float32)


def standardize(train_x: np.ndarray, *arrays: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean, std, [((arr - mean) / std).astype(np.float32) for arr in arrays]


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))


def train_logreg(x: np.ndarray, y: np.ndarray, epochs: int, lr: float, l2: float, seed: int) -> Tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 0.01, size=x.shape[1]).astype(np.float32)
    b = 0.0
    pos_weight = (len(y) - float(y.sum())) / max(float(y.sum()), 1.0)
    weights = np.where(y > 0.5, pos_weight, 1.0).astype(np.float32)
    for _ in range(epochs):
        probs = sigmoid(x @ w + b)
        grad = (probs - y) * weights
        w -= lr * ((x.T @ grad) / len(y) + l2 * w)
        b -= lr * float(grad.mean())
    return w, float(b)


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


def connected_components(clusters: Sequence[Cluster], edges: Sequence[Tuple[Cluster, Cluster]]) -> List[List[Cluster]]:
    parent = {cluster: cluster for cluster in clusters}

    def find(item: Cluster) -> Cluster:
        if parent[item] != item:
            parent[item] = find(parent[item])
        return parent[item]

    def union(a: Cluster, b: Cluster) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in edges:
        union(a, b)
    groups: Dict[Cluster, List[Cluster]] = {}
    for cluster in clusters:
        groups.setdefault(find(cluster), []).append(cluster)
    return list(groups.values())


def plan_tree(
    record: Dict[str, Any],
    mode: str,
    weights: np.ndarray,
    bias: float,
    mean: np.ndarray,
    std: np.ndarray,
    threshold: float,
) -> Node:
    features = part_feature_map(record, mode)
    composites = composite_feature_map(record, mode)
    current: set[Cluster] = {frozenset([part]) for part in range(int(record["num_parts"]))}
    child_specs: List[List[str]] = []
    max_steps = max(1, int(record["num_parts"]) * 2)
    for _ in range(max_steps):
        if len(current) <= 1:
            break
        clusters = sorted(current, key=lambda item: (len(item), tuple(sorted(item))))
        scored: List[Tuple[float, Cluster, Cluster]] = []
        for i, a in enumerate(clusters):
            for b in clusters[i + 1 :]:
                feat = (pair_feature(a, b, features, composites) - mean) / std
                prob = float(sigmoid(feat @ weights + bias))
                scored.append((prob, a, b))
        if not scored:
            break
        scored.sort(key=lambda item: item[0], reverse=True)
        best_prob, best_a, best_b = scored[0]
        edges = [(a, b) for prob, a, b in scored if prob >= threshold]
        group = [best_a, best_b]
        if edges:
            comps = connected_components(clusters, edges)
            best_set = best_a | best_b
            for comp in comps:
                comp_union = frozenset().union(*comp)
                if best_set <= comp_union:
                    group = comp
                    break
        if len(group) < 2:
            group = [best_a, best_b]
        parent = frozenset().union(*group)
        child_specs.append([cluster_token(cluster) for cluster in sorted(group, key=lambda item: (len(item), tuple(sorted(item))))])
        for cluster in group:
            current.discard(cluster)
        current.add(parent)
    return step_tree_from_child_specs(child_specs, int(record["num_parts"]))


def evaluate_records(
    records: Sequence[Dict[str, Any]],
    mode: str,
    weights: np.ndarray,
    bias: float,
    mean: np.ndarray,
    std: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    rows = []
    examples = []
    for record in records:
        gt = build_tree_from_list(record["assembly_tree"])
        pred = plan_tree(record, mode, weights, bias, mean, std, threshold)
        metrics = eval_tree(gt, pred)
        rows.append(metrics)
        if len(examples) < 10:
            examples.append(
                {
                    "category": record["category"],
                    "name": record["name"],
                    "metrics": metrics,
                }
            )
    return {"objects": len(records), "metrics": average_metrics(rows), "examples": examples}


def tune_threshold(
    records: Sequence[Dict[str, Any]],
    mode: str,
    weights: np.ndarray,
    bias: float,
    mean: np.ndarray,
    std: np.ndarray,
) -> Tuple[float, Dict[str, Any]]:
    best_threshold = 0.5
    best_eval: Dict[str, Any] | None = None
    best_score = -1.0
    for threshold in np.linspace(0.1, 0.9, 17):
        result = evaluate_records(records, mode, weights, bias, mean, std, float(threshold))
        score = result["metrics"]["hard"]["f1"]
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_eval = result
    assert best_eval is not None
    return best_threshold, best_eval


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    records = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    fit_records, val_records, test_records = split_records(records, args.val_fraction, args.seed)
    train_x, train_y = build_pair_dataset(fit_records, args.feature_mode)
    val_x, val_y = build_pair_dataset(val_records, args.feature_mode) if val_records else (train_x, train_y)
    test_x, test_y = build_pair_dataset(test_records, args.feature_mode)
    mean, std, [train_xs, val_xs, test_xs] = standardize(train_x, train_x, val_x, test_x)
    weights, bias = train_logreg(train_xs, train_y, args.epochs, args.lr, args.l2, args.seed)
    train_probs = sigmoid(train_xs @ weights + bias)
    val_probs = sigmoid(val_xs @ weights + bias)
    test_probs = sigmoid(test_xs @ weights + bias)
    threshold, val_tree_eval = tune_threshold(val_records or fit_records, args.feature_mode, weights, bias, mean, std)

    report = {
        "model": "greedy_cluster_tree_planner",
        "feature_mode": args.feature_mode,
        "dataset": args.dataset,
        "splits": {
            "fit_objects": len(fit_records),
            "val_objects": len(val_records),
            "test_objects": len(test_records),
            "all_objects": len(records),
        },
        "pair_examples": {
            "train": int(len(train_y)),
            "val": int(len(val_y)),
            "test": int(len(test_y)),
            "positive_rate_train": float(train_y.mean()),
            "positive_rate_val": float(val_y.mean()) if len(val_y) else None,
            "positive_rate_test": float(test_y.mean()) if len(test_y) else None,
        },
        "pair_metrics": {
            "train": pair_metrics(train_probs, train_y),
            "val": pair_metrics(val_probs, val_y),
            "test": pair_metrics(test_probs, test_y),
        },
        "threshold": {
            "selected": threshold,
            "selection_metric": "val hard F1",
            "val_tree_eval": val_tree_eval,
        },
        "tree_metrics": {
            "train": evaluate_records(fit_records, args.feature_mode, weights, bias, mean, std, threshold),
            "val": evaluate_records(val_records, args.feature_mode, weights, bias, mean, std, threshold) if val_records else None,
            "test": evaluate_records(test_records, args.feature_mode, weights, bias, mean, std, threshold),
            "all": evaluate_records(records, args.feature_mode, weights, bias, mean, std, threshold),
        },
        "notes": [
            "This is an object-level planner: it receives all primitive part tokens and predicts a tree without reading per-step SVGs at inference.",
            "SVG-enhanced mode uses aggregated primitive SVG prototypes learned from manual instances.",
            "The greedy connected-component decoder is a baseline, not a final tree decoder.",
        ],
        "config": vars(args),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez(args.model_output, weights=weights, bias=bias, mean=mean, std=std, threshold=threshold, feature_mode=args.feature_mode)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote {out}")
    print(f"Wrote {args.model_output}")


if __name__ == "__main__":
    main()
