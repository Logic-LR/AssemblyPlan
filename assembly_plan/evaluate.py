"""Evaluation metrics for assembly tree prediction.

Wraps the IKEA-Manual paper metrics (Simple/Hard matching F1) and adds
connection-graph-aware diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Set

import numpy as np


# ---------------------------------------------------------------------------
# Tree data structure (same as evaluate_paper_tree_metrics.py)
# ---------------------------------------------------------------------------

PartSet = frozenset[int]


@dataclass
class Node:
    children: List["Node"]
    parts: PartSet

    @staticmethod
    def leaf(part: int) -> "Node":
        return Node(children=[], parts=frozenset([part]))

    @staticmethod
    def parent(children: Sequence["Node"]) -> "Node":
        parts: Set[int] = set()
        for child in children:
            parts.update(child.parts)
        return Node(children=list(children), parts=frozenset(parts))


# ---------------------------------------------------------------------------
# Tree parsing
# ---------------------------------------------------------------------------

def build_tree_from_list(value: Any) -> Node:
    """Convert nested list (main_data.json format) to Node tree."""
    if isinstance(value, int):
        return Node.leaf(value)
    return Node.parent([build_tree_from_list(child) for child in value])


def nonleaf_nodes(tree: Node) -> List[Node]:
    """Collect all non-leaf (internal) nodes."""
    out: List[Node] = []
    if tree.children:
        out.append(tree)
        for child in tree.children:
            out.extend(nonleaf_nodes(child))
    return out


# ---------------------------------------------------------------------------
# Core metrics: Simple / Hard matching
# ---------------------------------------------------------------------------

def eval_tree(gt_tree: Node, pred_tree: Node) -> Dict[str, Dict[str, float]]:
    """Compute Simple and Hard matching precision/recall/F1.

    Simple: predicted non-leaf node matches GT if same part set.
    Hard: same part set AND same child partition (ignoring leaf-only children).
    """
    gt_nodes = nonleaf_nodes(gt_tree)
    pred_nodes = nonleaf_nodes(pred_tree)
    counts = {"simple": 0, "hard": 0}

    for gt in gt_nodes:
        for pred in pred_nodes:
            if gt.parts == pred.parts:
                counts["simple"] += 1
                gt_child_sets = {child.parts for child in gt.children if len(child.parts) > 1}
                pred_child_sets = {child.parts for child in pred.children if len(child.parts) > 1}
                if gt_child_sets == pred_child_sets:
                    counts["hard"] += 1

    result: Dict[str, Dict[str, float]] = {}
    for key, matched in counts.items():
        precision = matched / len(pred_nodes) if pred_nodes else 0.0
        recall = matched / len(gt_nodes) if gt_nodes else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result[key] = {"precision": precision, "recall": recall, "f1": f1}
    return result


def average_metrics(rows: Sequence[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    """Macro-average metrics across objects."""
    out: Dict[str, Dict[str, float]] = {}
    for criterion in ["simple", "hard"]:
        out[criterion] = {}
        for metric in ["precision", "recall", "f1"]:
            vals = [row[criterion][metric] for row in rows]
            out[criterion][metric] = sum(vals) / len(vals) if vals else 0.0
    return out


# ---------------------------------------------------------------------------
# Connection-graph diagnostics
# ---------------------------------------------------------------------------

def connection_accuracy(gt_tree: Node, pred_tree: Node,
                        edges: List[tuple]) -> Dict[str, float]:
    """Check whether predicted merges respect the connection graph.

    For each merge step (non-leaf node), check if the merged parts
    have at least one edge between them in the connection graph.
    """
    edge_set = set()
    for a, b in edges:
        edge_set.add((min(a, b), max(a, b)))

    gt_nodes = nonleaf_nodes(gt_tree)
    pred_nodes = nonleaf_nodes(pred_tree)

    def has_connection(node: Node) -> bool:
        """Check if any two parts in this node are connected."""
        parts = sorted(node.parts)
        for i in range(len(parts)):
            for j in range(i + 1, len(parts)):
                if (parts[i], parts[j]) in edge_set:
                    return True
        return False

    gt_connected = sum(1 for n in gt_nodes if has_connection(n))
    pred_connected = sum(1 for n in pred_nodes if has_connection(n))

    return {
        "gt_connection_rate": gt_connected / len(gt_nodes) if gt_nodes else 0.0,
        "pred_connection_rate": pred_connected / len(pred_nodes) if pred_nodes else 0.0,
        "gt_nodes": len(gt_nodes),
        "pred_nodes": len(pred_nodes),
    }


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------

def evaluate_predictions(
    predictions: List[Any],
    ground_truths: List[Any],
    connection_relations: List[List[tuple]],
    categories: List[str] = None,
) -> Dict[str, Any]:
    """Evaluate a batch of predicted trees against ground truth.

    Args:
        predictions: list of predicted trees (nested list format)
        ground_truths: list of GT trees (nested list format)
        connection_relations: list of edge lists per object
        categories: optional category labels for per-category breakdown

    Returns:
        dict with overall and per-category metrics
    """
    all_metrics = []
    per_category: Dict[str, List] = {}

    for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        gt_tree = build_tree_from_list(gt)
        pred_tree = build_tree_from_list(pred)

        m = eval_tree(gt_tree, pred_tree)
        conn = connection_accuracy(gt_tree, pred_tree, connection_relations[i])
        m["connection"] = conn
        all_metrics.append(m)

        if categories:
            cat = categories[i]
            per_category.setdefault(cat, []).append(m)

    result = {
        "overall": average_metrics(all_metrics),
        "count": len(all_metrics),
    }

    # Per-category breakdown
    if categories:
        result["per_category"] = {}
        for cat, cat_metrics in per_category.items():
            result["per_category"][cat] = {
                "metrics": average_metrics(cat_metrics),
                "count": len(cat_metrics),
            }

    return result


def format_metrics(metrics: Dict[str, Any]) -> str:
    """Format metrics dict as a readable string."""
    lines = []
    overall = metrics["overall"]
    lines.append(f"Overall ({metrics['count']} objects):")
    for crit in ["simple", "hard"]:
        m = overall[crit]
        lines.append(f"  {crit.capitalize():6s}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}")

    if "per_category" in metrics:
        lines.append("")
        for cat, cat_data in sorted(metrics["per_category"].items()):
            m = cat_data["metrics"]
            lines.append(f"  {cat:10s} ({cat_data['count']})  "
                         f"Simple F1={m['simple']['f1']:.3f}  Hard F1={m['hard']['f1']:.3f}")

    return "\n".join(lines)
