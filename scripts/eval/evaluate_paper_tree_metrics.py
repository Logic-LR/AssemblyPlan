#!/usr/bin/env python3
"""Evaluate assembly trees with the IKEA-Manual paper metrics."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-json", default="main_data.json")
    parser.add_argument("--diagnostic-report", default="experiments/svg_assembly/reports/end_to_end_diagnostic_all_report.json")
    parser.add_argument("--output", default="experiments/svg_assembly/reports/paper_tree_metric_report.json")
    return parser.parse_args()


def build_tree_from_list(value: Any) -> Node:
    if isinstance(value, int):
        return Node.leaf(value)
    return Node.parent([build_tree_from_list(child) for child in value])


def parse_part_set(value: Any) -> PartSet:
    if isinstance(value, int):
        return frozenset([value])
    text = str(value)
    return frozenset(int(piece) for piece in text.split(",") if piece != "")


def part_count(tree: Node) -> int:
    return max(tree.parts) + 1 if tree.parts else 0


def single_step_tree(n_parts: int) -> Node:
    return Node.parent([Node.leaf(i) for i in range(n_parts)])


def nonleaf_nodes(tree: Node) -> List[Node]:
    out: List[Node] = []
    if tree.children:
        out.append(tree)
        for child in tree.children:
            out.extend(nonleaf_nodes(child))
    return out


def eval_tree(gt_tree: Node, pred_tree: Node) -> Dict[str, Dict[str, float]]:
    gt_nodes = nonleaf_nodes(gt_tree)
    pred_nodes = nonleaf_nodes(pred_tree)
    counts = {"simple": 0, "hard": 0}
    for gt in gt_nodes:
        for pred in pred_nodes:
            if gt.parts == pred.parts:
                counts["simple"] += 1
                # This follows the released code: leaf children are ignored for
                # the hard/children_set criterion.
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
    out: Dict[str, Dict[str, float]] = {}
    for criterion in ["simple", "hard"]:
        out[criterion] = {}
        for metric in ["precision", "recall", "f1"]:
            out[criterion][metric] = sum(row[criterion][metric] for row in rows) / len(rows) if rows else 0.0
    return out


def step_tree_from_child_specs(child_specs: Iterable[Iterable[Any]], n_parts: int) -> Node:
    known: Dict[PartSet, Node] = {frozenset([i]): Node.leaf(i) for i in range(n_parts)}
    for specs in child_specs:
        child_sets = [parse_part_set(spec) for spec in specs]
        if not child_sets:
            continue
        children = []
        for child_set in child_sets:
            children.append(known.get(child_set) or Node.parent([known[frozenset([i])] for i in sorted(child_set)]))
        parent = Node.parent(children)
        known[parent.parts] = parent
    root_set = frozenset(range(n_parts))
    if root_set in known:
        return known[root_set]
    return build_inclusion_tree(root_set, list(known.values()))


def build_inclusion_tree(root_set: PartSet, known_nodes: Sequence[Node]) -> Node:
    candidates = [node for node in known_nodes if len(node.parts) > 1 and node.parts < root_set]
    selected: List[Node] = []
    used: Set[int] = set()
    for node in sorted(candidates, key=lambda item: len(item.parts), reverse=True):
        if node.parts & used:
            continue
        selected.append(node)
        used.update(node.parts)
    for part in sorted(root_set - frozenset(used)):
        selected.append(Node.leaf(part))
    return Node.parent(selected)


def load_diagnostic_step_specs(path: Path) -> Dict[Tuple[str, str], List[List[str]]]:
    if not path.exists():
        return {}
    report = json.loads(path.read_text(encoding="utf-8"))
    records = report.get("records") or report.get("error_examples") or []
    out: Dict[Tuple[str, str], List[List[str]]] = {}
    for record in records:
        step_key = record["step_key"]
        category, name, _ = step_key.split("/")
        endpoints = sorted({part for pair in record.get("pred_connections", []) for part in pair})
        out.setdefault((category, name), []).append(endpoints)
    return out


def connection_step_specs_from_report(path: Path) -> Optional[Dict[Tuple[str, str], List[List[str]]]]:
    if not path.exists():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    records = report.get("records")
    if records is None:
        return None
    out: Dict[Tuple[str, str], List[List[str]]] = {}
    for record in records:
        category, name, _ = record["step_key"].split("/")
        endpoints = sorted({part for pair in record.get("pred_connections", []) for part in pair})
        out.setdefault((category, name), []).append(endpoints)
    return out


def evaluate_model(data: Sequence[Dict[str, Any]], pred_trees: Dict[Tuple[str, str], Node]) -> Dict[str, Any]:
    rows = []
    missing = []
    for obj in data:
        key = (obj["category"], obj["name"])
        gt_tree = build_tree_from_list(obj["assembly_tree"])
        pred_tree = pred_trees.get(key)
        if pred_tree is None:
            missing.append({"category": key[0], "name": key[1]})
            continue
        rows.append(eval_tree(gt_tree, pred_tree))
    return {"objects": len(rows), "missing": missing, "metrics": average_metrics(rows)}


def main() -> None:
    args = parse_args()
    data = json.loads(Path(args.data_json).read_text(encoding="utf-8"))

    single = {}
    gt_steps = {}
    for obj in data:
        key = (obj["category"], obj["name"])
        gt_tree = build_tree_from_list(obj["assembly_tree"])
        n_parts = part_count(gt_tree)
        single[key] = single_step_tree(n_parts)
        gt_steps[key] = step_tree_from_child_specs((step["parts"] for step in obj.get("steps", [])), n_parts)

    report = {
        "paper_metric_mapping": {
            "simple": "paper Simple Matching / released code no_children",
            "hard": "paper Hard Matching / released code children_set; leaf children ignored",
            "aggregation": "object-level precision/recall/F1 averaged over objects",
        },
        "single_step_reproduction": evaluate_model(data, single),
        "gt_step_tree_sanity": evaluate_model(data, gt_steps),
    }

    diagnostic_specs = connection_step_specs_from_report(Path(args.diagnostic_report))
    if diagnostic_specs is not None:
        pred_trees = {}
        for obj in data:
            key = (obj["category"], obj["name"])
            gt_tree = build_tree_from_list(obj["assembly_tree"])
            specs = diagnostic_specs.get(key)
            if specs is not None:
                pred_trees[key] = step_tree_from_child_specs(specs, part_count(gt_tree))
        report["current_connection_induced_tree"] = evaluate_model(data, pred_trees)
        report["current_connection_induced_tree"]["note"] = (
            "Uses current grounding+connection predictions to form each step node from predicted connection endpoints. "
            "This is not the same task as paper assembly plan generation because our input includes manual step SVGs."
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
