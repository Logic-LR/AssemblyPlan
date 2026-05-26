#!/usr/bin/env python3
"""Export predicted assembly trees and equivalence-aware diagnostics."""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from eval.evaluate_paper_tree_metrics import (
    Node,
    build_tree_from_list,
    eval_tree,
    part_count,
    step_tree_from_child_specs,
)


PartToken = str
Pair = Tuple[PartToken, PartToken]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-json", default="main_data.json")
    parser.add_argument("--diagnostic-report", default="experiments/svg_assembly/reports/end_to_end_diagnostic_all_report.json")
    parser.add_argument("--output-report", default="experiments/svg_assembly/reports/equivalence_and_tree_export_report.json")
    parser.add_argument("--output-dir", default="experiments/svg_assembly/predicted_assembly_trees")
    return parser.parse_args()


def part_sort_key(value: str) -> Tuple[int, Any]:
    return (0, int(value)) if value.isdigit() else (1, value)


def parse_token(value: Any) -> Tuple[int, ...]:
    if isinstance(value, int):
        return (value,)
    return tuple(sorted((int(piece) for piece in str(value).split(",") if piece != ""), key=int))


def normalize_token(value: Any) -> PartToken:
    return ",".join(str(piece) for piece in parse_token(value))


def normalize_pair(pair: Sequence[Any]) -> Pair:
    a, b = normalize_token(pair[0]), normalize_token(pair[1])
    return (a, b) if (len(parse_token(a)), a) <= (len(parse_token(b)), b) else (b, a)


def tree_to_list(tree: Node) -> Any:
    if not tree.children:
        return next(iter(tree.parts))
    return [tree_to_list(child) for child in tree.children]


class DSU:
    def __init__(self) -> None:
        self.parent: Dict[int, int] = {}

    def find(self, item: int) -> int:
        if item not in self.parent:
            self.parent[item] = item
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


class Equivalence:
    def __init__(self, n_parts: int, relation: Dict[str, List[str]]) -> None:
        self.n_parts = n_parts
        self.dsu = DSU()
        for part in range(n_parts):
            self.dsu.find(part)
        for key, values in (relation or {}).items():
            key_parts = parse_token(key)
            for value in values:
                value_parts = parse_token(value)
                if len(key_parts) == 1 and len(value_parts) == 1:
                    self.dsu.union(key_parts[0], value_parts[0])

    def signature(self, token: Any) -> Tuple[int, ...]:
        return tuple(sorted(self.dsu.find(part) for part in parse_token(token)))

    def token_equiv(self, a: Any, b: Any) -> bool:
        if normalize_token(a) == normalize_token(b):
            return True
        return self.signature(a) == self.signature(b)

    def pair_equiv(self, pred: Sequence[Any], gold: Sequence[Any]) -> bool:
        pa = normalize_pair(pred)
        ga = normalize_pair(gold)
        return (
            self.token_equiv(pa[0], ga[0])
            and self.token_equiv(pa[1], ga[1])
        ) or (
            self.token_equiv(pa[0], ga[1])
            and self.token_equiv(pa[1], ga[0])
        )


def max_pair_matches(pred_pairs: Sequence[Sequence[Any]], gold_pairs: Sequence[Sequence[Any]], equiv: Equivalence) -> int:
    graph: List[List[int]] = []
    for pred in pred_pairs:
        graph.append([j for j, gold in enumerate(gold_pairs) if equiv.pair_equiv(pred, gold)])
    match_to_pred: Dict[int, int] = {}

    def dfs(i: int, seen: Set[int]) -> bool:
        for j in graph[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in match_to_pred or dfs(match_to_pred[j], seen):
                match_to_pred[j] = i
                return True
        return False

    return sum(1 for i in range(len(graph)) if dfs(i, set()))


def pair_set_metrics(records: Sequence[Dict[str, Any]], equiv_by_object: Dict[Tuple[str, str], Equivalence]) -> Dict[str, Any]:
    strict_tp = strict_fp = strict_fn = strict_exact = 0
    equiv_tp = equiv_fp = equiv_fn = equiv_exact = 0
    for record in records:
        category, name, _ = record["step_key"].split("/")
        equiv = equiv_by_object[(category, name)]
        pred = [normalize_pair(pair) for pair in record.get("pred_connections", [])]
        gold = [normalize_pair(pair) for pair in record.get("gold_connections", [])]
        pred_set, gold_set = set(pred), set(gold)
        strict_tp += len(pred_set & gold_set)
        strict_fp += len(pred_set - gold_set)
        strict_fn += len(gold_set - pred_set)
        strict_exact += int(pred_set == gold_set)
        matches = max_pair_matches(pred, gold, equiv)
        equiv_tp += matches
        equiv_fp += max(0, len(pred) - matches)
        equiv_fn += max(0, len(gold) - matches)
        equiv_exact += int(matches == len(pred) == len(gold))

    def make(tp: int, fp: int, fn: int, exact: int) -> Dict[str, float]:
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "exact_match": exact / len(records) if records else 0.0,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    return {"strict": make(strict_tp, strict_fp, strict_fn, strict_exact), "equivalence_aware": make(equiv_tp, equiv_fp, equiv_fn, equiv_exact)}


def grounding_metrics(records: Sequence[Dict[str, Any]], equiv_by_object: Dict[Tuple[str, str], Equivalence]) -> Dict[str, Any]:
    strict_correct = equiv_correct = total = 0
    strict_exact = equiv_exact = 0
    harmless_swaps = 0
    for record in records:
        category, name, _ = record["step_key"].split("/")
        equiv = equiv_by_object[(category, name)]
        pred = record.get("pred_grounding", {})
        gold = record.get("gold_grounding", {})
        step_strict = True
        step_equiv = True
        for svg_id, gold_token in gold.items():
            pred_token = pred.get(svg_id)
            strict_ok = normalize_token(pred_token) == normalize_token(gold_token) if pred_token is not None else False
            equiv_ok = equiv.token_equiv(pred_token, gold_token) if pred_token is not None else False
            strict_correct += int(strict_ok)
            equiv_correct += int(equiv_ok)
            total += 1
            step_strict = step_strict and strict_ok
            step_equiv = step_equiv and equiv_ok
        strict_exact += int(step_strict)
        equiv_exact += int(step_equiv)
        harmless_swaps += int((not step_strict) and step_equiv)
    return {
        "steps": len(records),
        "instances": total,
        "strict_instance_accuracy": strict_correct / total if total else 0.0,
        "equivalence_instance_accuracy": equiv_correct / total if total else 0.0,
        "strict_exact_match": strict_exact / len(records) if records else 0.0,
        "equivalence_exact_match": equiv_exact / len(records) if records else 0.0,
        "strict_wrong_but_equivalent_steps": harmless_swaps,
    }


def classify_record(record: Dict[str, Any], equiv: Equivalence) -> str:
    pred = [normalize_pair(pair) for pair in record.get("pred_connections", [])]
    gold = [normalize_pair(pair) for pair in record.get("gold_connections", [])]
    oracle = [normalize_pair(pair) for pair in record.get("oracle_grounding_connections", [])]
    strict_ok = set(pred) == set(gold)
    equiv_ok = max_pair_matches(pred, gold, equiv) == len(pred) == len(gold)
    oracle_ok = set(oracle) == set(gold)
    oracle_equiv_ok = max_pair_matches(oracle, gold, equiv) == len(oracle) == len(gold)
    pred_ground = record.get("pred_grounding", {})
    gold_ground = record.get("gold_grounding", {})
    grounding_strict = pred_ground == gold_ground
    grounding_equiv = all(equiv.token_equiv(pred_ground.get(svg_id), gold_token) for svg_id, gold_token in gold_ground.items())
    if strict_ok:
        return "correct"
    if equiv_ok:
        return "equivalent_connection_correct"
    if (not grounding_strict) and grounding_equiv:
        return "id_swap_only"
    if oracle_ok or oracle_equiv_ok:
        return "grounding_caused_connection_error"
    return "connection_model_error"


def child_specs_from_records(records: Sequence[Dict[str, Any]]) -> List[List[str]]:
    specs = []
    for record in sorted(records, key=lambda item: int(item["step_key"].rsplit("_", 1)[-1]) if item["step_key"].rsplit("_", 1)[-1].isdigit() else item["step_key"]):
        endpoints = sorted({normalize_token(part) for pair in record.get("pred_connections", []) for part in pair}, key=part_sort_key)
        if endpoints:
            specs.append(endpoints)
    return specs


def main() -> None:
    args = parse_args()
    data = json.loads(Path(args.data_json).read_text(encoding="utf-8"))
    diagnostic = json.loads(Path(args.diagnostic_report).read_text(encoding="utf-8"))
    records = diagnostic.get("records") or []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    object_data = {(obj["category"], obj["name"]): obj for obj in data}
    equiv_by_object: Dict[Tuple[str, str], Equivalence] = {}
    for key, obj in object_data.items():
        gt_tree = build_tree_from_list(obj["assembly_tree"])
        equiv_by_object[key] = Equivalence(part_count(gt_tree), obj.get("geometric_equivalence_relation") or {})

    records_by_object: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        category, name, _ = record["step_key"].split("/")
        records_by_object[(category, name)].append(record)

    exported = []
    tree_metric_rows = []
    class_counts = Counter()
    for key, obj in sorted(object_data.items()):
        obj_records = records_by_object.get(key, [])
        if not obj_records:
            continue
        gt_tree = build_tree_from_list(obj["assembly_tree"])
        pred_tree = step_tree_from_child_specs(child_specs_from_records(obj_records), part_count(gt_tree))
        tree_metrics = eval_tree(gt_tree, pred_tree)
        tree_metric_rows.append(tree_metrics)
        equiv = equiv_by_object[key]
        for record in obj_records:
            class_counts[classify_record(record, equiv)] += 1
        category, name = key
        obj_out_dir = output_dir / category / name
        obj_out_dir.mkdir(parents=True, exist_ok=True)
        obj_payload = {
            "category": category,
            "name": name,
            "predicted_assembly_tree": tree_to_list(pred_tree),
            "ground_truth_assembly_tree": obj["assembly_tree"],
            "paper_tree_metrics": tree_metrics,
            "grounding": grounding_metrics(obj_records, equiv_by_object),
            "connections": pair_set_metrics(obj_records, equiv_by_object),
            "steps": obj_records,
        }
        (obj_out_dir / "assembly_tree_prediction.json").write_text(
            json.dumps(obj_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        exported.append(
            {
                "category": category,
                "name": name,
                "path": str((obj_out_dir / "assembly_tree_prediction.json").as_posix()),
                "paper_tree_metrics": tree_metrics,
            }
        )

    def avg_tree(metric_name: str, field: str) -> float:
        return sum(row[metric_name][field] for row in tree_metric_rows) / len(tree_metric_rows) if tree_metric_rows else 0.0

    report = {
        "source_diagnostic": args.diagnostic_report,
        "num_objects_exported": len(exported),
        "num_step_records": len(records),
        "grounding": grounding_metrics(records, equiv_by_object),
        "connections": pair_set_metrics(records, equiv_by_object),
        "error_class_counts": dict(class_counts),
        "paper_tree_metrics_from_exported_predictions": {
            "simple": {
                "precision": avg_tree("simple", "precision"),
                "recall": avg_tree("simple", "recall"),
                "f1": avg_tree("simple", "f1"),
            },
            "hard": {
                "precision": avg_tree("hard", "precision"),
                "recall": avg_tree("hard", "recall"),
                "f1": avg_tree("hard", "f1"),
            },
        },
        "exported_predictions": exported,
        "notes": [
            "Equivalence-aware grounding treats primitive IDs in the same geometric-equivalence class as interchangeable.",
            "Composite tokens are equivalent when their primitive equivalence-class signatures match.",
            "Paper tree metrics remain optimistic here because the predictions are induced from manual step SVGs.",
        ],
    }
    out = Path(args.output_report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "index.json").write_text(json.dumps(exported, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote {out}")
    print(f"Wrote {output_dir / 'index.json'}")


if __name__ == "__main__":
    main()
