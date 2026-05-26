#!/usr/bin/env python3
"""Summarize SVG assembly experiment reports into JSON and Markdown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", default="experiments/svg_assembly/reports")
    parser.add_argument("--grounding-summary", default="experiments/svg_assembly/datasets/grounding_summary.json")
    parser.add_argument("--output-json", default="experiments/svg_assembly/experiment_summary.json")
    parser.add_argument("--output-md", default="experiments/svg_assembly/experiment_summary.md")
    return parser.parse_args()


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def metric(report: Optional[Dict[str, Any]], *keys: str) -> Any:
    cur: Any = report
    for key in keys:
        if cur is None or not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def add_connection(rows: List[Dict[str, Any]], name: str, path: Path, report: Optional[Dict[str, Any]]) -> None:
    rows.append(
        {
            "name": name,
            "report": str(path),
            "pair_f1_test": metric(report, "pair_metrics_test", "f1"),
            "step_topk_f1_test": metric(report, "step_topk_test", "f1"),
            "step_exact_test": metric(report, "step_topk_test", "exact_match"),
        }
    )


def add_grounding(rows: List[Dict[str, Any]], name: str, path: Path, report: Optional[Dict[str, Any]]) -> None:
    rows.append(
        {
            "name": name,
            "report": str(path),
            "primitive_only": metric(report, "primitive_only"),
            "pair_f1_test": metric(report, "pair_metrics_test", "f1"),
            "assignment_instance_accuracy_test": metric(report, "assignment_test", "instance_accuracy"),
            "assignment_exact_test": metric(report, "assignment_test", "exact_match"),
            "assignment_instance_accuracy_val": metric(report, "assignment_val", "instance_accuracy"),
            "assignment_equivalence_accuracy_test": metric(
                report, "assignment_equivalence_test", "instance_accuracy"
            ),
            "assignment_equivalence_accuracy_val": metric(report, "assignment_equivalence_val", "instance_accuracy"),
            "selection_metric": metric(report, "selection", "metric"),
        }
    )


def add_tree_planner(rows: List[Dict[str, Any]], name: str, path: Path, report: Optional[Dict[str, Any]]) -> None:
    rows.append(
        {
            "name": name,
            "report": str(path),
            "pair_f1_test": metric(report, "pair_metrics", "test", "f1"),
            "candidate_f1_test": metric(report, "candidate_metrics", "test", "f1"),
            "candidate_f1_at_decoder_test": metric(report, "candidate_metrics_at_decoder_threshold", "test", "f1"),
            "avg_selected_test": metric(report, "tree_metrics", "test", "avg_selected_candidates"),
            "avg_gold_candidate_recall_test": metric(report, "tree_metrics", "test", "avg_gold_candidate_recall"),
            "simple_f1_test": metric(report, "tree_metrics", "test", "metrics", "simple", "f1"),
            "hard_f1_test": metric(report, "tree_metrics", "test", "metrics", "hard", "f1"),
            "simple_f1_all": metric(report, "tree_metrics", "all", "metrics", "simple", "f1"),
            "hard_f1_all": metric(report, "tree_metrics", "all", "metrics", "hard", "f1"),
        }
    )


def markdown_table(rows: List[Dict[str, Any]], columns: List[str]) -> str:
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(fmt(row.get(col)) for col in columns) + " |")
    return "\n".join(out)


def main() -> None:
    args = parse_args()
    reports_dir = Path(args.reports_dir)
    grounding_summary = load_json(Path(args.grounding_summary))

    connection_specs = [
        ("sampled SVG + mask alignment", reports_dir / "pairwise_connection_report_mask.json"),
        ("simplified SVG + mask alignment", reports_dir / "simplified_connection_report.json"),
    ]
    # The sampled-SVG report historically lives under svg_features.
    if not connection_specs[0][1].exists():
        connection_specs[0] = ("sampled SVG + mask alignment", Path("svg_features/pairwise_connection_report_mask.json"))

    grounding_specs = [
        ("geometry primitive", reports_dir / "grounding_geometric_primitive_report.json"),
        ("geometry all", reports_dir / "grounding_geometric_all_report.json"),
        ("linear image primitive", reports_dir / "grounding_image_primitive_report.json"),
        ("linear image all", reports_dir / "grounding_image_all_report.json"),
        ("tiny CNN primitive e25", reports_dir / "grounding_cnn_primitive_e25_report.json"),
        ("tiny CNN all e25", reports_dir / "grounding_cnn_all_e25_report.json"),
        ("tiny CNN all val", reports_dir / "grounding_cnn_all_val_report.json"),
        ("tiny CNN all val aug", reports_dir / "grounding_cnn_all_val_aug_report.json"),
        ("tiny CNN all equiv-label val", reports_dir / "grounding_cnn_all_equiv_val_report.json"),
        ("residual CNN all improved val", reports_dir / "grounding_cnn_improved_val_report.json"),
        ("tiny CNN primitive val", reports_dir / "grounding_cnn_primitive_val_report.json"),
    ]
    tree_planner_specs = [
        ("geometry-only greedy planner", reports_dir / "tree_planner_geometry_report.json"),
        ("SVG-only greedy planner", reports_dir / "tree_planner_svg_report.json"),
        ("SVG+geometry greedy planner", reports_dir / "tree_planner_svg_geometry_report.json"),
        ("SVG+geometry+composite greedy planner", reports_dir / "tree_planner_svg_geometry_composite_report.json"),
        ("geometry-only neural planner", reports_dir / "tree_planner_nn_geometry_report.json"),
        ("SVG-only neural planner", reports_dir / "tree_planner_nn_svg_report.json"),
        ("SVG+geometry neural planner", reports_dir / "tree_planner_nn_svg_geometry_report.json"),
        ("SVG+composite neural planner", reports_dir / "tree_planner_nn_svg_composite_report.json"),
        ("SVG+geometry+composite neural planner", reports_dir / "tree_planner_nn_svg_geometry_composite_report.json"),
        ("manual composite-context oracle", reports_dir / "tree_planner_composite_context_report.json"),
        ("predicted subassembly candidates recall-heavy", reports_dir / "subassembly_candidate_svg_geometry_report.json"),
        ("predicted subassembly candidates conservative", reports_dir / "subassembly_candidate_svg_geometry_precision_report.json"),
        ("predicted subassembly candidates neg20", reports_dir / "subassembly_candidate_svg_geometry_neg20_report.json"),
    ]

    connection_rows: List[Dict[str, Any]] = []
    for name, path in connection_specs:
        add_connection(connection_rows, name, path, load_json(path))

    grounding_rows: List[Dict[str, Any]] = []
    for name, path in grounding_specs:
        if path.exists():
            add_grounding(grounding_rows, name, path, load_json(path))

    tree_planner_rows: List[Dict[str, Any]] = []
    for name, path in tree_planner_specs:
        if path.exists():
            add_tree_planner(tree_planner_rows, name, path, load_json(path))

    summary = {
        "grounding_dataset": grounding_summary,
        "tree_generation_dataset": load_json(Path("experiments/svg_assembly/datasets/tree_generation_summary.json")),
        "connection_models": connection_rows,
        "grounding_models": grounding_rows,
        "tree_planners": tree_planner_rows,
        "end_to_end_diagnostic": load_json(reports_dir / "end_to_end_diagnostic_report.json"),
        "end_to_end_diagnostic_improved": load_json(reports_dir / "end_to_end_diagnostic_improved_report.json"),
        "paper_tree_metrics": load_json(reports_dir / "paper_tree_metric_report.json"),
        "equivalence_test": load_json(reports_dir / "equivalence_test_report.json"),
        "equivalence_all": load_json(reports_dir / "equivalence_and_tree_export_report.json"),
        "equivalence_test_equiv_model": load_json(reports_dir / "equivalence_test_equiv_model_report.json"),
        "equivalence_test_improved": load_json(reports_dir / "equivalence_test_improved_report.json"),
        "equivalence_all_equiv_model": load_json(
            reports_dir / "equivalence_and_tree_export_equiv_model_report.json"
        ),
        "equivalence_all_improved": load_json(reports_dir / "equivalence_and_tree_export_improved_report.json"),
        "end_to_end_diagnostic_equiv_test": load_json(reports_dir / "end_to_end_diagnostic_equiv_test_report.json"),
        "end_to_end_diagnostic_equiv_all": load_json(reports_dir / "end_to_end_diagnostic_equiv_all_report.json"),
        "current_reading": {
            "connection": "Simplified SVG geometry preserves nearly all step-connection signal when instance-to-part grounding is known.",
            "grounding": "Part-to-SVG grounding remains the bottleneck; the improved residual CNN gives the strongest current test connection metrics, especially under equivalence-aware evaluation.",
            "tree_generation": "The object-level planner predicts assembly trees from primitive part tokens without reading per-step SVGs at inference; primitive SVG prototypes improve the hard tree metric over geometry-only features in the greedy baseline.",
            "neural_tree_generation": "The first neural merge scorer is implemented, but it still uses a connected-component merge decoder; current neural results should be treated as a baseline rather than the final set/graph decoder.",
            "composite_context": "Composite SVG prototypes from manual subassembly instances encode nearly the whole tree; they are useful as supervision or manual/RAG upper-bound context, not as pure observed-part inference input.",
            "predicted_subassemblies": "The first learned subassembly-candidate predictor does not use manual composite tokens at inference, but it currently has very low candidate precision and weak tree metrics.",
        },
    }

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# SVG Assembly Experiment Summary",
        "",
        "## Dataset",
        "",
    ]
    if grounding_summary:
        lines.extend(
            [
                f"- grounding samples: {grounding_summary.get('num_samples')}",
                f"- primitive samples: {grounding_summary.get('primitive_samples')}",
                f"- composite/subassembly samples: {grounding_summary.get('composite_samples')}",
                f"- train/test samples: {grounding_summary.get('splits', {}).get('train')}/{grounding_summary.get('splits', {}).get('test')}",
                "",
            ]
        )
    tree_summary = summary["tree_generation_dataset"]
    if tree_summary:
        lines.extend(
            [
                "## Tree Generation Dataset",
                "",
                f"- objects: {tree_summary.get('num_objects')}",
                f"- object split train/test: {tree_summary.get('splits', {}).get('train')}/{tree_summary.get('splits', {}).get('test')}",
                f"- primitive parts: {tree_summary.get('total_parts')}",
                f"- tree actions: {tree_summary.get('total_tree_actions')}",
                f"- parts with SVG prototype: {fmt(tree_summary.get('parts_with_svg_prototype_rate'))}",
                f"- composite SVG prototypes: {tree_summary.get('composite_svg_prototypes')}",
                f"- tree-action parent composite coverage: {fmt(tree_summary.get('tree_action_parent_composite_svg_coverage'))}",
                "",
            ]
        )
    lines.extend(
        [
            "## Connection Prediction",
            "",
            markdown_table(connection_rows, ["name", "pair_f1_test", "step_topk_f1_test", "step_exact_test"]),
            "",
            "## Grounding",
            "",
            markdown_table(
                grounding_rows,
                [
                    "name",
                    "pair_f1_test",
                    "assignment_instance_accuracy_test",
                    "assignment_equivalence_accuracy_test",
                    "assignment_exact_test",
                    "assignment_instance_accuracy_val",
                ],
            ),
            "",
            "## Current Reading",
            "",
            "- Manual-side SVG simplification is usable for connection prediction.",
            "- End-to-end performance is currently limited by grounding from observed part images/shapes to SVG instances.",
            "- The improved residual CNN is the strongest current grounding model on the test diagnostic.",
            "- Equivalence-aware labels remain important because repeated legs, slats, and connectors often have multiple valid instance assignments.",
            "- Object-level tree planning has started: the greedy SVG planner improves hard tree generation metrics over geometry-only features without using per-step SVG at inference.",
            "- The first neural merge scorer is now implemented, but it still uses a simple connected-component merge decoder. Its results are useful as a baseline, not as the final tree decoder.",
            "- Composite/subassembly SVG prototypes cover most tree-action parent sets. A context-only oracle recovers the full tree, which confirms that these manual middle-step tokens are tree supervision/RAG context rather than pure inference input.",
            "- The first no-leakage subassembly-candidate predictor is implemented. Negative sampling and conservative decoder selection reduce false positives and improve test Hard F1, but predicted candidates are still not strong enough to replace manual composite context.",
            "- The next useful work is better composite/subassembly representation, a stronger graph/tree decoder, and a real-image/3D observation grounding benchmark.",
            "",
        ]
    )
    if tree_planner_rows:
        lines.extend(
            [
                "## Object-Level Tree Planner",
                "",
                markdown_table(
                    tree_planner_rows,
                    [
                        "name",
                        "pair_f1_test",
                        "candidate_f1_test",
                        "candidate_f1_at_decoder_test",
                        "avg_selected_test",
                        "avg_gold_candidate_recall_test",
                        "simple_f1_test",
                        "hard_f1_test",
                        "simple_f1_all",
                        "hard_f1_all",
                    ],
                ),
                "",
                "Rows without `composite` use primitive part tokens and do not read per-step SVGs at inference. Rows with `composite` additionally use manual-derived subassembly SVG prototypes as optional manual/RAG context, so they should be reported separately from pure observed-part inference. The manual composite-context oracle shows that these tokens can directly encode the tree. The greedy rows use a logistic pair scorer; the neural rows use an MLP merge scorer but still decode with connected-component merges, so they are not yet a full learned tree decoder.",
                "",
            ]
        )
    end_to_end = summary["end_to_end_diagnostic"]
    end_to_end_equiv = summary["end_to_end_diagnostic_equiv_test"]
    end_to_end_improved = summary["end_to_end_diagnostic_improved"]
    if end_to_end:
        end_to_end_rows = [
            {
                "name": "base predicted grounding",
                "f1": metric(end_to_end, "connection_with_predicted_grounding", "f1"),
                "exact_match": metric(end_to_end, "connection_with_predicted_grounding", "exact_match"),
            }
        ]
        if end_to_end_equiv:
            end_to_end_rows.append(
                {
                    "name": "equiv-label predicted grounding",
                    "f1": metric(end_to_end_equiv, "connection_with_predicted_grounding", "f1"),
                    "exact_match": metric(end_to_end_equiv, "connection_with_predicted_grounding", "exact_match"),
                }
            )
        if end_to_end_improved:
            end_to_end_rows.append(
                {
                    "name": "improved predicted grounding",
                    "f1": metric(end_to_end_improved, "connection_with_predicted_grounding", "f1"),
                    "exact_match": metric(end_to_end_improved, "connection_with_predicted_grounding", "exact_match"),
                }
            )
        end_to_end_rows.append(
            {
                "name": "oracle grounding",
                "f1": metric(end_to_end, "connection_with_oracle_grounding", "f1"),
                "exact_match": metric(end_to_end, "connection_with_oracle_grounding", "exact_match"),
            }
        )
        lines.extend(
            [
                "## End-to-End Diagnostic",
                "",
                markdown_table(
                    end_to_end_rows,
                    ["name", "f1", "exact_match"],
                ),
                "",
            ]
        )
    paper_tree = summary["paper_tree_metrics"]
    if paper_tree and paper_tree.get("current_connection_induced_tree"):
        tree_metrics = paper_tree["current_connection_induced_tree"]["metrics"]
        lines.extend(
            [
                "## Paper Tree Metrics",
                "",
                markdown_table(
                    [
                        {
                            "name": "simple",
                            "precision": metric(tree_metrics, "simple", "precision"),
                            "recall": metric(tree_metrics, "simple", "recall"),
                            "f1": metric(tree_metrics, "simple", "f1"),
                        },
                        {
                            "name": "hard",
                            "precision": metric(tree_metrics, "hard", "precision"),
                            "recall": metric(tree_metrics, "hard", "recall"),
                            "f1": metric(tree_metrics, "hard", "f1"),
                        },
                    ],
                    ["name", "precision", "recall", "f1"],
                ),
                "",
                "These metrics are high because the diagnostic uses manual step SVGs; they should not be compared directly with the paper's shape-only assembly-plan generation task.",
                "",
            ]
        )
    equivalence_rows = []
    for name, report in [
        ("test/base-label", summary["equivalence_test"]),
        ("test/equiv-label", summary["equivalence_test_equiv_model"]),
        ("test/improved", summary["equivalence_test_improved"]),
        ("all/export/base-label", summary["equivalence_all"]),
        ("all/export/equiv-label", summary["equivalence_all_equiv_model"]),
        ("all/export/improved", summary["equivalence_all_improved"]),
    ]:
        if report:
            equivalence_rows.append(
                {
                    "name": name,
                    "ground_strict": metric(report, "grounding", "strict_instance_accuracy"),
                    "ground_equiv": metric(report, "grounding", "equivalence_instance_accuracy"),
                    "conn_strict_f1": metric(report, "connections", "strict", "f1"),
                    "conn_equiv_f1": metric(report, "connections", "equivalence_aware", "f1"),
                }
            )
    if equivalence_rows:
        lines.extend(
            [
                "## Equivalence-Aware Metrics",
                "",
                markdown_table(equivalence_rows, ["name", "ground_strict", "ground_equiv", "conn_strict_f1", "conn_equiv_f1"]),
                "",
                "The test row is the stricter generalization view; the all/export row is used to write complete per-object predicted assembly trees.",
                "",
            ]
        )
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
