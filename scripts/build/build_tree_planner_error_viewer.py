#!/usr/bin/env python3
"""Build an HTML viewer for object-level tree-planner errors."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prediction-dir",
        default="experiments/svg_assembly/tree_planner_predictions_nn_svg_geometry_test",
        help="Directory containing per-object assembly_tree_prediction.json files.",
    )
    parser.add_argument(
        "--output-html",
        default="experiments/svg_assembly/reports/tree_planner_error_viewer_nn_svg_geometry.html",
    )
    parser.add_argument(
        "--output-json",
        default="experiments/svg_assembly/reports/tree_planner_error_viewer_nn_svg_geometry.json",
    )
    parser.add_argument("--max-objects", type=int, default=60)
    return parser.parse_args()


def load_predictions(root: Path) -> List[Dict[str, Any]]:
    rows = []
    for path in root.rglob("assembly_tree_prediction.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload.get("paper_tree_metrics") or {}
        simple = metrics.get("simple") or {}
        hard = metrics.get("hard") or {}
        rows.append(
            {
                "category": payload.get("category"),
                "name": payload.get("name"),
                "split": payload.get("split"),
                "path": str(path.as_posix()),
                "simple_f1": float(simple.get("f1") or 0.0),
                "hard_f1": float(hard.get("f1") or 0.0),
                "simple": simple,
                "hard": hard,
                "predicted_assembly_tree": payload.get("predicted_assembly_tree"),
                "ground_truth_assembly_tree": payload.get("ground_truth_assembly_tree"),
                "part_tokens": payload.get("part_tokens") or [],
                "composite_tokens": payload.get("composite_tokens") or [],
                "predicted_subassembly_sets": payload.get("predicted_subassembly_sets") or [],
            }
        )
    return sorted(rows, key=lambda row: (row["hard_f1"], row["simple_f1"], row["category"] or "", row["name"] or ""))


def fmt_tree(value: Any) -> str:
    return html.escape(json.dumps(value, ensure_ascii=False, indent=2))


def summarize_part(token: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "part_id": token.get("part_id"),
        "extent": token.get("extent"),
        "svg_feature_count": token.get("svg_feature_count"),
        "shape_histogram": token.get("shape_histogram"),
        "svg_examples": token.get("svg_examples"),
    }


def fmt_parts(tokens: List[Dict[str, Any]]) -> str:
    rows = []
    for token in tokens:
        part = summarize_part(token)
        examples = part.get("svg_examples") or []
        example_text = ", ".join(
            f"step {item.get('step_id')}: {item.get('source_svg')}" for item in examples[:2]
        )
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(part.get('part_id')))}</td>"
            f"<td>{html.escape(json.dumps(part.get('extent'), ensure_ascii=False))}</td>"
            f"<td>{html.escape(str(part.get('svg_feature_count')))}</td>"
            f"<td>{html.escape(json.dumps(part.get('shape_histogram'), ensure_ascii=False))}</td>"
            f"<td>{html.escape(example_text)}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def fmt_composites(tokens: List[Dict[str, Any]]) -> str:
    rows = []
    for token in tokens:
        examples = token.get("svg_examples") or []
        example_text = ", ".join(
            f"step {item.get('step_id')}: {item.get('source_svg')}" for item in examples[:2]
        )
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(token.get('part_token') or ','.join(token.get('part_ids') or [])))}</td>"
            f"<td>{html.escape(str(token.get('svg_feature_count')))}</td>"
            f"<td>{html.escape(json.dumps(token.get('shape_histogram'), ensure_ascii=False))}</td>"
            f"<td>{html.escape(example_text)}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def fmt_predicted_sets(tokens: List[Dict[str, Any]]) -> str:
    rows = []
    for token in tokens:
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(token.get('token') or ','.join(token.get('parts') or [])))}</td>"
            f"<td>{html.escape(json.dumps(token.get('parts'), ensure_ascii=False))}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def build_html(rows: List[Dict[str, Any]], max_objects: int) -> str:
    cards = []
    for row in rows[:max_objects]:
        parts = fmt_parts(row["part_tokens"])
        composites = fmt_composites(row["composite_tokens"])
        predicted_sets = fmt_predicted_sets(row["predicted_subassembly_sets"])
        cards.append(
            f"""
<section class="case">
  <header>
    <h2>{html.escape(str(row["category"]))}/{html.escape(str(row["name"]))}</h2>
    <p>split={html.escape(str(row["split"]))} | simple F1={row["simple_f1"]:.3f} | hard F1={row["hard_f1"]:.3f}</p>
    <p class="path">{html.escape(row["path"])}</p>
  </header>
  <div class="trees">
    <div>
      <h3>Predicted Tree</h3>
      <pre>{fmt_tree(row["predicted_assembly_tree"])}</pre>
    </div>
    <div>
      <h3>Ground Truth Tree</h3>
      <pre>{fmt_tree(row["ground_truth_assembly_tree"])}</pre>
    </div>
  </div>
  <table>
    <caption>Primitive part tokens</caption>
    <thead>
      <tr><th>part_id</th><th>extent</th><th>svg_count</th><th>shape_histogram</th><th>svg_examples</th></tr>
    </thead>
    <tbody>
      {parts}
    </tbody>
  </table>
  <table>
    <caption>Composite SVG prototypes</caption>
    <thead>
      <tr><th>part_set</th><th>svg_count</th><th>shape_histogram</th><th>svg_examples</th></tr>
    </thead>
    <tbody>
      {composites}
    </tbody>
  </table>
  <table>
    <caption>Predicted subassembly sets</caption>
    <thead>
      <tr><th>token</th><th>parts</th></tr>
    </thead>
    <tbody>
      {predicted_sets}
    </tbody>
  </table>
</section>
"""
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Tree Planner Error Viewer</title>
  <style>
    body {{
      margin: 24px;
      font-family: Arial, sans-serif;
      color: #202124;
      background: #f6f7f9;
    }}
    h1 {{ margin-bottom: 4px; }}
    .summary {{ margin-bottom: 24px; color: #555; }}
    .case {{
      background: #fff;
      border: 1px solid #d8dde6;
      border-radius: 8px;
      padding: 18px;
      margin-bottom: 18px;
    }}
    .case h2 {{ margin: 0 0 4px; font-size: 20px; }}
    .case h3 {{ margin: 0 0 8px; font-size: 14px; }}
    .path {{ color: #666; font-size: 12px; }}
    .trees {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
      margin: 14px 0;
    }}
    pre {{
      overflow: auto;
      max-height: 420px;
      padding: 12px;
      background: #f0f2f5;
      border-radius: 6px;
      font-size: 12px;
      line-height: 1.45;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }}
    th, td {{
      text-align: left;
      vertical-align: top;
      border-top: 1px solid #e2e6ee;
      padding: 8px;
    }}
    caption {{
      margin-top: 10px;
      padding: 8px;
      text-align: left;
      font-weight: 700;
      color: #344054;
    }}
    th {{ color: #4a5568; background: #f8fafc; }}
    @media (max-width: 900px) {{
      .trees {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <h1>Tree Planner Error Viewer</h1>
  <p class="summary">Showing {min(len(rows), max_objects)} of {len(rows)} objects, sorted by worst Hard F1 first.</p>
  {''.join(cards)}
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    rows = load_predictions(Path(args.prediction_dir))
    output_json = Path(args.output_json)
    output_html = Path(args.output_html)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    output_html.write_text(build_html(rows, args.max_objects), encoding="utf-8")
    print(f"Loaded {len(rows)} predictions")
    print(f"Wrote {output_json}")
    print(f"Wrote {output_html}")


if __name__ == "__main__":
    main()
