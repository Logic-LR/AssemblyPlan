#!/usr/bin/env python3
"""Run key experiments with fixed --split-seed 0 and varied --seed for model init.

Usage:
  python scripts/run_split_seed_experiments.py
"""

import json
import subprocess
import sys
from pathlib import Path

REPORTS_DIR = Path("experiments/svg_assembly/reports")
MODELS_DIR = Path("experiments/svg_assembly/reports")

EXPERIMENTS = [
    # --- BCE 100% label baselines (5 seeds) ---
    *[
        {
            "name": f"bce100_seed{s}",
            "cmd": [
                "python", "scripts/train/train_tree_planner_context.py",
                "--feature-mode", "svg_geometry_composite",
                "--epochs", "120",
                "--hidden-dim", "192",
                "--entropy-weight", "0.0",
                "--label-ratio", "1.0",
                "--split-seed", "0",
                "--seed", str(s),
                "--output", str(REPORTS_DIR / f"splitseed_bce100_s{s}.json"),
                "--model-output", str(MODELS_DIR / f"splitseed_bce100_s{s}.pt"),
            ],
        }
        for s in [0, 1, 2, 3, 4]
    ],
    # --- BCE 10% label baselines (3 seeds) ---
    *[
        {
            "name": f"bce010_seed{s}",
            "cmd": [
                "python", "scripts/train/train_tree_planner_context.py",
                "--feature-mode", "svg_geometry_composite",
                "--epochs", "120",
                "--hidden-dim", "192",
                "--entropy-weight", "0.0",
                "--label-ratio", "0.10",
                "--split-seed", "0",
                "--seed", str(s),
                "--output", str(REPORTS_DIR / f"splitseed_bce010_s{s}.json"),
                "--model-output", str(MODELS_DIR / f"splitseed_bce010_s{s}.pt"),
            ],
        }
        for s in [0, 1, 2]
    ],
    # --- BCE 25% label baselines (3 seeds) ---
    *[
        {
            "name": f"bce025_seed{s}",
            "cmd": [
                "python", "scripts/train/train_tree_planner_context.py",
                "--feature-mode", "svg_geometry_composite",
                "--epochs", "120",
                "--hidden-dim", "192",
                "--entropy-weight", "0.0",
                "--label-ratio", "0.25",
                "--split-seed", "0",
                "--seed", str(s),
                "--output", str(REPORTS_DIR / f"splitseed_bce025_s{s}.json"),
                "--model-output", str(MODELS_DIR / f"splitseed_bce025_s{s}.pt"),
            ],
        }
        for s in [0, 1, 2]
    ],
    # --- BCE geometry-only 100% (3 seeds) ---
    *[
        {
            "name": f"bce_geo100_seed{s}",
            "cmd": [
                "python", "scripts/train/train_tree_planner_context.py",
                "--feature-mode", "geometry",
                "--epochs", "120",
                "--hidden-dim", "192",
                "--entropy-weight", "0.0",
                "--label-ratio", "1.0",
                "--split-seed", "0",
                "--seed", str(s),
                "--output", str(REPORTS_DIR / f"splitseed_bce_geo100_s{s}.json"),
                "--model-output", str(MODELS_DIR / f"splitseed_bce_geo100_s{s}.pt"),
            ],
        }
        for s in [0, 1, 2]
    ],
    # --- BCE geometry-only 10% (3 seeds) ---
    *[
        {
            "name": f"bce_geo010_seed{s}",
            "cmd": [
                "python", "scripts/train/train_tree_planner_context.py",
                "--feature-mode", "geometry",
                "--epochs", "120",
                "--hidden-dim", "192",
                "--entropy-weight", "0.0",
                "--label-ratio", "0.10",
                "--split-seed", "0",
                "--seed", str(s),
                "--output", str(REPORTS_DIR / f"splitseed_bce_geo010_s{s}.json"),
                "--model-output", str(MODELS_DIR / f"splitseed_bce_geo010_s{s}.pt"),
            ],
        }
        for s in [0, 1, 2]
    ],
]


def run_one(exp: dict) -> dict | None:
    print(f"\n{'='*60}")
    print(f"RUNNING: {exp['name']}")
    print(f"{'='*60}", flush=True)
    result = subprocess.run(exp["cmd"], capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"FAILED: {exp['name']}")
        print(result.stderr[-500:])
        return None
    # Parse the output JSON to extract key metrics
    try:
        # The last line of stdout should be the report JSON
        report = json.loads(result.stdout.strip().splitlines()[-1])
        tm = report.get("tree_metrics", report)
        return {
            "name": exp["name"],
            "test_simple": tm["test"]["metrics"]["simple"]["f1"],
            "test_hard": tm["test"]["metrics"]["hard"]["f1"],
            "all_simple": tm["all"]["metrics"]["simple"]["f1"],
            "all_hard": tm["all"]["metrics"]["hard"]["f1"],
        }
    except Exception as e:
        print(f"Parse error for {exp['name']}: {e}")
        # Try reading the output file
        out_path = None
        for arg in exp["cmd"]:
            if arg.endswith(".json") and "experiments" in arg:
                out_path = arg
                break
        if out_path and Path(out_path).exists():
            with open(out_path, encoding="utf-8") as f:
                report = json.load(f)
            tm = report.get("tree_metrics", report)
            return {
                "name": exp["name"],
                "test_simple": tm["test"]["metrics"]["simple"]["f1"],
                "test_hard": tm["test"]["metrics"]["hard"]["f1"],
                "all_simple": tm["all"]["metrics"]["simple"]["f1"],
                "all_hard": tm["all"]["metrics"]["hard"]["f1"],
            }
        return None


def print_summary(results: list):
    print("\n\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"{'Experiment':<30s} {'Test Simple':>11s} {'Test Hard':>9s} {'All Hard':>9s}")
    print("-"*60)
    for r in results:
        if r:
            print(f"{r['name']:<30s} {r['test_simple']:11.4f} {r['test_hard']:9.4f} {r['all_hard']:9.4f}")

    # Group by experiment type
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        if r:
            prefix = r["name"].rsplit("_s", 1)[0]  # remove _s0, _s1, etc.
            groups[prefix].append(r)

    print("\n\nGROUPED (mean ± std across seeds):")
    print(f"{'Experiment':<30s} {'Test Hard':>20s}")
    print("-"*55)
    for prefix, items in sorted(groups.items()):
        hard_vals = [it["test_hard"] for it in items]
        mean = sum(hard_vals) / len(hard_vals)
        if len(hard_vals) > 1:
            std = (sum((h - mean)**2 for h in hard_vals) / (len(hard_vals) - 1)) ** 0.5
            print(f"{prefix:<30s} {mean:.4f} ± {std:.4f}  (n={len(hard_vals)})")
        else:
            print(f"{prefix:<30s} {mean:.4f}  (n=1)")


def main():
    results = []
    for exp in EXPERIMENTS:
        r = run_one(exp)
        results.append(r)
        print_summary(results)

    # Save full results
    out = Path("experiments/svg_assembly/reports/splitseed_summary.json")
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
