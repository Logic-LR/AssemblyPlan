#!/usr/bin/env python3
"""Run GRPO experiments with fixed --split-seed 0 and varied --seed."""

import json
import subprocess
from pathlib import Path

REPORTS_DIR = Path("experiments/svg_assembly/reports")
MODELS_DIR = Path("experiments/svg_assembly/reports")

EXPERIMENTS = [
    # --- GRPO SVG-only composite (5 seeds) - HEADLINE ---
    *[
        {
            "name": f"grpo_svgonly_comp_s{s}",
            "cmd": [
                "python", "scripts/train/train_tree_grpo.py",
                "--from-scratch",
                "--feature-mode", "svg_geometry_composite",
                "--epochs", "80",
                "--hidden-dim", "128",
                "--samples-per-object", "16",
                "--temperature", "1.5",
                "--kl-beta", "0.1",
                "--reward-gt-f1", "0",
                "--reward-svg-coherence", "0.4",
                "--reward-spatial-svg", "0.6",
                "--gt-label-ratio", "0",
                "--split-seed", "0",
                "--seed", str(s),
                "--lr", "0.0005",
                "--output", str(REPORTS_DIR / f"splitseed_grpo_svgonly_s{s}.json"),
                "--model-output", str(MODELS_DIR / f"splitseed_grpo_svgonly_s{s}.pt"),
            ],
        }
        for s in [0, 1, 2, 3, 4]
    ],
    # --- GRPO 10% label composite (3 seeds) ---
    *[
        {
            "name": f"grpo_010_comp_s{s}",
            "cmd": [
                "python", "scripts/train/train_tree_grpo.py",
                "--warm-start", str(MODELS_DIR / f"splitseed_bce010_s{s}.pt"),
                "--feature-mode", "svg_geometry_composite",
                "--epochs", "80",
                "--hidden-dim", "192",
                "--samples-per-object", "16",
                "--temperature", "1.5",
                "--kl-beta", "0.1",
                "--reward-gt-f1", "0.5",
                "--reward-svg-coherence", "0.2",
                "--reward-spatial-svg", "0.3",
                "--gt-label-ratio", "0.10",
                "--gt-label-seed", str(s),
                "--split-seed", "0",
                "--seed", str(s),
                "--lr", "0.0001",
                "--output", str(REPORTS_DIR / f"splitseed_grpo_010_s{s}.json"),
                "--model-output", str(MODELS_DIR / f"splitseed_grpo_010_s{s}.pt"),
            ],
        }
        for s in [0, 1, 2]
    ],
]


def run_one(exp: dict) -> dict | None:
    print(f"\n{'='*60}")
    print(f"RUNNING: {exp['name']}")
    print(f"{'='*60}", flush=True)
    result = subprocess.run(exp["cmd"], capture_output=True, text=True, timeout=3600)
    if result.returncode != 0:
        print(f"FAILED: {exp['name']}")
        print(result.stderr[-500:])
        return None
    try:
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
    except Exception as e:
        print(f"Parse error: {e}")
    return None


def main():
    results = []
    for exp in EXPERIMENTS:
        r = run_one(exp)
        results.append(r)
        # Print progress
        for r in results:
            if r:
                print(f"  {r['name']}: Test Hard={r['test_hard']:.4f}")

    out = Path("experiments/svg_assembly/reports/splitseed_grpo_summary.json")
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
