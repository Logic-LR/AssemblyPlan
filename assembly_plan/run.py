#!/usr/bin/env python3
"""CLI entry point for assembly plan generation.

Usage (from dataset/ directory, using assembly_plan venv):
    assembly_plan/.venv/Scripts/python.exe -m assembly_plan.run --mode train
    assembly_plan/.venv/Scripts/python.exe -m assembly_plan.run --mode train_grpo --ckpt experiments/svg_assembly/reports/gnn_supervised_best.pt
    assembly_plan/.venv/Scripts/python.exe -m assembly_plan.run --mode eval --ckpt experiments/svg_assembly/reports/gnn_best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from .data import load_dataset, split_dataset, pair_feature_dim
from .model import build_model
from .train import train_supervised, train_grpo, _eval_tree_metrics
from .decoder import greedy_decode, group_greedy_decode, beam_search_decode
from .evaluate import build_tree_from_list, eval_tree, average_metrics, format_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["train", "train_grpo", "eval"],
                   default="train", help="Run mode")
    p.add_argument("--data-json", type=Path,
                   default=Path("experiments/svg_assembly/datasets/tree_generation_dataset.json"))
    p.add_argument("--svg-dir", type=Path,
                   default=Path("experiments/svg_assembly/simplified_svg"))
    p.add_argument("--ckpt", type=Path, default=None,
                   help="Checkpoint path for loading/saving")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gnn-hidden", type=int, default=128)
    p.add_argument("--gnn-layers", type=int, default=2)
    p.add_argument("--scorer-hidden", type=int, default=192)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--no-context", action="store_true",
                   help="Disable global context features")
    p.add_argument("--beam-width", type=int, default=5)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--group-decode", action="store_true",
                   help="Use group-aware decoder (detects k-ary merges)")
    p.add_argument("--threshold", type=float, default=-1.5,
                   help="Clique threshold for group-aware decoder")
    p.add_argument("--device", default="auto")
    # GRPO-specific
    p.add_argument("--grpo-epochs", type=int, default=50)
    p.add_argument("--grpo-lr", type=float, default=5e-5)
    p.add_argument("--grpo-samples", type=int, default=8)
    p.add_argument("--grpo-temperature", type=float, default=1.2)
    p.add_argument("--grpo-kl", type=float, default=0.1)
    return p.parse_args()


def get_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    include_context = not args.no_context

    print(f"Device: {device}")
    print(f"Loading data from {args.data_json} ...")
    objects = load_dataset(args.data_json, args.svg_dir)
    train_records, test_records = split_dataset(objects)
    print(f"  {len(train_records)} train, {len(test_records)} test")

    part_feat_dim = len(train_records[0].part_features[0].feature_vec)
    print(f"  Part feature dim: {part_feat_dim}")
    print(f"  Pair feature dim: {pair_feature_dim(include_context)}")

    model = build_model(
        part_feat_dim=part_feat_dim,
        gnn_hidden=args.gnn_hidden,
        gnn_layers=args.gnn_layers,
        scorer_hidden=args.scorer_hidden,
        include_context=include_context,
        dropout=args.dropout,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {total_params:,}")

    # Default checkpoint paths
    report_dir = Path("experiments/svg_assembly/reports")

    if args.mode == "train":
        ckpt_path = args.ckpt or (report_dir / "gnn_supervised_best.pt")
        print(f"\n=== Supervised Training ({args.epochs} epochs) ===")
        history = train_supervised(
            train_records=train_records,
            val_records=test_records,
            model=model,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            eval_every=args.eval_every,
            beam_width=args.beam_width,
            checkpoint_path=ckpt_path,
            include_context=include_context,
        )
        print(f"\nBest val Hard F1: {history['best_val_hard_f1']:.3f}")
        print(f"Checkpoint saved to: {ckpt_path}")

        # Final evaluation
        if args.group_decode:
            print(f"\n=== Final Evaluation (group-aware decoder) ===")
        else:
            print(f"\n=== Final Evaluation (beam_width={args.beam_width}) ===")
        if ckpt_path.exists():
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        tree_metrics = _eval_tree_metrics(model, test_records, device, args.beam_width,
                                          use_group_decode=args.group_decode,
                                          threshold=args.threshold)
        print(format_metrics({"overall": tree_metrics, "count": len(test_records)}))

    elif args.mode == "train_grpo":
        ckpt_path = args.ckpt or (report_dir / "gnn_supervised_best.pt")
        grpo_ckpt = args.ckpt.with_name(args.ckpt.stem + "_grpo.pt") if args.ckpt else (report_dir / "gnn_grpo_best.pt")

        if ckpt_path.exists():
            print(f"Loading supervised checkpoint: {ckpt_path}")
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        else:
            print(f"WARNING: No checkpoint found at {ckpt_path}, training from scratch")

        print(f"\n=== GRPO Fine-tuning ({args.grpo_epochs} epochs) ===")
        history = train_grpo(
            train_records=train_records,
            val_records=test_records,
            model=model,
            device=device,
            epochs=args.grpo_epochs,
            lr=args.grpo_lr,
            samples_per_object=args.grpo_samples,
            temperature=args.grpo_temperature,
            kl_coeff=args.grpo_kl,
            eval_every=args.eval_every,
            beam_width=args.beam_width,
            checkpoint_path=grpo_ckpt,
        )
        print(f"\nBest val Hard F1: {history['best_val_hard_f1']:.3f}")

    elif args.mode == "eval":
        ckpt_path = args.ckpt
        if not ckpt_path or not ckpt_path.exists():
            print("ERROR: --ckpt required and must exist for eval mode")
            sys.exit(1)

        print(f"Loading checkpoint: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))

        print(f"\n=== Evaluation ===")
        # Greedy
        print("\n--- Greedy Decode ---")
        greedy_metrics = _eval_tree_metrics(model, test_records, device, beam_width=1)
        print(format_metrics({"overall": greedy_metrics, "count": len(test_records)}))

        # Group-aware greedy
        if args.group_decode:
            print("\n--- Group-Aware Greedy Decode ---")
            group_metrics = _eval_tree_metrics(model, test_records, device, beam_width=1,
                                               use_group_decode=True,
                                               threshold=args.threshold)
            print(format_metrics({"overall": group_metrics, "count": len(test_records)}))

        # Beam search
        if args.beam_width > 1:
            print(f"\n--- Beam Search (width={args.beam_width}) ---")
            beam_metrics = _eval_tree_metrics(model, test_records, device, args.beam_width)
            print(format_metrics({"overall": beam_metrics, "count": len(test_records)}))


if __name__ == "__main__":
    main()
