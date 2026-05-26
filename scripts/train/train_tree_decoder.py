#!/usr/bin/env python3
"""Train a transformer set-context tree decoder for assembly-tree generation.

Unlike the flat pair scorer (train_tree_planner_nn.py), this model encodes all
current clusters jointly with a transformer before scoring candidate merges.
The decoder is still greedy connected-components, but the scores are now
context-aware: each cluster's representation depends on the full set of
co-existing clusters via self-attention.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from eval.evaluate_paper_tree_metrics import (
    average_metrics,
    build_tree_from_list,
    eval_tree,
    step_tree_from_child_specs,
)
from export.export_tree_predictions_and_equivalence_report import tree_to_list
from train.train_tree_planner_baseline import (
    Cluster,
    cluster_repr,
    cluster_token,
    composite_feature_map,
    connected_components,
    node_actions,
    part_feature_map,
    part_count,
    split_records,
    uses_composite_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="experiments/svg_assembly/datasets/tree_generation_dataset.json",
    )
    parser.add_argument(
        "--feature-mode",
        choices=[
            "geometry",
            "svg",
            "svg_geometry",
            "svg_composite",
            "svg_geometry_composite",
        ],
        default="svg_geometry",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        default="experiments/svg_assembly/reports/tree_decoder_svg_geometry_report.json",
    )
    parser.add_argument(
        "--model-output",
        default="experiments/svg_assembly/reports/tree_decoder_svg_geometry_model.pt",
    )
    parser.add_argument(
        "--pred-output-dir",
        default="experiments/svg_assembly/tree_decoder_predictions_svg_geometry_test",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class SetContextTreeDecoder(nn.Module):
    """Transformer encoder over the current set of clusters, plus a pair head."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pair_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def encode(self, cluster_features: torch.Tensor) -> torch.Tensor:
        """Encode a set of clusters into contextual embeddings.

        Args:
            cluster_features: [M, input_dim]

        Returns:
            [M, hidden_dim]
        """
        x = self.input_proj(cluster_features)  # [M, H]
        x = self.encoder(x.unsqueeze(0)).squeeze(0)  # [M, H]
        return x

    def score_pairs(
        self, ctx: torch.Tensor, pairs: List[Tuple[int, int]]
    ) -> torch.Tensor:
        """Score candidate merge pairs from contextual embeddings.

        Args:
            ctx: [M, hidden_dim]
            pairs: list of (i, j) index tuples

        Returns:
            [P] logit scores
        """
        feats = []
        for i, j in pairs:
            a, b = ctx[i], ctx[j]
            feats.append(torch.cat([a, b, a - b, a * b]))
        x = torch.stack(feats)  # [P, H*4]
        return self.pair_scorer(x).squeeze(-1)  # [P]


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def _clusters_from_parts(num_parts: int) -> Tuple[set[Cluster], Dict[int, Cluster]]:
    """Return singleton clusters and a part-id -> singleton map."""
    singletons = {frozenset([p]) for p in range(num_parts)}
    part_to_cluster = {p: frozenset([p]) for p in range(num_parts)}
    return singletons, part_to_cluster


def _pair_indices(num_clusters: int) -> List[Tuple[int, int]]:
    """All unordered pair indices for 0..num_clusters-1."""
    return [(i, j) for i in range(num_clusters) for j in range(i + 1, num_clusters)]


def train_one_epoch(
    model: SetContextTreeDecoder,
    records: Sequence[Dict[str, Any]],
    mode: str,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Teacher-forcing training over one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    total_pairs = 0

    for record in records:
        tree = build_tree_from_list(record["assembly_tree"])
        features = part_feature_map(record, mode)
        composites = composite_feature_map(record, mode) if uses_composite_features(mode) else None
        num_parts = int(record["num_parts"])

        current_clusters, _ = _clusters_from_parts(num_parts)

        for action in node_actions(tree):
            children = [frozenset(child) for child in action["children"]]
            parent = frozenset(action["parent"])

            # Build ordered cluster list for this step
            clusters_list = sorted(
                current_clusters, key=lambda c: (len(c), tuple(sorted(c)))
            )

            if len(clusters_list) < 2:
                break

            # Cluster features
            feats = torch.stack(
                [
                    torch.from_numpy(
                        cluster_repr(c, features, composites)
                    ).to(device)
                    for c in clusters_list
                ]
            )  # [M, D]

            # Contextual encoding
            ctx = model.encode(feats)  # [M, H]

            # All candidate pairs
            pairs = _pair_indices(len(clusters_list))
            scores = model.score_pairs(ctx, pairs)  # [P]

            # Ground-truth positive pairs
            gt_set = set(children)
            labels = torch.tensor(
                [
                    1.0
                    if clusters_list[i] in gt_set
                    and clusters_list[j] in gt_set
                    else 0.0
                    for i, j in pairs
                ],
                device=device,
            )

            # Weighted BCE
            pos_count = labels.sum().clamp_min(1)
            neg_count = len(labels) - pos_count
            pos_weight = neg_count / pos_count
            weights = torch.where(labels > 0.5, pos_weight, 1.0)

            loss = nn.functional.binary_cross_entropy_with_logits(
                scores, labels, weight=weights
            )
            loss.backward()

            total_loss += float(loss.item()) * len(labels)
            total_pairs += len(labels)

            # Advance to next step
            for child in children:
                current_clusters.discard(child)
            current_clusters.add(parent)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return total_loss / max(total_pairs, 1)


@torch.no_grad()
def predict_tree(
    record: Dict[str, Any],
    mode: str,
    model: SetContextTreeDecoder,
    threshold: float,
    device: torch.device,
) -> Any:
    """Greedy tree prediction with context-aware scores."""
    features = part_feature_map(record, mode)
    composites = composite_feature_map(record, mode) if uses_composite_features(mode) else None
    num_parts = int(record["num_parts"])

    current_clusters, _ = _clusters_from_parts(num_parts)
    child_specs: List[List[str]] = []
    max_steps = max(1, num_parts * 2)

    for _ in range(max_steps):
        if len(current_clusters) <= 1:
            break

        clusters_list = sorted(
            current_clusters, key=lambda c: (len(c), tuple(sorted(c)))
        )

        # Contextual scores
        feats = torch.stack(
            [
                torch.from_numpy(cluster_repr(c, features, composites)).to(device)
                for c in clusters_list
            ]
        )
        ctx = model.encode(feats)
        pairs = _pair_indices(len(clusters_list))
        logits = model.score_pairs(ctx, pairs)
        probs = torch.sigmoid(logits).cpu().numpy()

        scored = sorted(
            [(float(p), pairs[k][0], pairs[k][1]) for k, p in enumerate(probs)],
            key=lambda item: item[0],
            reverse=True,
        )

        best_prob, best_i, best_j = scored[0]
        edges = [
            (clusters_list[i], clusters_list[j])
            for prob, i, j in scored
            if prob >= threshold
        ]

        group = [
            clusters_list[best_i],
            clusters_list[best_j],
        ]
        if edges:
            comps = connected_components(clusters_list, edges)
            best_set = clusters_list[best_i] | clusters_list[best_j]
            for comp in comps:
                comp_union = frozenset().union(*comp)
                if best_set <= comp_union:
                    group = comp
                    break

        # Build merge spec for this step
        child_specs.append(
            [
                cluster_token(c)
                for c in sorted(
                    group, key=lambda item: (len(item), tuple(sorted(item)))
                )
            ]
        )

        parent = frozenset().union(*group)
        for c in group:
            current_clusters.discard(c)
        current_clusters.add(parent)

    return step_tree_from_child_specs(child_specs, num_parts)


@torch.no_grad()
def evaluate_records(
    records: Sequence[Dict[str, Any]],
    mode: str,
    model: SetContextTreeDecoder,
    threshold: float,
    device: torch.device,
) -> Dict[str, Any]:
    rows = []
    examples = []
    for record in records:
        gt = build_tree_from_list(record["assembly_tree"])
        pred = predict_tree(record, mode, model, threshold, device)
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
    return {
        "objects": len(records),
        "metrics": average_metrics(rows),
        "examples": examples,
    }


@torch.no_grad()
def tune_threshold(
    records: Sequence[Dict[str, Any]],
    mode: str,
    model: SetContextTreeDecoder,
    device: torch.device,
) -> Tuple[float, Dict[str, Any]]:
    best_threshold = 0.5
    best_eval: Dict[str, Any] | None = None
    best_score = -1.0
    for threshold in np.linspace(0.15, 0.9, 16):
        result = evaluate_records(records, mode, model, float(threshold), device)
        score = result["metrics"]["hard"]["f1"]
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_eval = result
    assert best_eval is not None
    return best_threshold, best_eval


@torch.no_grad()
def export_predictions(
    records: Sequence[Dict[str, Any]],
    mode: str,
    model: SetContextTreeDecoder,
    threshold: float,
    device: torch.device,
    output_dir: Path,
) -> List[Dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    exported = []
    for record in records:
        gt = build_tree_from_list(record["assembly_tree"])
        pred = predict_tree(record, mode, model, threshold, device)
        metrics = eval_tree(gt, pred)
        obj_dir = output_dir / record["category"] / record["name"]
        obj_dir.mkdir(parents=True, exist_ok=True)
        path = obj_dir / "assembly_tree_prediction.json"
        path.write_text(
            json.dumps(
                {
                    "category": record["category"],
                    "name": record["name"],
                    "split": record.get("split"),
                    "feature_mode": mode,
                    "threshold": threshold,
                    "predicted_assembly_tree": tree_to_list(pred),
                    "ground_truth_assembly_tree": record["assembly_tree"],
                    "paper_tree_metrics": metrics,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        exported.append(
            {
                "category": record["category"],
                "name": record["name"],
                "split": record.get("split"),
                "path": str(path.as_posix()),
                "paper_tree_metrics": metrics,
            }
        )
    (output_dir / "index.json").write_text(
        json.dumps(exported, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return exported


def compute_input_dim(mode: str) -> int:
    """Compute the cluster_repr dimension for a feature mode.

    Dimensions are determined from the actual data to avoid hardcoding.
    """
    geom_dim = 14  # bbox_min(3) + bbox_max(3) + extent(3) + center(3) + num_vertices + num_faces
    svg_dim = 17
    shape_dim = 4
    svg_count_dim = 1
    agg_dim = 2  # len, log1p(len)

    if mode == "geometry":
        base = geom_dim
    elif mode in ("svg", "svg_composite"):
        base = svg_dim + shape_dim + svg_count_dim
    else:
        base = geom_dim + svg_dim + shape_dim + svg_count_dim

    total = base * 3 + agg_dim  # mean + max + min + agg
    if uses_composite_features(mode):
        total += 23  # composite feature dim
    return total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    fit_records, val_records, test_records = split_records(
        records, args.val_fraction, args.seed
    )

    input_dim = compute_input_dim(args.feature_mode)
    model = SetContextTreeDecoder(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_val_score = -1.0
    best_state: Dict[str, torch.Tensor] | None = None
    history: List[Dict[str, Any]] = []
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, fit_records, args.feature_mode, optimizer, device
        )

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            _, val_eval = tune_threshold(
                val_records or fit_records,
                args.feature_mode,
                model,
                device,
            )
            val_score = val_eval["metrics"]["hard"]["f1"]
            if val_score > best_val_score:
                best_val_score = val_score
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
                best_epoch = epoch

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_hard_f1": val_score,
                    "best_val_hard_f1": best_val_score,
                }
            )
            print(
                f"epoch {epoch:3d}  loss={train_loss:.4f}  "
                f"val_hard_f1={val_score:.4f}  best={best_val_score:.4f}",
                flush=True,
            )

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # Final threshold tuning on best model
    threshold, val_tree_eval = tune_threshold(
        val_records or fit_records, args.feature_mode, model, device
    )

    # Evaluate
    test_eval = evaluate_records(
        test_records, args.feature_mode, model, threshold, device
    )
    all_eval = evaluate_records(
        records, args.feature_mode, model, threshold, device
    )
    train_eval = evaluate_records(
        fit_records, args.feature_mode, model, threshold, device
    )

    # Export predictions
    exported = export_predictions(
        test_records, args.feature_mode, model, threshold, device,
        Path(args.pred_output_dir),
    )

    report = {
        "model": "set_context_tree_decoder",
        "feature_mode": args.feature_mode,
        "architecture": {
            "input_dim": input_dim,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "dropout": args.dropout,
        },
        "dataset": args.dataset,
        "splits": {
            "fit_objects": len(fit_records),
            "val_objects": len(val_records),
            "test_objects": len(test_records),
            "all_objects": len(records),
        },
        "training": {
            "epochs": args.epochs,
            "best_epoch": best_epoch,
            "best_val_hard_f1": best_val_score,
            "history": history,
            "device": str(device),
        },
        "threshold": {
            "selected": threshold,
            "selection_metric": "val hard F1",
            "val_tree_eval": val_tree_eval,
        },
        "tree_metrics": {
            "train": train_eval,
            "val": val_tree_eval,
            "test": test_eval,
            "all": all_eval,
        },
        "exported_test_predictions": {
            "output_dir": args.pred_output_dir,
            "num_objects": len(exported),
        },
        "notes": [
            "Transformer set-context decoder: encodes all current clusters jointly before scoring merge pairs.",
            "Training uses teacher forcing over the GT merge sequence.",
            "Inference is greedy connected-components with context-aware scores.",
            "Compare against the flat MLP scorer (train_tree_planner_nn.py) which uses independent pair scores.",
        ],
        "config": vars(args),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    torch.save(
        {
            "model_state": model.cpu().state_dict(),
            "feature_mode": args.feature_mode,
            "input_dim": input_dim,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "dropout": args.dropout,
            "threshold": threshold,
            "config": vars(args),
        },
        args.model_output,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote {out}")
    print(f"Wrote {args.model_output}")
    print(f"Wrote {Path(args.pred_output_dir) / 'index.json'}")


if __name__ == "__main__":
    main()
