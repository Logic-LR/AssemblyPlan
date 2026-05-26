#!/usr/bin/env python3
"""Train an MLP merge scorer augmented with global set-context features.

Unlike the flat pair scorer (train_tree_planner_nn.py), this model adds
per-step global context features — cluster count, average cluster sizes,
and pairwise statistics — so the scorer is aware of the current merge stage
without requiring a transformer.
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
from torch.utils.data import DataLoader, TensorDataset

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
    training_examples_for_record,
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
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--entropy-weight", type=float, default=0.0, help="Binary entropy bonus weight (0=off)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        default="experiments/svg_assembly/reports/context_planner_svg_geometry_report.json",
    )
    parser.add_argument(
        "--model-output",
        default="experiments/svg_assembly/reports/context_planner_svg_geometry_model.pt",
    )
    parser.add_argument(
        "--pred-output-dir",
        default="experiments/svg_assembly/context_planner_predictions_svg_geometry_test",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Global context features
# ---------------------------------------------------------------------------


def _global_context_features(
    clusters: Sequence[Cluster],
    features: Dict[int, np.ndarray],
    composites: Dict[Cluster, np.ndarray] | None,
) -> np.ndarray:
    """Compute global set-level features for the current step.

    Returns a fixed-size feature vector describing the whole set of clusters.
    """
    M = len(clusters)
    sizes = [len(c) for c in clusters]
    total_parts = sum(sizes)

    # Cluster count and size stats
    ctx = [
        float(M),
        math.log1p(M),
        float(total_parts),
        float(np.mean(sizes)) if sizes else 0.0,
        float(np.std(sizes)) if sizes else 0.0,
        float(max(sizes)) if sizes else 0.0,
        float(min(sizes)) if sizes else 0.0,
    ]

    # Average cluster_repr across all clusters
    reprs = []
    for c in clusters:
        r = cluster_repr(c, features, composites)
        reprs.append(r)
    if reprs:
        arr = np.vstack(reprs)
        mean_repr = arr.mean(axis=0)
        std_repr = arr.std(axis=0)
        ctx.extend(mean_repr.tolist())
        ctx.extend(std_repr.tolist())
    else:
        # Dummy padding — shouldn't happen
        dummy_dim = len(cluster_repr(clusters[0], features, composites))
        ctx.extend([0.0] * dummy_dim * 2)

    return np.asarray(ctx, dtype=np.float32)


def pair_feature_context(
    a: Cluster,
    b: Cluster,
    features: Dict[int, np.ndarray],
    composites: Dict[Cluster, np.ndarray] | None,
    global_ctx: np.ndarray,
) -> np.ndarray:
    """Pair feature augmented with global context."""
    if min(a) > min(b):
        a, b = b, a
    ra = cluster_repr(a, features, composites)
    rb = cluster_repr(b, features, composites)
    union = cluster_repr(a | b, features, composites)

    # Local pair features (same as original)
    local = np.concatenate(
        [ra, rb, np.abs(ra - rb), ra * rb, union]
    ).astype(np.float32)

    return np.concatenate([local, global_ctx]).astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset construction with context features
# ---------------------------------------------------------------------------


def training_examples_with_context(
    record: Dict[str, Any], mode: str
) -> Tuple[List[np.ndarray], List[float]]:
    tree = build_tree_from_list(record["assembly_tree"])
    features = part_feature_map(record, mode)
    composites = composite_feature_map(record, mode) if uses_composite_features(mode) else None
    current: set[Cluster] = {frozenset([part]) for part in range(part_count(tree))}
    xs: List[np.ndarray] = []
    ys: List[float] = []
    for action in node_actions(tree):
        children = [frozenset(child) for child in action["children"]]
        positives = {
            (a, b) if tuple(sorted(a)) <= tuple(sorted(b)) else (b, a)
            for i, a in enumerate(children)
            for b in children[i + 1:]
        }
        clusters = sorted(current, key=lambda item: (len(item), tuple(sorted(item))))
        global_ctx = _global_context_features(clusters, features, composites)
        for i, a in enumerate(clusters):
            for b in clusters[i + 1:]:
                key = (a, b) if tuple(sorted(a)) <= tuple(sorted(b)) else (b, a)
                xs.append(pair_feature_context(a, b, features, composites, global_ctx))
                ys.append(1.0 if key in positives else 0.0)
        for child in children:
            current.discard(child)
        current.add(frozenset(action["parent"]))
    return xs, ys


def build_pair_dataset(
    records: Sequence[Dict[str, Any]], mode: str
) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[np.ndarray] = []
    ys: List[float] = []
    for record in records:
        rec_x, rec_y = training_examples_with_context(record, mode)
        xs.extend(rec_x)
        ys.extend(rec_y)
    return np.vstack(xs).astype(np.float32), np.asarray(ys, dtype=np.float32)


def standardize(
    train_x: np.ndarray, *arrays: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean, std, [((arr - mean) / std).astype(np.float32) for arr in arrays]


def pair_metrics(probs: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    pred = probs >= 0.5
    gold = labels > 0.5
    tp = int(np.logical_and(pred, gold).sum())
    fp = int(np.logical_and(pred, ~gold).sum())
    fn = int(np.logical_and(~pred, gold).sum())
    tn = int(np.logical_and(~pred, ~gold).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall) if precision + recall else 0.0
    )
    return {
        "accuracy": (tp + tn) / len(labels),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class ContextMergeMLP(nn.Module):
    """MLP merge scorer (same as MergeMLP but handles variable input dim)."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@torch.no_grad()
def predict_probs(
    model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int = 2048
) -> np.ndarray:
    model.eval()
    out = []
    for start in range(0, len(x), batch_size):
        batch = torch.from_numpy(x[start : start + batch_size]).to(device)
        out.append(torch.sigmoid(model(batch)).cpu().numpy())
    return np.concatenate(out) if out else np.asarray([], dtype=np.float32)


def train_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[ContextMergeMLP, Dict[str, Any]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ContextMergeMLP(train_x.shape[1], args.hidden_dim, args.dropout).to(device)
    ds = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    pos_weight = torch.tensor(
        [(len(train_y) - float(train_y.sum())) / max(float(train_y.sum()), 1.0)],
        device=device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    use_entropy = getattr(args, "entropy_weight", 0.0) > 0
    best_state = None
    best_val = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            bce_loss = criterion(logits, yb)
            if use_entropy:
                probs = torch.sigmoid(logits)
                eps = 1e-8
                entropy = -(probs * torch.log(probs + eps) + (1 - probs) * torch.log(1 - probs + eps)).mean()
                loss = bce_loss - args.entropy_weight * entropy
            else:
                loss = bce_loss
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(yb)
            total += len(yb)
        if epoch == args.epochs or epoch % 10 == 0:
            val_probs = predict_probs(model, val_x, device)
            val_metrics = pair_metrics(val_probs, val_y)
            if val_metrics["f1"] > best_val:
                best_val = val_metrics["f1"]
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
            history.append(
                {
                    "epoch": epoch,
                    "loss": total_loss / max(total, 1),
                    "val_pair_f1": val_metrics["f1"],
                    "best_val_pair_f1": best_val,
                }
            )
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, {
        "device": str(device),
        "history": history,
        "best_val_pair_f1": best_val,
    }


# ---------------------------------------------------------------------------
# Inference (greedy connected-components, same as baseline)
# ---------------------------------------------------------------------------


@torch.no_grad()
def score_cluster_pairs(
    record: Dict[str, Any],
    clusters: Sequence[Cluster],
    mode: str,
    model: nn.Module,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
) -> List[Tuple[float, Cluster, Cluster]]:
    features = part_feature_map(record, mode)
    composites = composite_feature_map(record, mode) if uses_composite_features(mode) else None
    global_ctx = _global_context_features(clusters, features, composites)
    raw = []
    pairs: List[Tuple[Cluster, Cluster]] = []
    for i, a in enumerate(clusters):
        for b in clusters[i + 1:]:
            raw.append(pair_feature_context(a, b, features, composites, global_ctx))
            pairs.append((a, b))
    if not raw:
        return []
    x = ((np.vstack(raw).astype(np.float32) - mean) / std).astype(np.float32)
    probs = predict_probs(model, x, device)
    return sorted(
        [(float(prob), a, b) for prob, (a, b) in zip(probs, pairs)],
        key=lambda item: item[0],
        reverse=True,
    )


@torch.no_grad()
def plan_tree(
    record: Dict[str, Any],
    mode: str,
    model: nn.Module,
    mean: np.ndarray,
    std: np.ndarray,
    threshold: float,
    device: torch.device,
) -> Any:
    current: set[Cluster] = {
        frozenset([part]) for part in range(int(record["num_parts"]))
    }
    child_specs: List[List[str]] = []
    max_steps = max(1, int(record["num_parts"]) * 2)
    for _ in range(max_steps):
        if len(current) <= 1:
            break
        clusters = sorted(current, key=lambda item: (len(item), tuple(sorted(item))))
        scored = score_cluster_pairs(record, clusters, mode, model, mean, std, device)
        if not scored:
            break
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
        parent = frozenset().union(*group)
        child_specs.append(
            [
                cluster_token(cluster)
                for cluster in sorted(
                    group, key=lambda item: (len(item), tuple(sorted(item)))
                )
            ]
        )
        for cluster in group:
            current.discard(cluster)
        current.add(parent)
    return step_tree_from_child_specs(child_specs, int(record["num_parts"]))


@torch.no_grad()
def evaluate_records(
    records: Sequence[Dict[str, Any]],
    mode: str,
    model: nn.Module,
    mean: np.ndarray,
    std: np.ndarray,
    threshold: float,
    device: torch.device,
) -> Dict[str, Any]:
    rows = []
    examples = []
    for record in records:
        gt = build_tree_from_list(record["assembly_tree"])
        pred = plan_tree(record, mode, model, mean, std, threshold, device)
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
    model: nn.Module,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
) -> Tuple[float, Dict[str, Any]]:
    best_threshold = 0.5
    best_eval = None
    best_score = -1.0
    for threshold in np.linspace(0.15, 0.9, 16):
        result = evaluate_records(
            records, mode, model, mean, std, float(threshold), device
        )
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
    model: nn.Module,
    mean: np.ndarray,
    std: np.ndarray,
    threshold: float,
    device: torch.device,
    output_dir: Path,
) -> List[Dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    exported = []
    for record in records:
        gt = build_tree_from_list(record["assembly_tree"])
        pred = plan_tree(record, mode, model, mean, std, threshold, device)
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    records = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    fit_records, val_records, test_records = split_records(
        records, args.val_fraction, args.seed
    )
    train_x, train_y = build_pair_dataset(fit_records, args.feature_mode)
    val_x, val_y = (
        build_pair_dataset(val_records, args.feature_mode)
        if val_records
        else (train_x, train_y)
    )
    test_x, test_y = build_pair_dataset(test_records, args.feature_mode)
    mean, std, [train_xs, val_xs, test_xs] = standardize(
        train_x, train_x, val_x, test_x
    )
    model, train_info = train_model(train_xs, train_y, val_xs, val_y, args)
    device = torch.device(train_info["device"])
    train_probs = predict_probs(model, train_xs, device)
    val_probs = predict_probs(model, val_xs, device)
    test_probs = predict_probs(model, test_xs, device)
    threshold, val_tree_eval = tune_threshold(
        val_records or fit_records, args.feature_mode, model, mean, std, device
    )

    tree_metrics = {
        "train": evaluate_records(
            fit_records, args.feature_mode, model, mean, std, threshold, device
        ),
        "val": (
            evaluate_records(
                val_records, args.feature_mode, model, mean, std, threshold, device
            )
            if val_records
            else None
        ),
        "test": evaluate_records(
            test_records, args.feature_mode, model, mean, std, threshold, device
        ),
        "all": evaluate_records(
            records, args.feature_mode, model, mean, std, threshold, device
        ),
    }
    exported = export_predictions(
        test_records,
        args.feature_mode,
        model,
        mean,
        std,
        threshold,
        device,
        Path(args.pred_output_dir),
    )

    report = {
        "model": "context_aware_merge_mlp",
        "feature_mode": args.feature_mode,
        "architecture": {
            "input_dim": int(train_x.shape[1]),
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
        },
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
        "tree_metrics": tree_metrics,
        "exported_test_predictions": {
            "output_dir": args.pred_output_dir,
            "num_objects": len(exported),
        },
        "training": train_info,
        "notes": [
            "MLP merge scorer augmented with per-step global context features (cluster count, sizes, mean/std of cluster reprs).",
            "Same greedy connected-components decoder as the baseline.",
            "Compare against the flat MLP without context (train_tree_planner_nn.py).",
        ],
        "config": vars(args),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    torch.save(
        {
            "model_state": model.cpu().state_dict(),
            "feature_mode": args.feature_mode,
            "input_dim": int(train_x.shape[1]),
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "mean": mean,
            "std": std,
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
