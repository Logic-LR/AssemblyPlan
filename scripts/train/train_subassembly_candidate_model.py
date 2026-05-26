#!/usr/bin/env python3
"""Train a subassembly-candidate predictor without manual composite tokens at inference."""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import itertools
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from eval.evaluate_composite_context_decoder import maximal_children
from eval.evaluate_paper_tree_metrics import average_metrics, build_tree_from_list, eval_tree, step_tree_from_child_specs
from export.export_tree_predictions_and_equivalence_report import tree_to_list
from train.train_tree_planner_baseline import Cluster, cluster_repr, cluster_token, node_actions, part_feature_map, split_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="experiments/svg_assembly/datasets/tree_generation_dataset.json")
    parser.add_argument("--feature-mode", choices=["geometry", "svg", "svg_geometry"], default="svg_geometry")
    parser.add_argument("--max-subset-size", type=int, default=6)
    parser.add_argument("--max-complement-size", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--train-negatives-per-positive",
        type=int,
        default=0,
        help="For training only, sample at most this many negative candidate sets per positive; 0 keeps all negatives.",
    )
    parser.add_argument(
        "--selection-penalty",
        type=float,
        default=0.0,
        help="Penalty subtracted from validation hard F1 for each selected candidate on average.",
    )
    parser.add_argument("--output", default="experiments/svg_assembly/reports/subassembly_candidate_svg_geometry_report.json")
    parser.add_argument("--model-output", default="experiments/svg_assembly/reports/subassembly_candidate_svg_geometry_model.pt")
    parser.add_argument("--pred-output-dir", default="experiments/svg_assembly/tree_planner_predictions_subassembly_candidates_test")
    return parser.parse_args()


class CandidateMLP(nn.Module):
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


def nonroot_tree_sets(record: Dict[str, Any]) -> set[Cluster]:
    n_parts = int(record["num_parts"])
    root = frozenset(range(n_parts))
    tree = build_tree_from_list(record["assembly_tree"])
    out = {frozenset(action["parent"]) for action in node_actions(tree)}
    return {cluster for cluster in out if len(cluster) > 1 and cluster != root}


def candidate_sets(record: Dict[str, Any], max_subset_size: int, max_complement_size: int) -> List[Cluster]:
    n_parts = int(record["num_parts"])
    parts = tuple(range(n_parts))
    root = frozenset(parts)
    out: set[Cluster] = set()
    direct_max = min(max_subset_size, n_parts - 1)
    for size in range(2, direct_max + 1):
        for combo in itertools.combinations(parts, size):
            out.add(frozenset(combo))
    comp_max = min(max_complement_size, n_parts - 2)
    for comp_size in range(1, comp_max + 1):
        for combo in itertools.combinations(parts, comp_size):
            cluster = root - frozenset(combo)
            if 1 < len(cluster) < n_parts:
                out.add(cluster)
    return sorted(out, key=lambda item: (len(item), tuple(sorted(item))))


def candidate_feature_from_features(cluster: Cluster, features: Dict[int, np.ndarray], n_parts: int) -> np.ndarray:
    base = cluster_repr(cluster, features)
    n_parts = max(n_parts, 1)
    extras = np.asarray(
        [
            len(cluster) / n_parts,
            (n_parts - len(cluster)) / n_parts,
            1.0 if len(cluster) == 2 else 0.0,
            1.0 if len(cluster) > n_parts / 2 else 0.0,
        ],
        dtype=np.float32,
    )
    return np.concatenate([base, extras]).astype(np.float32)


def candidate_feature(record: Dict[str, Any], cluster: Cluster, mode: str) -> np.ndarray:
    return candidate_feature_from_features(cluster, part_feature_map(record, mode), int(record["num_parts"]))


def build_candidate_dataset(
    records: Sequence[Dict[str, Any]],
    mode: str,
    max_subset_size: int,
    max_complement_size: int,
    negatives_per_positive: int = 0,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    xs: List[np.ndarray] = []
    ys: List[float] = []
    counts = []
    kept_counts = []
    positives = []
    kept_positives = []
    rng = random.Random(seed)
    for record_index, record in enumerate(records):
        features = part_feature_map(record, mode)
        n_parts = int(record["num_parts"])
        gold = nonroot_tree_sets(record)
        candidates = candidate_sets(record, max_subset_size, max_complement_size)
        original_candidate_count = len(candidates)
        positive_candidates = [cluster for cluster in candidates if cluster in gold]
        negative_candidates = [cluster for cluster in candidates if cluster not in gold]
        if negatives_per_positive > 0 and negative_candidates:
            # Keep every positive and sample a bounded set of negatives so the
            # classifier is not dominated by the combinatorial candidate space.
            local_rng = random.Random(seed + record_index * 1009)
            local_rng.shuffle(negative_candidates)
            target_negatives = negatives_per_positive * max(len(positive_candidates), 1)
            target_negatives = min(len(negative_candidates), target_negatives)
            candidates = positive_candidates + negative_candidates[:target_negatives]
            rng.shuffle(candidates)
        counts.append(original_candidate_count)
        kept_counts.append(len(candidates))
        positives.append(len(positive_candidates))
        kept_positives.append(sum(1 for cluster in candidates if cluster in gold))
        for cluster in candidates:
            xs.append(candidate_feature_from_features(cluster, features, n_parts))
            ys.append(1.0 if cluster in gold else 0.0)
    info = {
        "objects": len(records),
        "examples": int(len(ys)),
        "positive_examples": int(sum(kept_positives)),
        "avg_candidates_per_object": float(sum(counts) / len(counts)) if counts else 0.0,
        "avg_positive_candidates_per_object": float(sum(kept_positives) / len(kept_positives)) if kept_positives else 0.0,
        "sampled_negatives_per_positive": int(negatives_per_positive),
        "original_positive_examples": int(sum(positives)),
        "avg_kept_candidates_per_object": float(sum(kept_counts) / len(kept_counts)) if kept_counts else 0.0,
    }
    return np.vstack(xs).astype(np.float32), np.asarray(ys, dtype=np.float32), info


def standardize(train_x: np.ndarray, *arrays: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean, std, [((arr - mean) / std).astype(np.float32) for arr in arrays]


def candidate_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    pred = probs >= threshold
    gold = labels > 0.5
    tp = int(np.logical_and(pred, gold).sum())
    fp = int(np.logical_and(pred, ~gold).sum())
    fn = int(np.logical_and(~pred, gold).sum())
    tn = int(np.logical_and(~pred, ~gold).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"accuracy": (tp + tn) / len(labels), "precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


@torch.no_grad()
def predict_probs(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int = 8192) -> np.ndarray:
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
) -> Tuple[CandidateMLP, Dict[str, Any]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CandidateMLP(train_x.shape[1], args.hidden_dim, args.dropout).to(device)
    loader = DataLoader(TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)), batch_size=args.batch_size, shuffle=True)
    pos_weight = torch.tensor([(len(train_y) - float(train_y.sum())) / max(float(train_y.sum()), 1.0)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(yb)
            total += len(yb)
        if epoch == args.epochs or epoch % 10 == 0:
            val_probs = predict_probs(model, val_x, device)
            val_metrics = candidate_metrics(val_probs, val_y)
            if val_metrics["f1"] > best_val:
                best_val = val_metrics["f1"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            history.append({"epoch": epoch, "loss": total_loss / max(total, 1), "val_candidate_f1": val_metrics["f1"], "best_val_candidate_f1": best_val})
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, {"device": str(device), "history": history, "best_val_candidate_f1": best_val}


@torch.no_grad()
def score_record_candidates(
    record: Dict[str, Any],
    mode: str,
    model: nn.Module,
    mean: np.ndarray,
    std: np.ndarray,
    max_subset_size: int,
    max_complement_size: int,
    device: torch.device,
) -> List[Tuple[float, Cluster]]:
    candidates = candidate_sets(record, max_subset_size, max_complement_size)
    if not candidates:
        return []
    features = part_feature_map(record, mode)
    n_parts = int(record["num_parts"])
    x = np.vstack([candidate_feature_from_features(cluster, features, n_parts) for cluster in candidates]).astype(np.float32)
    x = ((x - mean) / std).astype(np.float32)
    probs = predict_probs(model, x, device)
    return sorted([(float(prob), cluster) for prob, cluster in zip(probs, candidates)], key=lambda item: item[0], reverse=True)


def score_records(
    records: Sequence[Dict[str, Any]],
    mode: str,
    model: nn.Module,
    mean: np.ndarray,
    std: np.ndarray,
    max_subset_size: int,
    max_complement_size: int,
    device: torch.device,
) -> List[List[Tuple[float, Cluster]]]:
    return [
        score_record_candidates(record, mode, model, mean, std, max_subset_size, max_complement_size, device)
        for record in records
    ]


def decode_from_predicted_sets(record: Dict[str, Any], predicted_sets: Sequence[Cluster]) -> Any:
    n_parts = int(record["num_parts"])
    root = frozenset(range(n_parts))
    known: set[Cluster] = {frozenset([part]) for part in range(n_parts)}
    child_specs: List[List[str]] = []
    for part_set in sorted(set(predicted_sets), key=lambda item: (len(item), tuple(sorted(item)))):
        if len(part_set) <= 1 or part_set in known or not part_set <= root:
            continue
        children = maximal_children(part_set, list(known))
        if len(children) < 2:
            continue
        child_specs.append([cluster_token(child) for child in children])
        known.add(part_set)
    if root not in known:
        children = maximal_children(root, list(known))
        if len(children) >= 2:
            child_specs.append([cluster_token(child) for child in children])
    return step_tree_from_child_specs(child_specs, n_parts)


def laminar_with_selected(candidate: Cluster, selected: Sequence[Cluster]) -> bool:
    for other in selected:
        if candidate & other and not (candidate <= other or other <= candidate):
            return False
    return True


def selected_sets(
    scored: Sequence[Tuple[float, Cluster]],
    threshold: float,
    max_selected: int,
    policy: str,
) -> List[Cluster]:
    if policy == "laminar":
        out: List[Cluster] = []
        for score, cluster in scored:
            if score < threshold:
                continue
            if laminar_with_selected(cluster, out):
                out.append(cluster)
            if max_selected > 0 and len(out) >= max_selected:
                break
        return out
    out = [cluster for score, cluster in scored if score >= threshold]
    if max_selected > 0:
        out = out[:max_selected]
    return out


def evaluate_records(
    records: Sequence[Dict[str, Any]],
    threshold: float,
    max_selected: int,
    scored_records: Sequence[Sequence[Tuple[float, Cluster]]],
    policy: str,
) -> Dict[str, Any]:
    rows = []
    examples = []
    selected_counts = []
    gold_recalls = []
    for record, scored in zip(records, scored_records):
        pred_sets = selected_sets(scored, threshold, max_selected, policy)
        gold = nonroot_tree_sets(record)
        selected_counts.append(len(pred_sets))
        gold_recalls.append(len(set(pred_sets) & gold) / len(gold) if gold else 1.0)
        gt = build_tree_from_list(record["assembly_tree"])
        pred = decode_from_predicted_sets(record, pred_sets)
        metrics = eval_tree(gt, pred)
        rows.append(metrics)
        if len(examples) < 10:
            examples.append(
                {
                    "category": record["category"],
                    "name": record["name"],
                    "metrics": metrics,
                    "selected_candidates": [cluster_token(cluster) for cluster in pred_sets[:20]],
                    "gold_nonroot_sets": [cluster_token(cluster) for cluster in sorted(gold, key=lambda item: (len(item), tuple(sorted(item))))],
                }
            )
    return {
        "objects": len(records),
        "metrics": average_metrics(rows),
        "avg_selected_candidates": float(sum(selected_counts) / len(selected_counts)) if selected_counts else 0.0,
        "avg_gold_candidate_recall": float(sum(gold_recalls) / len(gold_recalls)) if gold_recalls else 0.0,
        "examples": examples,
    }


def tune_decoder(
    records: Sequence[Dict[str, Any]],
    scored_records: Sequence[Sequence[Tuple[float, Cluster]]],
    selection_penalty: float,
) -> Tuple[float, int, str, Dict[str, Any]]:
    best_threshold = 0.5
    best_max_selected = 16
    best_policy = "topk"
    best_eval = None
    best_score = -1.0
    thresholds = list(np.linspace(0.2, 0.9, 15)) + [0.93, 0.95, 0.97, 0.99]
    for threshold in thresholds:
        for max_selected in [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]:
            for policy in ["topk", "laminar"]:
                result = evaluate_records(records, float(threshold), max_selected, scored_records, policy)
                score = result["metrics"]["hard"]["f1"] - selection_penalty * result["avg_selected_candidates"]
                if score > best_score:
                    best_score = score
                    best_threshold = float(threshold)
                    best_max_selected = int(max_selected)
                    best_policy = policy
                    best_eval = result
    assert best_eval is not None
    best_eval["selection_objective"] = {
        "metric": "val hard F1 - selection_penalty * avg_selected_candidates",
        "selection_penalty": float(selection_penalty),
        "score": float(best_score),
    }
    return best_threshold, best_max_selected, best_policy, best_eval


def export_predictions(
    records: Sequence[Dict[str, Any]],
    mode: str,
    threshold: float,
    max_selected: int,
    scored_records: Sequence[Sequence[Tuple[float, Cluster]]],
    policy: str,
    output_dir: Path,
) -> List[Dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    exported = []
    for record, scored in zip(records, scored_records):
        pred_sets = selected_sets(scored, threshold, max_selected, policy)
        gt = build_tree_from_list(record["assembly_tree"])
        pred = decode_from_predicted_sets(record, pred_sets)
        metrics = eval_tree(gt, pred)
        obj_dir = output_dir / record["category"] / record["name"]
        obj_dir.mkdir(parents=True, exist_ok=True)
        path = obj_dir / "assembly_tree_prediction.json"
        payload = {
            "category": record["category"],
            "name": record["name"],
            "split": record.get("split"),
            "feature_mode": f"{mode}_predicted_subassemblies",
            "threshold": threshold,
            "max_selected": max_selected,
            "selection_policy": policy,
            "predicted_subassembly_sets": [
                {"parts": [str(part) for part in sorted(cluster)], "token": cluster_token(cluster)}
                for cluster in pred_sets
            ],
            "predicted_assembly_tree": tree_to_list(pred),
            "ground_truth_assembly_tree": record["assembly_tree"],
            "paper_tree_metrics": metrics,
            "part_tokens": record["part_tokens"],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        exported.append(
            {
                "category": record["category"],
                "name": record["name"],
                "split": record.get("split"),
                "path": str(path.as_posix()),
                "paper_tree_metrics": metrics,
                "num_predicted_subassembly_sets": len(pred_sets),
            }
        )
    (output_dir / "index.json").write_text(json.dumps(exported, ensure_ascii=False, indent=2), encoding="utf-8")
    return exported


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    records = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    fit_records, val_records, test_records = split_records(records, args.val_fraction, args.seed)

    train_x, train_y, train_info = build_candidate_dataset(
        fit_records,
        args.feature_mode,
        args.max_subset_size,
        args.max_complement_size,
        args.train_negatives_per_positive,
        args.seed,
    )
    val_x, val_y, val_info = (
        build_candidate_dataset(val_records, args.feature_mode, args.max_subset_size, args.max_complement_size)
        if val_records
        else (train_x, train_y, train_info)
    )
    test_x, test_y, test_info = build_candidate_dataset(test_records, args.feature_mode, args.max_subset_size, args.max_complement_size)
    mean, std, [train_xs, val_xs, test_xs] = standardize(train_x, train_x, val_x, test_x)

    model, train_state = train_model(train_xs, train_y, val_xs, val_y, args)
    device = torch.device(train_state["device"])
    train_probs = predict_probs(model, train_xs, device)
    val_probs = predict_probs(model, val_xs, device)
    test_probs = predict_probs(model, test_xs, device)
    fit_scores = score_records(
        fit_records,
        args.feature_mode,
        model,
        mean,
        std,
        args.max_subset_size,
        args.max_complement_size,
        device,
    )
    val_eval_records = val_records or fit_records
    val_scores = score_records(
        val_eval_records,
        args.feature_mode,
        model,
        mean,
        std,
        args.max_subset_size,
        args.max_complement_size,
        device,
    )
    test_scores = score_records(
        test_records,
        args.feature_mode,
        model,
        mean,
        std,
        args.max_subset_size,
        args.max_complement_size,
        device,
    )
    all_scores = score_records(
        records,
        args.feature_mode,
        model,
        mean,
        std,
        args.max_subset_size,
        args.max_complement_size,
        device,
    )
    threshold, max_selected, selection_policy, val_tree_eval = tune_decoder(
        val_eval_records,
        val_scores,
        args.selection_penalty,
    )

    tree_metrics = {
        "train": evaluate_records(fit_records, threshold, max_selected, fit_scores, selection_policy),
        "val": evaluate_records(val_records, threshold, max_selected, val_scores, selection_policy) if val_records else None,
        "test": evaluate_records(test_records, threshold, max_selected, test_scores, selection_policy),
        "all": evaluate_records(records, threshold, max_selected, all_scores, selection_policy),
    }
    exported = export_predictions(
        test_records,
        args.feature_mode,
        threshold,
        max_selected,
        test_scores,
        selection_policy,
        Path(args.pred_output_dir),
    )

    report = {
        "model": "subassembly_candidate_predictor",
        "feature_mode": args.feature_mode,
        "dataset": args.dataset,
        "splits": {
            "fit_objects": len(fit_records),
            "val_objects": len(val_records),
            "test_objects": len(test_records),
            "all_objects": len(records),
        },
        "candidate_examples": {
            "train": train_info,
            "val": val_info,
            "test": test_info,
        },
        "candidate_metrics": {
            "train": candidate_metrics(train_probs, train_y),
            "val": candidate_metrics(val_probs, val_y),
            "test": candidate_metrics(test_probs, test_y),
        },
        "candidate_metrics_at_decoder_threshold": {
            "train": candidate_metrics(train_probs, train_y, threshold),
            "val": candidate_metrics(val_probs, val_y, threshold),
            "test": candidate_metrics(test_probs, test_y, threshold),
        },
        "decoder_selection": {
            "threshold": threshold,
            "max_selected": max_selected,
            "selection_policy": selection_policy,
            "selection_metric": "val hard F1 - selection_penalty * avg_selected_candidates",
            "selection_penalty": args.selection_penalty,
            "val_tree_eval": val_tree_eval,
        },
        "tree_metrics": tree_metrics,
        "exported_test_predictions": {
            "output_dir": args.pred_output_dir,
            "num_objects": len(exported),
        },
        "training": train_state,
        "notes": [
            "This model predicts subassembly candidate part sets from primitive part tokens only.",
            "It does not consume manual composite_tokens at inference; GT tree nodes are used only as training supervision.",
            "Training can downsample negative candidate sets; validation/test scoring still enumerates the full candidate space.",
            "The decoder turns predicted candidate sets into an assembly tree and adds the full root node automatically.",
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
            "max_selected": max_selected,
            "selection_policy": selection_policy,
            "max_subset_size": args.max_subset_size,
            "max_complement_size": args.max_complement_size,
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
