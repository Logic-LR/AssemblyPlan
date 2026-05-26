#!/usr/bin/env python3
"""GRPO fine-tuning of a merge scorer with SVG-derived rewards.

Warm-starts from a pretrained context-aware MLP (train_tree_planner_context.py),
then uses Group Relative Policy Optimization to shape the policy with:

  1. SVG step-group coherence: does each merge produce a subassembly that
     appears in at least one manual step?
  2. GT tree F1: conventional supervised signal (where available).

At each merge step the policy samples one pair from the scored distribution
(temperature τ), producing diverse trees. Rewards are normalized within each
object's group, and the policy is updated with clipped importance sampling
plus a KL penalty towards the reference model.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from eval.evaluate_paper_tree_metrics import (
    Node,
    PartSet,
    average_metrics,
    build_tree_from_list,
    eval_tree,
    nonleaf_nodes,
    step_tree_from_child_specs,
)
from export.export_tree_predictions_and_equivalence_report import tree_to_list
from train.train_tree_planner_baseline import (
    Cluster,
    cluster_repr,
    cluster_token,
    composite_feature_map,
    part_feature_map,
    part_count,
    split_records,
    uses_composite_features,
)
from train.train_tree_planner_context import (
    ContextMergeMLP,
    _global_context_features,
    pair_feature_context,
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
    parser.add_argument(
        "--warm-start",
        default="",
        help="Pretrained context MLP checkpoint (empty = from scratch)",
    )
    parser.add_argument("--from-scratch", action="store_true", help="Random init, no pretraining")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--samples-per-object", type=int, default=8, help="K in GRPO")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k-sample", type=int, default=0, help="Top-K sampling (0=softmax all)")
    parser.add_argument("--kl-beta", type=float, default=0.01, help="KL penalty weight")
    parser.add_argument("--clip-eps", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--svg-reward-weight", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        default="experiments/svg_assembly/reports/grpo_svg_geometry_report.json",
    )
    parser.add_argument(
        "--model-output",
        default="experiments/svg_assembly/reports/grpo_svg_geometry_model.pt",
    )
    parser.add_argument(
        "--pred-output-dir",
        default="experiments/svg_assembly/grpo_predictions_svg_geometry_test",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# SVG reward helpers
# ---------------------------------------------------------------------------


def _parse_part_set(raw: str | int) -> PartSet:
    if isinstance(raw, int):
        return frozenset([raw])
    return frozenset(int(p) for p in str(raw).split(",") if p)


def _all_subassemblies(step_groups: List[Dict[str, Any]]) -> Set[PartSet]:
    """Collect all part-sets that appear as step groups in the manual."""
    subassemblies: Set[PartSet] = set()
    for step in step_groups:
        for part_str in step.get("parts") or []:
            ps = _parse_part_set(part_str)
            if len(ps) >= 2:
                subassemblies.add(ps)
    return subassemblies


def svg_reward(tree: Node, subassemblies: Set[PartSet]) -> float:
    """Fraction of non-leaf nodes whose part-set appears in a manual step."""
    nodes = nonleaf_nodes(tree)
    if not nodes:
        return 0.0
    hits = sum(1 for n in nodes if n.parts in subassemblies)
    return hits / len(nodes)


def gt_f1_reward(pred_tree: Node, gt_tree: Node) -> float:
    """Simple F1 between predicted and GT trees."""
    metrics = eval_tree(gt_tree, pred_tree)
    return float(metrics["simple"]["f1"])


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


@torch.no_grad()
def _sample_merge_step(
    record: Dict[str, Any],
    clusters: List[Cluster],
    mode: str,
    model: nn.Module,
    mean: np.ndarray,
    std: np.ndarray,
    temperature: float,
    top_k: int,
    device: torch.device,
) -> Tuple[Cluster, Cluster, float, np.ndarray]:
    """Sample one merge pair. Returns (a, b, log_prob, full_probs)."""
    features = part_feature_map(record, mode)
    composites = (
        composite_feature_map(record, mode)
        if uses_composite_features(mode)
        else None
    )
    global_ctx = _global_context_features(clusters, features, composites)

    raw_feats = []
    pairs: List[Tuple[int, int]] = []
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            raw_feats.append(
                pair_feature_context(
                    clusters[i], clusters[j], features, composites, global_ctx
                )
            )
            pairs.append((i, j))

    x = ((np.vstack(raw_feats).astype(np.float32) - mean) / std).astype(np.float32)
    logits = (
        model(torch.from_numpy(x).to(device)).cpu().numpy()
    )

    if top_k > 0 and top_k < len(pairs):
        # Top-K sampling: keep only top K, set rest to -inf
        top_indices = np.argpartition(logits, -top_k)[-top_k:]
        mask = np.full(len(logits), -1e10, dtype=np.float32)
        mask[top_indices] = logits[top_indices]
        scaled = mask / max(temperature, 1e-6)
    else:
        scaled = logits / max(temperature, 1e-6)

    scaled -= scaled.max()
    probs = np.exp(scaled) / np.exp(scaled).sum()

    chosen_idx = int(np.random.choice(len(pairs), p=probs))
    i, j = pairs[chosen_idx]
    log_prob = float(np.log(max(probs[chosen_idx], 1e-12)))

    return clusters[i], clusters[j], log_prob, probs


@torch.no_grad()
def sample_tree(
    record: Dict[str, Any],
    mode: str,
    model: nn.Module,
    mean: np.ndarray,
    std: np.ndarray,
    temperature: float,
    top_k: int,
    device: torch.device,
) -> Tuple[Node, List[float], List[np.ndarray]]:
    """Sample one assembly tree from the policy.

    Returns (tree, log_probs_per_step, probs_per_step).
    """
    current: Set[Cluster] = {
        frozenset([part]) for part in range(int(record["num_parts"]))
    }
    child_specs: List[List[str]] = []
    log_probs: List[float] = []
    all_probs: List[np.ndarray] = []

    max_steps = max(1, int(record["num_parts"]) * 2)
    for _ in range(max_steps):
        if len(current) <= 1:
            break

        clusters_list = sorted(
            current, key=lambda c: (len(c), tuple(sorted(c)))
        )

        a, b, log_prob, probs = _sample_merge_step(
            record, clusters_list, mode, model, mean, std, temperature, top_k, device
        )
        log_probs.append(log_prob)
        all_probs.append(probs)

        parent = a | b
        child_specs.append(
            [cluster_token(c) for c in sorted([a, b], key=lambda x: (len(x), tuple(sorted(x))))]
        )
        current.discard(a)
        current.discard(b)
        current.add(parent)

    tree = step_tree_from_child_specs(child_specs, int(record["num_parts"]))
    return tree, log_probs, all_probs


# ---------------------------------------------------------------------------
# GRPO training
# ---------------------------------------------------------------------------


def _compute_advantages(rewards: List[float]) -> np.ndarray:
    arr = np.asarray(rewards, dtype=np.float32)
    mean = arr.mean()
    std = arr.std()
    if std < 1e-8:
        std = 1.0
    return (arr - mean) / std


def grpo_train_one_epoch(
    model: nn.Module,
    ref_model: nn.Module,
    records: Sequence[Dict[str, Any]],
    mode: str,
    mean: np.ndarray,
    std: np.ndarray,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, float]:
    model.train()
    ref_model.eval()

    total_loss = 0.0
    total_kl = 0.0
    total_objects = 0
    all_rewards: List[float] = []

    for record in records:
        gt_tree = build_tree_from_list(record["assembly_tree"])
        step_groups = record.get("manual_step_groups") or []
        subassemblies = _all_subassemblies(step_groups)

        # Sample K trees
        samples: List[Tuple[Node, List[float], List[np.ndarray]]] = []
        for _ in range(args.samples_per_object):
            samples.append(
                sample_tree(record, mode, model, mean, std, args.temperature, args.top_k_sample, device)
            )

        # Compute rewards
        rewards = []
        for tree, _, _ in samples:
            r_svg = svg_reward(tree, subassemblies)
            r_gt = gt_f1_reward(tree, gt_tree)
            r = args.svg_reward_weight * r_svg + (1.0 - args.svg_reward_weight) * r_gt
            rewards.append(r)

        all_rewards.extend(rewards)
        advantages = _compute_advantages(rewards)

        # GRPO loss per sample
        obj_loss = 0.0
        kl_loss = 0.0
        active_samples = 0

        features = part_feature_map(record, mode)
        composites = (
            composite_feature_map(record, mode)
            if uses_composite_features(mode)
            else None
        )

        for sample_idx, (tree, log_probs, all_probs) in enumerate(samples):
            adv = float(advantages[sample_idx])
            if len(log_probs) == 0:
                continue
            active_samples += 1

            # Replay the tree's merge sequence to get current policy probs
            current: Set[Cluster] = {
                frozenset([part]) for part in range(int(record["num_parts"]))
            }
            nodes = nonleaf_nodes(tree)
            # Sort nodes by depth to replay in order
            node_order: List[Node] = []
            for node in nodes:
                node_order.append(node)
            node_order.sort(key=lambda n: (len(n.parts),))

            step = 0
            for node in node_order:
                if len(node.children) < 2:
                    continue
                if step >= len(log_probs):
                    break

                # Find which current clusters correspond to the children
                child_clusters = [child.parts for child in node.children]
                clusters_list = sorted(
                    current, key=lambda c: (len(c), tuple(sorted(c)))
                )

                # Find indices of the two children in the current clusters
                try:
                    child_sets = [frozenset(c) for c in child_clusters]
                    # We need exactly 2 children for binary merge
                    if len(child_sets) != 2:
                        step += 1
                        continue
                    a_set, b_set = child_sets[0], child_sets[1]

                    # Build pair features and get current logits
                    global_ctx = _global_context_features(
                        clusters_list, features, composites
                    )
                    raw_feats = []
                    pair_indices: List[Tuple[int, int]] = []
                    for i in range(len(clusters_list)):
                        for j in range(i + 1, len(clusters_list)):
                            raw_feats.append(
                                pair_feature_context(
                                    clusters_list[i],
                                    clusters_list[j],
                                    features,
                                    composites,
                                    global_ctx,
                                )
                            )
                            pair_indices.append((i, j))

                    x = (
                        (np.vstack(raw_feats).astype(np.float32) - mean) / std
                    ).astype(np.float32)
                    logits = model(torch.from_numpy(x).to(device))

                    # Find the pair index for (a_set, b_set)
                    chosen_idx = None
                    for k, (i, j) in enumerate(pair_indices):
                        if (
                            clusters_list[i] == a_set
                            and clusters_list[j] == b_set
                        ) or (
                            clusters_list[i] == b_set
                            and clusters_list[j] == a_set
                        ):
                            chosen_idx = k
                            break

                    if chosen_idx is None:
                        step += 1
                        continue

                    # Current log_prob
                    logits_scaled = logits / max(args.temperature, 1e-6)
                    logits_scaled = logits_scaled - logits_scaled.max()
                    cur_probs = torch.softmax(logits_scaled, dim=0)
                    cur_log_prob = torch.log(cur_probs[chosen_idx].clamp_min(1e-12))

                    old_log_prob = log_probs[step]

                    # Importance sampling ratio
                    ratio = torch.exp(cur_log_prob - old_log_prob)

                    # Clipped objective
                    clipped = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps)
                    obj_loss += -torch.min(ratio * adv, clipped * adv)

                    # KL penalty: KL(π_θ || π_ref) at this state
                    with torch.no_grad():
                        ref_logits_scaled = (
                            ref_model(torch.from_numpy(x).to(device))
                            / max(args.temperature, 1e-6)
                        )
                        ref_logits_scaled = ref_logits_scaled - ref_logits_scaled.max()
                        ref_probs = torch.softmax(ref_logits_scaled, dim=0)
                    kl = (ref_probs * (torch.log(ref_probs.clamp_min(1e-12)) - torch.log(cur_probs.clamp_min(1e-12)))).sum()
                    kl_loss += kl

                    # Advance state
                    parent = a_set | b_set
                    current.discard(a_set)
                    current.discard(b_set)
                    current.add(parent)

                except Exception:
                    pass

                step += 1

        if active_samples > 0:
            loss = (obj_loss + args.kl_beta * kl_loss) / active_samples
            loss.backward()
            total_loss += float(obj_loss.item()) / active_samples
            total_kl += float(kl_loss.item()) / active_samples

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        total_objects += 1

    n = max(total_objects, 1)
    return {
        "loss": total_loss / n,
        "kl": total_kl / n,
        "avg_reward": float(np.mean(all_rewards)) if all_rewards else 0.0,
        "max_reward": float(np.max(all_rewards)) if all_rewards else 0.0,
    }


# ---------------------------------------------------------------------------
# Evaluation (same as context planner, but with greedy argmax)
# ---------------------------------------------------------------------------


@torch.no_grad()
def _greedy_merge_step(
    record: Dict[str, Any],
    clusters: List[Cluster],
    mode: str,
    model: nn.Module,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
) -> Tuple[Cluster, Cluster, float]:
    features = part_feature_map(record, mode)
    composites = (
        composite_feature_map(record, mode)
        if uses_composite_features(mode)
        else None
    )
    global_ctx = _global_context_features(clusters, features, composites)

    raw_feats = []
    pairs: List[Tuple[int, int]] = []
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            raw_feats.append(
                pair_feature_context(
                    clusters[i], clusters[j], features, composites, global_ctx
                )
            )
            pairs.append((i, j))

    x = ((np.vstack(raw_feats).astype(np.float32) - mean) / std).astype(np.float32)
    logits = model(torch.from_numpy(x).to(device)).cpu().numpy()
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))

    best_idx = int(np.argmax(probs))
    i, j = pairs[best_idx]
    return clusters[i], clusters[j], float(probs[best_idx])


@torch.no_grad()
def plan_tree_greedy(
    record: Dict[str, Any],
    mode: str,
    model: nn.Module,
    mean: np.ndarray,
    std: np.ndarray,
    threshold: float,
    device: torch.device,
) -> Any:
    from train_tree_planner_baseline import connected_components, cluster_token

    current: Set[Cluster] = {
        frozenset([part]) for part in range(int(record["num_parts"]))
    }
    child_specs: List[List[str]] = []
    max_steps = max(1, int(record["num_parts"]) * 2)
    for _ in range(max_steps):
        if len(current) <= 1:
            break
        clusters_list = sorted(
            current, key=lambda c: (len(c), tuple(sorted(c)))
        )

        # Score all pairs
        features = part_feature_map(record, mode)
        composites = (
            composite_feature_map(record, mode)
            if uses_composite_features(mode)
            else None
        )
        global_ctx = _global_context_features(clusters_list, features, composites)

        raw_feats = []
        scored: List[Tuple[float, Cluster, Cluster]] = []
        for i in range(len(clusters_list)):
            for j in range(i + 1, len(clusters_list)):
                feat = pair_feature_context(
                    clusters_list[i],
                    clusters_list[j],
                    features,
                    composites,
                    global_ctx,
                )
                raw_feats.append(feat)
        if not raw_feats:
            break

        x = (
            (np.vstack(raw_feats).astype(np.float32) - mean) / std
        ).astype(np.float32)
        logits = model(torch.from_numpy(x).to(device)).cpu().numpy()
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))

        for k, (i, j) in enumerate(
            [(i, j) for i in range(len(clusters_list)) for j in range(i + 1, len(clusters_list))]
        ):
            scored.append((float(probs[k]), clusters_list[i], clusters_list[j]))

        scored.sort(key=lambda item: item[0], reverse=True)

        best_prob, best_a, best_b = scored[0]
        edges = [(a, b) for prob, a, b in scored if prob >= threshold]
        group = [best_a, best_b]
        if edges:
            comps = connected_components(clusters_list, edges)
            best_set = best_a | best_b
            for comp in comps:
                comp_union = frozenset().union(*comp)
                if best_set <= comp_union:
                    group = comp
                    break
        parent = frozenset().union(*group)
        child_specs.append(
            [
                cluster_token(c)
                for c in sorted(
                    group, key=lambda item: (len(item), tuple(sorted(item)))
                )
            ]
        )
        for c in group:
            current.discard(c)
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
    for record in records:
        gt = build_tree_from_list(record["assembly_tree"])
        pred = plan_tree_greedy(record, mode, model, mean, std, threshold, device)
        rows.append(eval_tree(gt, pred))
    return {"objects": len(records), "metrics": average_metrics(rows)}


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

    # Model initialization: warm-start or from scratch
    if args.from_scratch or not args.warm_start:
        # From-scratch: compute input dim from data, random init, no standardization
        from train.train_tree_planner_context import training_examples_with_context, build_pair_dataset
        temp_x, _ = build_pair_dataset([fit_records[0]], args.feature_mode)
        input_dim = int(temp_x.shape[1])
        hidden_dim = args.hidden_dim
        dropout = 0.15

        model = ContextMergeMLP(input_dim, hidden_dim, dropout).to(device)
        model.train()

        ref_model = ContextMergeMLP(input_dim, hidden_dim, dropout).to(device)
        ref_model.load_state_dict(model.state_dict())
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)

        # Compute standardization from training data
        all_x, _ = build_pair_dataset(fit_records, args.feature_mode)
        mean = all_x.mean(axis=0).astype(np.float32)
        std = all_x.std(axis=0).astype(np.float32)
        std[std < 1e-6] = 1.0

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

        print(f"From-scratch GRPO")
        print(f"  input_dim={input_dim} hidden_dim={hidden_dim} feature_mode={args.feature_mode}")
        print(f"  K={args.samples_per_object} τ={args.temperature} β={args.kl_beta}")
        print(f"  train_objects={len(fit_records)} val={len(val_records)} test={len(test_records)}")

        warm_val = {"metrics": {"simple": {"f1": 0.0}, "hard": {"f1": 0.0}}}
        warm_test = {"metrics": {"simple": {"f1": 0.0}, "hard": {"f1": 0.0}}}
        warm_threshold = 0.5
    else:
        ckpt = torch.load(args.warm_start, map_location=device, weights_only=False)
        input_dim = int(ckpt["input_dim"])
        hidden_dim = int(ckpt["hidden_dim"])
        dropout = float(ckpt.get("dropout", 0.15))
        mean = ckpt["mean"]
        std = ckpt["std"]

        model = ContextMergeMLP(input_dim, hidden_dim, dropout).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.train()

        ref_model = ContextMergeMLP(input_dim, hidden_dim, dropout).to(device)
        ref_model.load_state_dict(ckpt["model_state"])
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

        print(f"Warm-started from {args.warm_start}")
        print(f"  input_dim={input_dim} hidden_dim={hidden_dim} dropout={dropout}")
        print(f"  feature_mode={args.feature_mode}")
        print(f"  K={args.samples_per_object} τ={args.temperature} β={args.kl_beta}")
        print(f"  svg_reward_weight={args.svg_reward_weight}")
        print(f"  train_objects={len(fit_records)} val={len(val_records)} test={len(test_records)}")

        warm_threshold, warm_val = tune_threshold(
            val_records or fit_records, args.feature_mode, model, mean, std, device
        )
        warm_test = evaluate_records(
            test_records, args.feature_mode, model, mean, std, warm_threshold, device
        )
        print(
            f"Warm-start:  val Simple={warm_val['metrics']['simple']['f1']:.4f} "
            f"Hard={warm_val['metrics']['hard']['f1']:.4f}  "
            f"test Simple={warm_test['metrics']['simple']['f1']:.4f} "
            f"Hard={warm_test['metrics']['hard']['f1']:.4f}"
        )

    history: List[Dict[str, Any]] = []
    best_val_score = -1.0
    best_state = None
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        train_info = grpo_train_one_epoch(
            model,
            ref_model,
            fit_records,
            args.feature_mode,
            mean,
            std,
            optimizer,
            device,
            args,
        )

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            _, val_eval = tune_threshold(
                val_records or fit_records,
                args.feature_mode,
                model,
                mean,
                std,
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
                    "train_loss": train_info["loss"],
                    "train_kl": train_info["kl"],
                    "avg_reward": train_info["avg_reward"],
                    "max_reward": train_info["max_reward"],
                    "val_hard_f1": val_score,
                    "best_val_hard_f1": best_val_score,
                }
            )
            print(
                f"epoch {epoch:3d}  loss={train_info['loss']:.4f}  "
                f"kl={train_info['kl']:.4f}  avg_r={train_info['avg_reward']:.3f}  "
                f"val_hard={val_score:.4f}  best={best_val_score:.4f}",
                flush=True,
            )

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # Final evaluation
    final_threshold, val_eval = tune_threshold(
        val_records or fit_records, args.feature_mode, model, mean, std, device
    )
    test_eval = evaluate_records(
        test_records, args.feature_mode, model, mean, std, final_threshold, device
    )
    all_eval = evaluate_records(
        records, args.feature_mode, model, mean, std, final_threshold, device
    )

    report = {
        "model": "grpo_context_mlp",
        "feature_mode": args.feature_mode,
        "warm_start": args.warm_start,
        "grpo_config": {
            "samples_per_object": args.samples_per_object,
            "temperature": args.temperature,
            "kl_beta": args.kl_beta,
            "clip_eps": args.clip_eps,
            "svg_reward_weight": args.svg_reward_weight,
            "epochs": args.epochs,
        },
        "architecture": {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
        },
        "dataset": args.dataset,
        "splits": {
            "fit_objects": len(fit_records),
            "val_objects": len(val_records),
            "test_objects": len(test_records),
        },
        "warm_start_eval": {
            "val": warm_val,
            "test": warm_test,
            "threshold": warm_threshold,
        },
        "training": {
            "best_epoch": best_epoch,
            "best_val_hard_f1": best_val_score,
            "history": history,
            "device": str(device),
        },
        "tree_metrics": {
            "val": val_eval,
            "test": test_eval,
            "all": all_eval,
        },
        "notes": [
            "GRPO fine-tuning from a pretrained context-aware MLP.",
            "Reward = svg_reward_weight * SVG coherence + (1-w) * GT F1.",
            "SVG coherence: fraction of non-leaf nodes whose part-set appears in a manual step group.",
            "At inference, uses greedy connected-components decoding (same as baseline).",
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
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "mean": mean,
            "std": std,
            "threshold": final_threshold,
            "grpo_config": {
                "samples_per_object": args.samples_per_object,
                "temperature": args.temperature,
                "kl_beta": args.kl_beta,
                "svg_reward_weight": args.svg_reward_weight,
            },
            "config": vars(args),
        },
        args.model_output,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote {out}")
    print(f"Wrote {args.model_output}")


if __name__ == "__main__":
    main()
