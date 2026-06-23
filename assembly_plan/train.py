"""Training loop for assembly plan generation.

Supports:
1. Supervised training (BCE on pairwise merge labels)
2. GRPO fine-tuning (optimize Tree F1 directly)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .data import (
    ObjectRecord, extract_merge_examples, load_dataset, split_dataset,
)
from .decoder import beam_search_decode, greedy_decode, group_greedy_decode, sample_decode
from .evaluate import build_tree_from_list, eval_tree, average_metrics
from .model import GNNMergeModel, build_model


# ---------------------------------------------------------------------------
# Supervised training — GNN gets gradients
# ---------------------------------------------------------------------------

def _forward_record(
    model: GNNMergeModel,
    record: ObjectRecord,
    examples: List[Dict[str, Any]],
    device: torch.device,
    include_context: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward one record through GNN + scorer, returning logits and labels.

    The GNN encoding is part of the forward pass so gradients flow through.
    """
    part_feats = torch.tensor(record.feature_matrix(), dtype=torch.float32, device=device)
    edge_index = torch.tensor(record.graph.edge_index, dtype=torch.long, device=device)
    part_embeds = model.encode_parts(part_feats, edge_index)

    logits_list = []
    labels_list = []

    for ex in examples:
        ca = sorted(ex["cluster_a"])
        cb = sorted(ex["cluster_b"])
        active = [list(sorted(c)) for c in ex["active_clusters"]] if include_context else None

        logit = model.score_pair(part_embeds, ca, cb, active)
        logits_list.append(logit)
        labels_list.append(ex["label"])

    if not logits_list:
        return torch.tensor([], device=device), torch.tensor([], device=device)

    return torch.stack(logits_list), torch.tensor(labels_list, dtype=torch.float32, device=device)


def train_supervised(
    train_records: List[ObjectRecord],
    val_records: List[ObjectRecord],
    model: GNNMergeModel,
    device: torch.device,
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    eval_every: int = 20,
    beam_width: int = 5,
    checkpoint_path: Optional[Path] = None,
    include_context: bool = True,
) -> Dict[str, Any]:
    """Supervised training with BCE loss on pairwise merge labels.

    GNN encoder and MLP scorer are trained jointly.
    """
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    # Pre-extract examples (labels only, features computed on-the-fly)
    print("Extracting training examples...")
    train_examples = []
    total_pos, total_neg = 0, 0
    for rec in train_records:
        exs = extract_merge_examples(rec)
        if exs:
            train_examples.append((rec, exs))
            total_pos += sum(1 for e in exs if e["label"] == 1.0)
            total_neg += sum(1 for e in exs if e["label"] == 0.0)
    print(f"  {total_pos} positive, {total_neg} negative examples across {len(train_examples)} objects")

    val_examples = []
    for rec in val_records:
        exs = extract_merge_examples(rec)
        if exs:
            val_examples.append((rec, exs))

    pos_weight = torch.tensor([total_neg / max(total_pos, 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_f1 = 0.0
    history = {"train_loss": [], "val_pair_f1": [], "val_tree_simple_f1": [], "val_tree_hard_f1": []}

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for rec, exs in tqdm(train_examples, desc=f"Epoch {epoch}", leave=False):
            optimizer.zero_grad()
            logits, labels = _forward_record(model, rec, exs, device, include_context)
            if logits.numel() == 0:
                continue
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        history["train_loss"].append(avg_loss)

        if epoch % eval_every == 0 or epoch == epochs:
            # Pair-level metrics on validation
            model.eval()
            all_val_logits = []
            all_val_labels = []
            with torch.no_grad():
                for rec, exs in val_examples:
                    logits, labels = _forward_record(model, rec, exs, device, include_context)
                    if logits.numel() > 0:
                        all_val_logits.append(logits)
                        all_val_labels.append(labels)

            if all_val_logits:
                val_logits = torch.cat(all_val_logits)
                val_labels = torch.cat(all_val_labels)
                val_probs = torch.sigmoid(val_logits)
                val_preds = (val_probs > 0.5).float()
                tp = ((val_preds == 1) & (val_labels == 1)).sum().item()
                fp = ((val_preds == 1) & (val_labels == 0)).sum().item()
                fn = ((val_preds == 0) & (val_labels == 1)).sum().item()
                pair_prec = tp / max(tp + fp, 1)
                pair_rec = tp / max(tp + fn, 1)
                pair_f1 = 2 * pair_prec * pair_rec / max(pair_prec + pair_rec, 1e-8)
            else:
                pair_f1 = 0.0

            # Tree-level metrics
            tree_metrics = _eval_tree_metrics(model, val_records, device, beam_width=1)
            simple_f1 = tree_metrics["simple"]["f1"]
            hard_f1 = tree_metrics["hard"]["f1"]
            history["val_pair_f1"].append(pair_f1)
            history["val_tree_simple_f1"].append(simple_f1)
            history["val_tree_hard_f1"].append(hard_f1)

            improved = ""
            if hard_f1 > best_val_f1:
                best_val_f1 = hard_f1
                improved = " *BEST*"
                if checkpoint_path:
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), checkpoint_path)

            print(f"  Epoch {epoch:4d}  loss={avg_loss:.4f}  "
                  f"pair_F1={pair_f1:.3f}  "
                  f"Simple={simple_f1:.3f}  Hard={hard_f1:.3f}{improved}")

    history["best_val_hard_f1"] = best_val_f1
    return history


# ---------------------------------------------------------------------------
# GRPO fine-tuning
# ---------------------------------------------------------------------------

def train_grpo(
    train_records: List[ObjectRecord],
    val_records: List[ObjectRecord],
    model: GNNMergeModel,
    device: torch.device,
    epochs: int = 50,
    lr: float = 5e-5,
    samples_per_object: int = 8,
    temperature: float = 1.2,
    clip_eps: float = 0.2,
    kl_coeff: float = 0.1,
    eval_every: int = 5,
    beam_width: int = 5,
    checkpoint_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """GRPO fine-tuning: optimize Tree F1 directly via policy gradient."""
    ref_model = build_model(
        part_feat_dim=model.gnn.input_proj.in_features,
        gnn_hidden=model.gnn.convs[0].out_channels,
        gnn_layers=len(model.gnn.convs),
        include_context=model.include_context,
    ).to(device)
    ref_model.load_state_dict(model.state_dict())
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    best_val_f1 = 0.0
    history = {"grpo_loss": [], "avg_reward": [], "val_hard_f1": []}

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        epoch_rewards = []

        for record in tqdm(train_records, desc=f"GRPO epoch {epoch}", leave=False):
            if record.num_parts < 2:
                continue

            # Precompute GNN embeddings once per object (saves redundant encoding)
            part_feats = torch.tensor(record.feature_matrix(), dtype=torch.float32, device=device)
            edge_index = torch.tensor(record.graph.edge_index, dtype=torch.long, device=device)
            cur_embeds = model.encode_parts(part_feats, edge_index)
            with torch.no_grad():
                ref_embeds = ref_model.encode_parts(part_feats, edge_index)

            # Sample K trees (reuse precomputed embeddings)
            sampled_trees = []
            old_log_probs = []
            for _ in range(samples_per_object):
                tree, log_prob = sample_decode(model, record, temperature, device, part_embeds=cur_embeds)
                sampled_trees.append(tree)
                old_log_probs.append(log_prob)

            # Compute rewards
            gt_tree = build_tree_from_list(record.assembly_tree)
            rewards = []
            for tree in sampled_trees:
                pred_tree = build_tree_from_list(tree)
                m = eval_tree(gt_tree, pred_tree)
                reward = m["simple"]["f1"] * 0.5 + m["hard"]["f1"] * 0.5
                rewards.append(reward)
            epoch_rewards.extend(rewards)

            r = torch.tensor(rewards, device=device)
            if r.std() > 1e-6:
                advantages = (r - r.mean()) / r.std()
            else:
                advantages = r - r.mean()

            cur_log_probs = []
            for tree in sampled_trees:
                _, lp = _replay_log_prob(model, record, tree, temperature, device, part_embeds=cur_embeds)
                cur_log_probs.append(lp)

            old_lp = torch.tensor(old_log_probs, device=device)
            cur_lp = torch.stack(cur_log_probs)
            ratio = torch.exp(cur_lp - old_lp)
            clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
            policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()

            ref_log_probs = []
            for tree in sampled_trees:
                with torch.no_grad():
                    _, lp = _replay_log_prob(ref_model, record, tree, temperature, device, with_grad=False, part_embeds=ref_embeds)
                ref_log_probs.append(lp.detach())
            ref_lp = torch.stack(ref_log_probs)
            kl = (cur_lp - ref_lp).mean()
            loss = policy_loss + kl_coeff * kl

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        history["grpo_loss"].append(np.mean(epoch_losses))
        history["avg_reward"].append(np.mean(epoch_rewards))

        if epoch % eval_every == 0 or epoch == epochs:
            tree_metrics = _eval_tree_metrics(model, val_records, device, beam_width)
            hard_f1 = tree_metrics["hard"]["f1"]
            simple_f1 = tree_metrics["simple"]["f1"]
            history["val_hard_f1"].append(hard_f1)

            # Save per-epoch checkpoint
            if checkpoint_path:
                epoch_ckpt = checkpoint_path.parent / f"{checkpoint_path.stem}_ep{epoch}.pt"
                epoch_ckpt.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), epoch_ckpt)

            improved = ""
            if hard_f1 > best_val_f1:
                best_val_f1 = hard_f1
                improved = " *BEST*"
                if checkpoint_path:
                    torch.save(model.state_dict(), checkpoint_path)

            log_line = (f"  GRPO epoch {epoch:3d}  loss={np.mean(epoch_losses):.4f}  "
                        f"reward={np.mean(epoch_rewards):.3f}  "
                        f"Simple={simple_f1:.3f}  Hard={hard_f1:.3f}{improved}")
            print(log_line)

            # Append to log file
            if checkpoint_path:
                log_path = checkpoint_path.parent / "grpo_train_log.txt"
                with open(log_path, "a") as f:
                    f.write(log_line + "\n")

    history["best_val_hard_f1"] = best_val_f1
    return history


def _replay_log_prob(
    model: GNNMergeModel,
    record: ObjectRecord,
    tree: Any,
    temperature: float,
    device: torch.device,
    with_grad: bool = True,
    part_embeds: Optional[torch.Tensor] = None,
) -> Tuple[Any, torch.Tensor]:
    """Replay a tree to compute its log probability under the model."""
    model.eval()
    if part_embeds is None:
        part_feats = torch.tensor(record.feature_matrix(), dtype=torch.float32, device=device)
        edge_index = torch.tensor(record.graph.edge_index, dtype=torch.long, device=device)
        part_embeds = model.encode_parts(part_feats, edge_index)

    merge_seq = _extract_merge_sequence(tree, record.num_parts)

    from .decoder import MergeState
    state = MergeState(active_clusters=[frozenset([i]) for i in range(record.num_parts)])
    total_log_prob = torch.tensor(0.0, device=device, requires_grad=not with_grad)

    for ca, cb in merge_seq:
        clusters = state.active_clusters
        pairs = []
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                pairs.append((list(sorted(clusters[i])), list(sorted(clusters[j]))))
        if not pairs:
            break

        logits = model.score_pairs_batch(
            part_embeds, pairs,
            [list(sorted(c)) for c in clusters],
        )
        scaled = logits / temperature
        probs = torch.softmax(scaled, dim=0)

        target_pair = (frozenset(ca), frozenset(cb))
        chosen_idx = None
        for idx, (pa, pb) in enumerate(pairs):
            if (frozenset(pa), frozenset(pb)) == target_pair or \
               (frozenset(pb), frozenset(pa)) == target_pair:
                chosen_idx = idx
                break

        if chosen_idx is not None:
            total_log_prob = total_log_prob + torch.log(probs[chosen_idx] + 1e-10)

        state = state.merge(frozenset(ca), frozenset(cb), 0.0)

    model.train()
    return tree, total_log_prob


def _extract_merge_sequence(tree: Any, num_parts: int) -> List[Tuple[frozenset, frozenset]]:
    """Extract merge sequence from nested tree by postorder traversal."""
    merges = []

    def _collect(node: Any) -> frozenset:
        if isinstance(node, int):
            return frozenset([node])
        child_sets = [_collect(c) for c in node]
        if len(child_sets) >= 2:
            merged = child_sets[0]
            for i in range(1, len(child_sets)):
                merges.append((merged, child_sets[i]))
                merged = frozenset(merged | child_sets[i])
        elif len(child_sets) == 1:
            merged = child_sets[0]
        else:
            merged = frozenset()
        return merged

    _collect(tree)
    return merges


# ---------------------------------------------------------------------------
# Tree-level evaluation helper
# ---------------------------------------------------------------------------

def _eval_tree_metrics(
    model: GNNMergeModel,
    records: List[ObjectRecord],
    device: torch.device,
    beam_width: int = 1,
    use_group_decode: bool = False,
    threshold: float = -1.5,
) -> Dict[str, Dict[str, float]]:
    """Evaluate tree-level Simple/Hard F1."""
    model.eval()
    metrics_list = []

    for record in records:
        if record.num_parts < 2:
            gt_tree = build_tree_from_list(record.assembly_tree)
            pred_tree = gt_tree
        else:
            with torch.no_grad():
                if use_group_decode:
                    pred = group_greedy_decode(model, record, device, threshold=threshold)
                elif beam_width > 1:
                    pred = beam_search_decode(model, record, beam_width, device)
                else:
                    pred = greedy_decode(model, record, device)
            gt_tree = build_tree_from_list(record.assembly_tree)
            pred_tree = build_tree_from_list(pred)

        m = eval_tree(gt_tree, pred_tree)
        metrics_list.append(m)

    return average_metrics(metrics_list)
