"""Assembly tree decoders: greedy and beam search.

Decoders take a trained model and an ObjectRecord, and produce
a predicted assembly tree in nested list format (matching main_data.json).
"""

from __future__ import annotations

import heapq
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import math

import torch
import numpy as np

from .data import ObjectRecord
from .model import GNNMergeModel


# ---------------------------------------------------------------------------
# State representation
# ---------------------------------------------------------------------------

@dataclass
class MergeState:
    """Tracks the current state of tree construction.

    active_clusters: list of frozensets, each is a set of primitive part IDs
    merge_history: list of (child_a, child_b) merges performed so far
    score: cumulative log-probability score
    """
    active_clusters: List[frozenset]
    merge_history: List[Tuple[frozenset, frozenset]] = field(default_factory=list)
    score: float = 0.0

    def copy(self) -> "MergeState":
        return MergeState(
            active_clusters=list(self.active_clusters),
            merge_history=list(self.merge_history),
            score=self.score,
        )

    def is_done(self) -> bool:
        return len(self.active_clusters) == 1

    def merge(self, ca: frozenset, cb: frozenset, log_prob: float) -> "MergeState":
        """Return a new state after merging ca and cb."""
        new_state = self.copy()
        new_state.active_clusters.remove(ca)
        new_state.active_clusters.remove(cb)
        new_state.active_clusters.append(frozenset(ca | cb))
        new_state.merge_history.append((ca, cb))
        new_state.score += log_prob
        return new_state

    def merge_group(self, group: List[frozenset]) -> "MergeState":
        """Return a new state after merging a group of clusters into a flat node.

        Records a single group merge entry in merge_history with flat_children.
        """
        new_state = self.copy()
        flat_children = []
        for g in group:
            flat_children.extend(sorted(g))
            new_state.active_clusters.remove(g)
        merged = frozenset().union(*group)
        new_state.active_clusters.append(merged)
        # Record as group merge: (first, last, flat_children)
        new_state.merge_history.append((group[0], group[-1], flat_children))
        return new_state


# ---------------------------------------------------------------------------
# Tree construction from merge history
# ---------------------------------------------------------------------------

def merge_history_to_tree(merge_history: list,
                          num_parts: int) -> Any:
    """Convert a sequence of merges into a nested assembly tree.

    Returns nested list format matching main_data.json's assembly_tree.
    Leaf nodes are ints, internal nodes are lists.

    merge_history entries can be:
      - (ca, cb): binary merge → creates [child_a, child_b]
      - (ca, cb, flat_children): group merge → creates flat_children list

    Example:
        merges = [({0}, {1}), ({0,1}, {2})]
        → [[0, 1], 2]
    """
    # Map each frozenset to its tree representation
    node_map: Dict[frozenset, Any] = {}
    for i in range(num_parts):
        node_map[frozenset([i])] = i

    for entry in merge_history:
        if len(entry) == 3:
            # Group merge: flat_children is a list of part IDs
            ca, cb, flat_children = entry
            merged = frozenset(flat_children)
            node_map[merged] = list(flat_children)
        else:
            ca, cb = entry
            merged = frozenset(ca | cb)
            child_a = node_map[ca]
            child_b = node_map[cb]
            node_map[merged] = _make_tree_node(child_a, child_b)

    # The final root should be the single remaining cluster
    root_key = frozenset(range(num_parts))
    return node_map.get(root_key, [])


def _make_tree_node(a: Any, b: Any) -> list:
    """Combine two subtrees into one node.

    Always creates a binary node [a, b] preserving nesting structure.
    This allows the decoder to produce hierarchical trees with multiple
    non-leaf nodes that can match GT intermediate nodes.
    """
    return [a, b]


# ---------------------------------------------------------------------------
# Greedy decoder
# ---------------------------------------------------------------------------

@torch.no_grad()
def greedy_decode(
    model: GNNMergeModel,
    record: ObjectRecord,
    device: torch.device = torch.device("cpu"),
) -> Any:
    """Greedy decoding: at each step, merge the highest-scoring pair.

    Returns assembly tree in nested list format.
    """
    model.eval()
    part_feats = torch.tensor(record.feature_matrix(), dtype=torch.float32, device=device)
    edge_index = torch.tensor(record.graph.edge_index, dtype=torch.long, device=device)

    part_embeds = model.encode_parts(part_feats, edge_index)

    state = MergeState(active_clusters=[frozenset([i]) for i in range(record.num_parts)])

    while not state.is_done():
        clusters = state.active_clusters
        pairs = []
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                pairs.append((list(sorted(clusters[i])), list(sorted(clusters[j]))))

        if not pairs:
            break

        logits = model.score_pairs_batch(part_embeds, pairs,
                                         [list(sorted(c)) for c in clusters])
        probs = torch.sigmoid(logits)
        best_idx = probs.argmax().item()
        best_prob = probs[best_idx].item()

        ca_list, cb_list = pairs[best_idx]
        ca = frozenset(ca_list)
        cb = frozenset(cb_list)
        log_prob = math.log(max(best_prob, 1e-10))
        state = state.merge(ca, cb, log_prob)

    return merge_history_to_tree(state.merge_history, record.num_parts)


# ---------------------------------------------------------------------------
# Group-aware greedy decoder
# ---------------------------------------------------------------------------

def _find_best_singleton_clique(
    clusters: List[frozenset],
    pairs: List[Tuple[list, list]],
    logits: torch.Tensor,
    min_size: int = 3,
    min_avg_logit: float = 0.0,
) -> Optional[List[int]]:
    """Find the best clique among singleton clusters using greedy expansion.

    A clique is a set of singletons where ALL pairwise logits are above
    min_avg_logit. We greedily expand from the highest-scoring pair.

    Returns list of cluster indices forming the clique, or None.
    """
    # Build logit matrix for singletons
    singleton_indices = [i for i, c in enumerate(clusters) if len(c) == 1]
    if len(singleton_indices) < min_size:
        return None

    # Map singleton cluster index → position in singleton_indices
    si_set = set(singleton_indices)

    # Build pairwise logit lookup: (i, j) → logit
    pair_logits = {}
    for idx, (ca_list, cb_list) in enumerate(pairs):
        ca = frozenset(ca_list)
        cb = frozenset(cb_list)
        if len(ca) == 1 and len(cb) == 1:
            ia = clusters.index(ca)
            ib = clusters.index(cb)
            if ia in si_set and ib in si_set:
                pair_logits[(ia, ib)] = logits[idx].item()
                pair_logits[(ib, ia)] = logits[idx].item()

    # Sort singleton pairs by logit (descending)
    sorted_pairs = sorted(
        pair_logits.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    # Greedy clique expansion: start from best pair, expand greedily
    best_clique = None
    best_score = -float("inf")

    # Try top-K seed pairs
    for (seed_a, seed_b), seed_logit in sorted_pairs[:10]:
        clique = {seed_a, seed_b}
        clique_score = seed_logit

        # Greedily add singletons that have high logits with ALL clique members
        candidates = [s for s in singleton_indices if s not in clique]
        while candidates:
            best_cand = None
            best_cand_score = -float("inf")

            for cand in candidates:
                # Check min logit with all clique members
                min_logit = float("inf")
                for member in clique:
                    lp = pair_logits.get((cand, member), -999.0)
                    min_logit = min(min_logit, lp)

                if min_logit > best_cand_score:
                    best_cand_score = min_logit
                    best_cand = cand

            # Add if the minimum logit is above threshold
            if best_cand is not None and best_cand_score >= min_avg_logit:
                clique.add(best_cand)
                clique_score += best_cand_score
                candidates = [s for s in singleton_indices if s not in clique]
            else:
                break

        # Evaluate this clique
        if len(clique) >= min_size:
            avg_score = clique_score / (len(clique) * (len(clique) - 1) / 2)
            if avg_score > best_score:
                best_score = avg_score
                best_clique = sorted(clique)

    return best_clique


@torch.no_grad()
def group_greedy_decode(
    model: GNNMergeModel,
    record: ObjectRecord,
    device: torch.device = torch.device("cpu"),
    threshold: float = -1.5,
) -> Any:
    """Group-aware greedy: detect cliques of singletons that should merge
    together (matching GT's k-ary merges like [8, 4, 2, 9]).

    Uses greedy clique detection: finds a set of singletons where ALL
    pairwise logits are above threshold, then merges them as a flat node.

    Args:
        threshold: minimum logit for clique membership. 0.0 = above average.
    """
    model.eval()
    part_feats = torch.tensor(record.feature_matrix(), dtype=torch.float32, device=device)
    edge_index = torch.tensor(record.graph.edge_index, dtype=torch.long, device=device)

    part_embeds = model.encode_parts(part_feats, edge_index)

    state = MergeState(active_clusters=[frozenset([i]) for i in range(record.num_parts)])

    while not state.is_done():
        clusters = state.active_clusters

        if len(clusters) <= 1:
            break

        # Score all pairs
        pairs = []
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                pairs.append((list(sorted(clusters[i])), list(sorted(clusters[j]))))

        if not pairs:
            break

        logits = model.score_pairs_batch(part_embeds, pairs,
                                         [list(sorted(c)) for c in clusters])

        # Try to find a singleton clique for group merge
        clique_indices = _find_best_singleton_clique(
            clusters, pairs, logits,
            min_size=3,
            min_avg_logit=threshold,
        )

        if clique_indices is not None:
            group = [clusters[i] for i in clique_indices]
            state = state.merge_group(group)
            continue

        # Fallback: standard binary merge (best pair)
        probs = torch.sigmoid(logits)
        best_idx = probs.argmax().item()
        best_prob = probs[best_idx].item()

        ca_list, cb_list = pairs[best_idx]
        ca = frozenset(ca_list)
        cb = frozenset(cb_list)
        log_prob = math.log(max(best_prob, 1e-10))
        state = state.merge(ca, cb, log_prob)

    return merge_history_to_tree(state.merge_history, record.num_parts)


# ---------------------------------------------------------------------------
# Beam search decoder
# ---------------------------------------------------------------------------

@torch.no_grad()
def beam_search_decode(
    model: GNNMergeModel,
    record: ObjectRecord,
    beam_width: int = 10,
    device: torch.device = torch.device("cpu"),
) -> Any:
    """Beam search decoding: maintain top-K candidate states at each step.

    Returns the highest-scoring assembly tree in nested list format.
    """
    model.eval()
    part_feats = torch.tensor(record.feature_matrix(), dtype=torch.float32, device=device)
    edge_index = torch.tensor(record.graph.edge_index, dtype=torch.long, device=device)

    part_embeds = model.encode_parts(part_feats, edge_index)

    initial = MergeState(active_clusters=[frozenset([i]) for i in range(record.num_parts)])
    beam: List[MergeState] = [initial]

    for step in range(record.num_parts - 1):
        candidates: List[MergeState] = []

        for state in beam:
            if state.is_done():
                candidates.append(state)
                continue

            clusters = state.active_clusters
            pairs = []
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    pairs.append((list(sorted(clusters[i])), list(sorted(clusters[j]))))

            if not pairs:
                candidates.append(state)
                continue

            logits = model.score_pairs_batch(
                part_embeds, pairs,
                [list(sorted(c)) for c in clusters],
            )
            log_probs = torch.log(torch.sigmoid(logits) + 1e-10)

            for idx in range(len(pairs)):
                ca_list, cb_list = pairs[idx]
                ca = frozenset(ca_list)
                cb = frozenset(cb_list)
                new_state = state.merge(ca, cb, log_probs[idx].item())
                candidates.append(new_state)

        # Keep top beam_width candidates by score
        candidates.sort(key=lambda s: s.score, reverse=True)
        beam = candidates[:beam_width]

    # Return the best completed state
    best = max(beam, key=lambda s: s.score)
    return merge_history_to_tree(best.merge_history, record.num_parts)


# ---------------------------------------------------------------------------
# Diverse beam search (bonus: explores multiple strategies)
# ---------------------------------------------------------------------------

@torch.no_grad()
def diverse_beam_decode(
    model: GNNMergeModel,
    record: ObjectRecord,
    beam_width: int = 10,
    diversity_penalty: float = 0.5,
    device: torch.device = torch.device("cpu"),
) -> Any:
    """Diverse beam search: penalize beams that merge the same pair.

    Returns the highest-scoring assembly tree.
    """
    model.eval()
    part_feats = torch.tensor(record.feature_matrix(), dtype=torch.float32, device=device)
    edge_index = torch.tensor(record.graph.edge_index, dtype=torch.long, device=device)

    part_embeds = model.encode_parts(part_feats, edge_index)

    initial = MergeState(active_clusters=[frozenset([i]) for i in range(record.num_parts)])
    beam: List[MergeState] = [initial]

    for step in range(record.num_parts - 1):
        candidates: List[MergeState] = []

        for state in beam:
            if state.is_done():
                candidates.append(state)
                continue

            clusters = state.active_clusters
            pairs = []
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    pairs.append((list(sorted(clusters[i])), list(sorted(clusters[j]))))

            if not pairs:
                candidates.append(state)
                continue

            logits = model.score_pairs_batch(
                part_embeds, pairs,
                [list(sorted(c)) for c in clusters],
            )
            log_probs = torch.log(torch.sigmoid(logits) + 1e-10)

            for idx in range(len(pairs)):
                ca_list, cb_list = pairs[idx]
                ca = frozenset(ca_list)
                cb = frozenset(cb_list)
                new_state = state.merge(ca, cb, log_probs[idx].item())
                candidates.append(new_state)

        # Diverse selection: greedily pick top candidates, penalizing
        # merges that overlap with already-selected ones
        candidates.sort(key=lambda s: s.score, reverse=True)
        selected: List[MergeState] = []
        selected_merges: set = set()

        for cand in candidates:
            if len(selected) >= beam_width:
                break
            # Check if this candidate's last merge overlaps with selected
            if cand.merge_history:
                last_merge = cand.merge_history[-1]
                merge_key = (frozenset(last_merge[0]), frozenset(last_merge[1]))
                if merge_key in selected_merges and len(selected) > 0:
                    # Apply diversity penalty
                    cand.score -= diversity_penalty
            selected.append(cand)
            if cand.merge_history:
                last_merge = cand.merge_history[-1]
                selected_merges.add((frozenset(last_merge[0]), frozenset(last_merge[1])))

        beam = selected

    best = max(beam, key=lambda s: s.score)
    return merge_history_to_tree(best.merge_history, record.num_parts)


# ---------------------------------------------------------------------------
# Sampling decoder (for GRPO)
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_decode(
    model: GNNMergeModel,
    record: ObjectRecord,
    temperature: float = 1.0,
    device: torch.device = torch.device("cpu"),
    part_embeds: Optional[torch.Tensor] = None,
) -> Tuple[Any, float]:
    """Sample a tree from the model's distribution (for GRPO training).

    Returns (assembly_tree, total_log_prob).
    """
    model.eval()
    if part_embeds is None:
        part_feats = torch.tensor(record.feature_matrix(), dtype=torch.float32, device=device)
        edge_index = torch.tensor(record.graph.edge_index, dtype=torch.long, device=device)
        part_embeds = model.encode_parts(part_feats, edge_index)

    state = MergeState(active_clusters=[frozenset([i]) for i in range(record.num_parts)])
    total_log_prob = 0.0

    while not state.is_done():
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

        # Temperature scaling
        scaled_logits = logits / temperature
        probs = torch.softmax(scaled_logits, dim=0)

        # Sample
        idx = torch.multinomial(probs, 1).item()
        total_log_prob += torch.log(probs[idx] + 1e-10).item()

        ca_list, cb_list = pairs[idx]
        ca = frozenset(ca_list)
        cb = frozenset(cb_list)
        state = state.merge(ca, cb, 0.0)  # log_prob already tracked

    tree = merge_history_to_tree(state.merge_history, record.num_parts)
    return tree, total_log_prob
