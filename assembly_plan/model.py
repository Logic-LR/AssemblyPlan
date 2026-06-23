"""GNN encoder + MLP merge scorer for assembly plan generation.

Uses torch_geometric for proper GraphSAGE message passing.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


# ---------------------------------------------------------------------------
# GraphSAGE encoder using torch_geometric
# ---------------------------------------------------------------------------

class GNNEncoder(nn.Module):
    """Multi-layer GraphSAGE encoder.

    Input: per-part features [N, in_dim] + edge_index [2, E]
    Output: per-part embeddings [N, hidden_dim]
    """

    def __init__(self, in_dim: int, hidden_dim: int = 128, num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
            self.dropouts.append(nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, in_dim] raw part features
            edge_index: [2, E] COO edge index
        Returns:
            [N, hidden_dim] part embeddings
        """
        h = self.input_proj(x)
        for conv, norm, drop in zip(self.convs, self.norms, self.dropouts):
            h = h + drop(F.relu(norm(conv(h, edge_index))))  # residual
        return h


# ---------------------------------------------------------------------------
# Merge Scorer MLP
# ---------------------------------------------------------------------------

class MergeScorer(nn.Module):
    """MLP that scores whether two clusters should merge."""

    def __init__(self, input_dim: int, hidden_dim: int = 192, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, input_dim] -> [B] logits."""
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Combined model
# ---------------------------------------------------------------------------

class GNNMergeModel(nn.Module):
    """Full model: GNN encoder + MLP merge scorer.

    1. GNN encodes part features → part embeddings
    2. Cluster embeddings aggregated from part embeddings
    3. Pair features constructed from cluster embeddings
    4. MLP scores each pair
    """

    def __init__(self, part_feat_dim: int, gnn_hidden: int = 128,
                 gnn_layers: int = 2, scorer_hidden: int = 192,
                 include_context: bool = True, dropout: float = 0.15):
        super().__init__()
        self.gnn = GNNEncoder(part_feat_dim, gnn_hidden, gnn_layers, dropout)
        self.include_context = include_context

        cluster_dim = gnn_hidden * 3 + 2  # mean + max + min + size + log_size
        pair_dim = cluster_dim * 5         # repr_a, repr_b, |a-b|, a*b, repr_union
        if include_context:
            ctx_dim = 7 + cluster_dim * 2  # 7 stats + repr_mean + repr_std
            pair_dim += ctx_dim

        self.scorer = MergeScorer(pair_dim, scorer_hidden, dropout)
        self._cluster_dim = cluster_dim

    @property
    def cluster_dim(self) -> int:
        return self._cluster_dim

    def encode_parts(self, part_feats: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Encode parts through GNN. Returns [N, gnn_hidden]."""
        return self.gnn(part_feats, edge_index)

    def cluster_embed(self, part_embeds: torch.Tensor, cluster: list) -> torch.Tensor:
        """Aggregate part embeddings into a cluster embedding."""
        if len(cluster) == 0:
            return torch.zeros(self._cluster_dim, device=part_embeds.device)
        feats = part_embeds[cluster]
        mean = feats.mean(dim=0)
        mx = feats.max(dim=0).values
        mn = feats.min(dim=0).values
        size = torch.tensor([len(cluster), math.log1p(len(cluster))],
                            device=part_embeds.device, dtype=part_embeds.dtype)
        return torch.cat([mean, mx, mn, size])

    def score_pair(
        self,
        part_embeds: torch.Tensor,
        cluster_a: list,
        cluster_b: list,
        all_clusters: Optional[list] = None,
    ) -> torch.Tensor:
        """Score a single pair. Returns scalar logit."""
        ra = self.cluster_embed(part_embeds, cluster_a)
        rb = self.cluster_embed(part_embeds, cluster_b)
        union = sorted(set(cluster_a) | set(cluster_b))
        ru = self.cluster_embed(part_embeds, union)

        feat = torch.cat([ra, rb, torch.abs(ra - rb), ra * rb, ru])
        if self.include_context and all_clusters is not None:
            ctx = self._global_context(part_embeds, all_clusters)
            feat = torch.cat([feat, ctx])

        return self.scorer(feat.unsqueeze(0)).squeeze(0)

    def score_pairs_batch(
        self,
        part_embeds: torch.Tensor,
        pairs: list,
        all_clusters: Optional[list] = None,
    ) -> torch.Tensor:
        """Score multiple pairs. Returns [len(pairs)] logits."""
        if not pairs:
            return torch.tensor([], device=part_embeds.device)

        ctx = None
        if self.include_context and all_clusters is not None:
            ctx = self._global_context(part_embeds, all_clusters)

        feats = []
        for ca, cb in pairs:
            ra = self.cluster_embed(part_embeds, ca)
            rb = self.cluster_embed(part_embeds, cb)
            union = sorted(set(ca) | set(cb))
            ru = self.cluster_embed(part_embeds, union)
            feat = torch.cat([ra, rb, torch.abs(ra - rb), ra * rb, ru])
            if ctx is not None:
                feat = torch.cat([feat, ctx])
            feats.append(feat)

        return self.scorer(torch.stack(feats))

    def _global_context(self, part_embeds: torch.Tensor, all_clusters: list) -> torch.Tensor:
        """Compute global context from active clusters."""
        sizes = torch.tensor([len(c) for c in all_clusters], dtype=part_embeds.dtype,
                             device=part_embeds.device)
        n = len(all_clusters)
        reprs = torch.stack([self.cluster_embed(part_embeds, c) for c in all_clusters])
        repr_mean = reprs.mean(dim=0)
        repr_std = reprs.std(dim=0)
        stats = torch.tensor([
            n, math.log1p(n), sizes.sum().item(),
            sizes.mean().item(), sizes.std().item() if n > 1 else 0.0,
            sizes.max().item(), sizes.min().item(),
        ], dtype=part_embeds.dtype, device=part_embeds.device)
        return torch.cat([stats, repr_mean, repr_std])


def build_model(part_feat_dim: int = 38, gnn_hidden: int = 128,
                gnn_layers: int = 2, scorer_hidden: int = 192,
                include_context: bool = True, dropout: float = 0.15) -> GNNMergeModel:
    return GNNMergeModel(
        part_feat_dim=part_feat_dim,
        gnn_hidden=gnn_hidden,
        gnn_layers=gnn_layers,
        scorer_hidden=scorer_hidden,
        include_context=include_context,
        dropout=dropout,
    )
