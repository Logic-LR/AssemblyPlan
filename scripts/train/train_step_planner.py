#!/usr/bin/env python3
"""Step-conditioned sequential assembly tree planner.

Unlike the flat pair scorer, this model:
  - Processes merge steps sequentially (not pooled together)
  - Uses per-step simplified SVG spatial context during training
  - Tracks progress with a GRU hidden state
  - At inference, samples multiple trees and selects the best via SVG reward
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse, json, math, random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

import numpy as np
import torch
from torch import nn

from eval.evaluate_paper_tree_metrics import (
    average_metrics, build_tree_from_list, eval_tree, nonleaf_nodes,
    step_tree_from_child_specs, Node,
)
from train.train_tree_planner_baseline import (
    Cluster, cluster_token, split_records,
)
from train.train_tree_grpo import (
    _load_step_svg_data, _part_to_svg_color, spatial_svg_reward, _parse_part_set,
)


# ─── CLI ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="experiments/svg_assembly/datasets/tree_generation_dataset.json")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="experiments/svg_assembly/reports/step_planner_report.json")
    p.add_argument("--model-output", default="experiments/svg_assembly/reports/step_planner_model.pt")
    p.add_argument("--pred-output-dir", default="experiments/svg_assembly/step_planner_predictions_test")
    return p.parse_args()


# ─── Feature Extraction ─────────────────────────────────────────────────────

SHAPE_TYPES = ["elongated_bar", "plate_like", "irregular", "point_or_line"]


def _part_shape_idx(token: dict) -> int:
    dist = token.get("shape_distribution") or [0.25] * 4
    return int(np.argmax(dist))


def _svg_instance_features(inst: dict) -> np.ndarray:
    """Spatial features from one simplified SVG instance. Returns 9-dim."""
    center = np.asarray(inst.get("center", [0, 0]), dtype=np.float32)
    axis = np.asarray(inst.get("principal_axis", [[0, 0], [1, 0]]), dtype=np.float32)
    axis_vec = axis[1] - axis[0]
    axis_len = float(np.linalg.norm(axis_vec)) or 1.0
    axis_vec = axis_vec / axis_len
    bbox = np.asarray(inst.get("bbox", [0, 0, 1, 1]), dtype=np.float32)
    return np.concatenate([
        center / 200.0,
        axis_vec,
        np.asarray([float(inst.get("axis_length", 100)) / 200.0], dtype=np.float32),
        np.asarray([float(inst.get("axis_width", 50)) / 200.0], dtype=np.float32),
        np.asarray([float(inst.get("elongation", 1.0)) / 4.0], dtype=np.float32),
        (bbox[2:] - bbox[:2]) / 400.0,
    ]).astype(np.float32)


def _part_features(record: dict, step_svg_data: dict | None = None) -> Dict[int, np.ndarray]:
    """Build per-part features: shape_idx(1) + svg_spatial(9) = 10 dims."""
    feats: Dict[int, np.ndarray] = {}
    has_svg = step_svg_data is not None and len(step_svg_data) > 0
    part_color = _part_to_svg_color(record) if has_svg else {}
    for token in record.get("part_tokens") or []:
        pid = int(token["part_id"])
        shape_idx = float(_part_shape_idx(token))
        if has_svg:
            color = part_color.get(pid)
            svg_feat = np.zeros(9, dtype=np.float32)
            if color:
                for instances in step_svg_data.values():
                    for inst in instances:
                        if str(inst.get("id", "")).lower() == color:
                            svg_feat = _svg_instance_features(inst)
                            break
            feats[pid] = np.concatenate([
                np.asarray([shape_idx], dtype=np.float32), svg_feat
            ]).astype(np.float32)
        else:
            feats[pid] = np.asarray([shape_idx] + [0.0] * 9, dtype=np.float32)
    return feats


def _pair_spatial(a_inst: dict | None, b_inst: dict | None) -> np.ndarray:
    """Spatial relationship between two SVG instances. Returns 5-dim."""
    if a_inst is None or b_inst is None:
        return np.zeros(5, dtype=np.float32)
    ca = np.asarray(a_inst.get("center", [0, 0]), dtype=np.float32)
    cb = np.asarray(b_inst.get("center", [0, 0]), dtype=np.float32)
    delta = cb - ca
    dist = float(np.linalg.norm(delta))
    axis_a = np.asarray(a_inst.get("principal_axis", [[0, 0], [1, 0]]), dtype=np.float32)
    axis_b = np.asarray(b_inst.get("principal_axis", [[0, 0], [1, 0]]), dtype=np.float32)
    dir_a = axis_a[1] - axis_a[0]; dir_b = axis_b[1] - axis_b[0]
    na = float(np.linalg.norm(dir_a)) or 1.0; nb = float(np.linalg.norm(dir_b)) or 1.0
    alignment = float(abs(np.dot(dir_a / na, dir_b / nb)))
    return np.asarray([
        delta[0] / 200.0, delta[1] / 200.0, dist / 200.0,
        alignment, 1.0 if dist < 80 else 0.0,  # proximity
    ], dtype=np.float32)


# ─── Model ──────────────────────────────────────────────────────────────────

class StepMergePlanner(nn.Module):
    """Sequential merge planner: GRU state + step SVG spatial context."""

    def __init__(self, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.shape_emb = nn.Embedding(4, 16)
        self.spatial_proj = nn.Sequential(
            nn.Linear(9, 32), nn.ReLU(), nn.Linear(32, 16)
        )
        # Cluster: attention pool parts → hidden_dim
        self.cluster_attn = nn.MultiheadAttention(32, 4, dropout=dropout, batch_first=True)
        self.cluster_proj = nn.Linear(32, hidden_dim)
        # GRU
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        # Pair scorer: [emb_i, emb_j, GRU_h, spatial_ij]
        self.pair_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 5, hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _encode_parts(self, part_feats: dict) -> torch.Tensor:
        device = next(self.parameters()).device
        embs = []
        for pid in sorted(part_feats):
            f = part_feats[pid]
            spatial = torch.from_numpy(f[1:]).float().to(device)
            shape_emb = self.shape_emb(torch.tensor(int(f[0]), device=device))
            spatial_emb = self.spatial_proj(spatial.unsqueeze(0)).squeeze(0)
            embs.append(torch.cat([shape_emb, spatial_emb]))
        return torch.stack(embs)  # [P, 32]

    def _encode_clusters(self, part_embs: torch.Tensor,
                         clusters: List[Cluster]) -> torch.Tensor:
        cluster_embs = []
        for c in clusters:
            idx = torch.tensor(sorted(c), device=part_embs.device)
            c_embs = part_embs[idx]  # [|c|, 32]
            if len(c) == 1:
                pooled = c_embs.squeeze(0)
            else:
                attn_out, _ = self.cluster_attn(c_embs.unsqueeze(0), c_embs.unsqueeze(0), c_embs.unsqueeze(0))
                pooled = attn_out.squeeze(0).mean(0)
            cluster_embs.append(self.cluster_proj(pooled))
        return torch.stack(cluster_embs)  # [M, hidden_dim]

    def forward(self, clusters: List[Cluster], part_feats: dict,
                gru_hidden: torch.Tensor | None,
                pair_spatial_map: Dict[Tuple[int, int], np.ndarray] | None = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (scores [P], new_hidden [hidden_dim])."""
        device = next(self.parameters()).device
        part_embs = self._encode_parts(part_feats)
        cluster_embs = self._encode_clusters(part_embs, clusters)

        # GRU step
        gru_input = cluster_embs.mean(0, keepdim=True)
        if gru_hidden is None:
            gru_hidden = torch.zeros(1, 1, self.hidden_dim, device=device)
        else:
            gru_hidden = gru_hidden.unsqueeze(0).unsqueeze(0)
        _, new_hidden = self.gru(gru_input.unsqueeze(0), gru_hidden)
        h = new_hidden.squeeze(0).squeeze(0)

        # Score pairs
        M = len(clusters)
        pair_feats = []
        spatial_map = pair_spatial_map or {}
        for i in range(M):
            for j in range(i + 1, M):
                sp = torch.from_numpy(spatial_map.get((i, j), np.zeros(5, dtype=np.float32))).float().to(device)
                pair_feats.append(torch.cat([cluster_embs[i], cluster_embs[j], h, sp]))

        if not pair_feats:
            return torch.zeros(1, device=device), new_hidden.squeeze(0).squeeze(0)

        scores = self.pair_scorer(torch.stack(pair_feats)).squeeze(-1)
        return scores, new_hidden.squeeze(0).squeeze(0)


# ─── Training ────────────────────────────────────────────────────────────────

def _build_step_spatial_map(record: dict, clusters: List[Cluster],
                            step_svg_data: dict, step_id: int
                            ) -> Dict[Tuple[int, int], np.ndarray]:
    """Build per-pair spatial features from a specific step's SVG instances."""
    if step_id not in step_svg_data:
        return {}
    part_color = _part_to_svg_color(record)
    instances = step_svg_data[step_id]
    # Map cluster index → first matching SVG instance
    cluster_inst: Dict[int, dict | None] = {}
    for idx, c in enumerate(clusters):
        found = None
        for p in c:
            color = part_color.get(p)
            if color:
                for inst in instances:
                    if str(inst.get("id", "")).lower() == color:
                        found = inst; break
            if found: break
        cluster_inst[idx] = found

    spatial_map: Dict[Tuple[int, int], np.ndarray] = {}
    M = len(clusters)
    for i in range(M):
        for j in range(i + 1, M):
            spatial_map[(i, j)] = _pair_spatial(cluster_inst.get(i), cluster_inst.get(j))
    return spatial_map


def _find_matching_step(record: dict, parent_parts: frozenset) -> int:
    """Find which manual step first contains parent as an exact part group."""
    for sg in record.get("manual_step_groups") or []:
        for p_str in sg.get("parts") or []:
            if _parse_part_set(p_str) == parent_parts:
                return sg["step_id"]
    # Fallback: find first step where ALL parts of parent appear together
    parent_set = set(parent_parts)
    for sg in record.get("manual_step_groups") or []:
        all_parts_in_step: Set[int] = set()
        for p_str in sg.get("parts") or []:
            all_parts_in_step.update(_parse_part_set(p_str))
        if parent_set.issubset(all_parts_in_step):
            return sg["step_id"]
    return 0


def _get_tree_actions(tree: Node) -> list:
    """Postorder tree actions."""
    actions = []
    def _walk(n):
        for c in n.children: _walk(c)
        if n.children:
            actions.append({"parent": n.parts, "children": [c.parts for c in n.children]})
    _walk(tree)
    return actions


def _train_one_epoch(model, optimizer, records, step_svg_cache, device, args):
    """Teacher-forcing through GT merge actions."""
    model.train()
    total_loss, total_steps = 0.0, 0

    for rec in records:
        tree = build_tree_from_list(rec["assembly_tree"])
        num_parts = int(rec["num_parts"])
        cache_key = f"{rec['category']}/{rec['name']}"
        step_data = step_svg_cache.get(cache_key) or _load_step_svg_data(rec)
        step_svg_cache[cache_key] = step_data
        part_feats = _part_features(rec, step_data)
        current = {frozenset([p]) for p in range(num_parts)}
        gru_hidden = None

        obj_loss = 0.0
        for action in _get_tree_actions(tree):
            children = [frozenset(c) for c in action["children"]]
            parent = frozenset(action["parent"])
            clusters = sorted(current, key=lambda c: (len(c), tuple(sorted(c))))
            if len(clusters) < 2: break

            step_id = _find_matching_step(rec, parent)
            spatial_map = _build_step_spatial_map(rec, clusters, step_data, step_id) if step_data else {}

            scores, new_hidden = model(clusters, part_feats, gru_hidden, spatial_map)

            # GT: which pair indices are positive?
            children_set = set(children)
            M = len(clusters)
            labels = torch.zeros(M * (M - 1) // 2, device=device)
            idx = 0
            for i in range(M):
                for j in range(i + 1, M):
                    if clusters[i] in children_set and clusters[j] in children_set:
                        labels[idx] = 1.0
                    idx += 1

            n_pos = labels.sum().clamp_min(1)
            pos_weight = (len(labels) - n_pos) / n_pos
            weights = torch.where(labels > 0.5, pos_weight, 1.0)
            loss = nn.functional.binary_cross_entropy_with_logits(scores, labels, weight=weights)
            obj_loss += loss
            total_steps += 1

            gru_hidden = new_hidden.detach()
            for c in children: current.discard(c)
            current.add(parent)

        if total_steps > 0:
            (obj_loss / len(_get_tree_actions(tree))).backward()
            total_loss += float(obj_loss.item())
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return total_loss / max(total_steps, 1)


# ─── Inference ──────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_tree(record, model, device, num_samples=20, temperature=0.6):
    """Sample multiple trees, select best via SVG reward."""
    num_parts = int(record["num_parts"])
    step_data = _load_step_svg_data(record)
    part_feats = _part_features(record, step_data)
    # Build subassemblies for SVG reward
    subassemblies = set()
    for sg in record.get("manual_step_groups") or []:
        for p_str in sg.get("parts") or []:
            ps = _parse_part_set(p_str)
            if len(ps) >= 2: subassemblies.add(ps)

    best_tree, best_reward = None, -1.0
    for _ in range(num_samples):
        current = {frozenset([p]) for p in range(num_parts)}
        merges = []
        gru_hidden = None
        for _ in range(num_parts):
            if len(current) <= 1: break
            clusters = sorted(current, key=lambda c: (len(c), tuple(sorted(c))))
            scores, new_hidden = model(clusters, part_feats, gru_hidden, {})

            # Temperature sampling
            scaled = scores / max(temperature, 1e-6)
            probs = torch.softmax(scaled, dim=0).cpu().numpy()
            chosen = int(np.random.choice(len(probs), p=probs))

            # Decode pair index
            M = len(clusters)
            pi = pj = 0; found = False
            idx = 0
            for i in range(M):
                for j in range(i + 1, M):
                    if idx == chosen: pi, pj = i, j; found = True; break
                    idx += 1
                if found: break

            a, b = clusters[pi], clusters[pj]
            merges.append([cluster_token(a), cluster_token(b)])
            current = {c for k, c in enumerate(clusters) if k != pi and k != pj}
            current.add(a | b)
            gru_hidden = new_hidden.detach()

        tree = step_tree_from_child_specs(merges, num_parts)
        nodes = list(nonleaf_nodes(tree))
        r_svg = sum(1 for n in nodes if n.parts in subassemblies) / max(len(nodes), 1)
        r_spatial = spatial_svg_reward(tree, record)
        total_r = 0.4 * r_svg + 0.6 * r_spatial
        if total_r > best_reward:
            best_reward = total_r; best_tree = tree

    return best_tree


@torch.no_grad()
def evaluate(model, records, device):
    rows = []
    for rec in records:
        gt = build_tree_from_list(rec["assembly_tree"])
        pred = predict_tree(rec, model, device, num_samples=10, temperature=0.5)
        rows.append(eval_tree(gt, pred))
    return average_metrics(rows)


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    fit, val, test = split_records(records, args.val_fraction, args.seed, val_seed=args.split_seed)

    model = StepMergePlanner(hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    step_svg_cache = {}
    best_val, best_state = -1.0, None

    for epoch in range(1, args.epochs + 1):
        train_loss = _train_one_epoch(model, optimizer, fit, step_svg_cache, device, args)
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            val_m = evaluate(model, val or fit, device)
            vh = val_m["hard"]["f1"]
            if vh > best_val:
                best_val = vh
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"epoch {epoch:3d}  loss={train_loss:.4f}  val_hard={vh:.4f}  best={best_val:.4f}", flush=True)

    if best_state: model.load_state_dict(best_state)
    model.eval()
    val_m = evaluate(model, val or fit, device)
    test_m = evaluate(model, test, device)
    all_m = evaluate(model, records, device)

    report = {"model": "step_conditioned_planner",
              "splits": {"fit": len(fit), "val": len(val), "test": len(test)},
              "tree_metrics": {"val": val_m, "test": test_m, "all": all_m},
              "config": vars(args)}
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    torch.save({"model_state": model.cpu().state_dict(), "config": vars(args)}, args.model_output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
