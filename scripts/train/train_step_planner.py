#!/usr/bin/env python3
"""Step-conditioned sequential assembly tree planner.

Training: 393 manual steps, each with explicit connection labels from the SVG.
Inference: GRU tracks state, predicts connections → CC grouping → merge.
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
    step_tree_from_child_specs,
)
from train.train_tree_planner_baseline import (
    Cluster, cluster_token, connected_components, split_records,
)
from train.train_tree_grpo import (
    _load_step_svg_data, _part_to_svg_color, _parse_part_set,
)

# ─── CLI ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="experiments/svg_assembly/datasets/tree_generation_dataset.json")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="experiments/svg_assembly/reports/step_planner_report.json")
    p.add_argument("--model-output", default="experiments/svg_assembly/reports/step_planner_model.pt")
    return p.parse_args()


# ─── Per-Part Features ──────────────────────────────────────────────────────

def _part_features(record: dict, step_svg_data: dict | None = None) -> Dict[int, np.ndarray]:
    """Per-part: shape_idx(1) + svg_geometry(9) = 10 dims."""
    feats: Dict[int, np.ndarray] = {}
    has_svg = step_svg_data is not None and len(step_svg_data) > 0
    part_color = _part_to_svg_color(record) if has_svg else {}
    for token in record.get("part_tokens") or []:
        pid = int(token["part_id"])
        shape_idx = float(np.argmax(token.get("shape_distribution") or [0.25]*4))
        svg_feat = np.zeros(8, dtype=np.float32)
        if has_svg:
            color = part_color.get(pid)
            if color:
                for instances in step_svg_data.values():
                    for inst in instances:
                        if str(inst.get("id", "")).lower() == color:
                            c = np.asarray(inst.get("center", [0,0]), dtype=np.float32)/200.0
                            ax = np.asarray(inst.get("principal_axis", [[0,0],[1,0]]), dtype=np.float32)
                            axv = ax[1]-ax[0]; an = float(np.linalg.norm(axv)) or 1.0; axv = axv/an
                            svg_feat = np.concatenate([
                                c, axv,
                                [float(inst.get("axis_length",100))/200.0,
                                 float(inst.get("axis_width",50))/200.0,
                                 float(inst.get("elongation",1.0))/4.0,
                                 float(np.linalg.norm(np.asarray(inst.get("bbox",[0,0,1,1]),dtype=np.float32)[2:]-np.asarray(inst.get("bbox",[0,0,1,1]),dtype=np.float32)[:2]))/400.0]
                            ]).astype(np.float32)
                            break
        feats[pid] = np.concatenate([[shape_idx], svg_feat]).astype(np.float32)
    return feats


# ─── Model ──────────────────────────────────────────────────────────────────

class StepPlanner(nn.Module):
    """Sequential connection predictor: step_emb + GRU → which clusters connect."""

    def __init__(self, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.shape_emb = nn.Embedding(4, 16)
        self.spatial_proj = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 16))
        self.cluster_attn = nn.MultiheadAttention(32, 4, dropout=dropout, batch_first=True)
        self.cluster_proj = nn.Linear(32, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.step_emb = nn.Embedding(20, 32)
        # Pair scorer: [emb_i, emb_j, gru_h, step_emb]
        self.pair_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 32, hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _encode_parts(self, part_feats: dict) -> torch.Tensor:
        device = next(self.parameters()).device
        embs = []
        for pid in sorted(part_feats):
            f = part_feats[pid]; s = torch.tensor(int(f[0]), device=device)
            sp = torch.from_numpy(f[1:]).float().to(device)
            embs.append(torch.cat([self.shape_emb(s), self.spatial_proj(sp.unsqueeze(0)).squeeze(0)]))
        return torch.stack(embs)

    def _encode_clusters(self, part_embs: torch.Tensor, clusters: List[Cluster]) -> torch.Tensor:
        embs = []
        for c in clusters:
            idx = torch.tensor(sorted(c), device=part_embs.device)
            ce = part_embs[idx]
            if len(c) == 1: pooled = ce.squeeze(0)
            else: attn_out, _ = self.cluster_attn(ce.unsqueeze(0), ce.unsqueeze(0), ce.unsqueeze(0)); pooled = attn_out.squeeze(0).mean(0)
            embs.append(self.cluster_proj(pooled))
        return torch.stack(embs)

    def forward(self, clusters: List[Cluster], part_feats: dict,
                gru_hidden: torch.Tensor | None, step_idx: int = 0):
        """Returns (pair_scores [P], new_hidden)."""
        device = next(self.parameters()).device
        part_embs = self._encode_parts(part_feats)
        cluster_embs = self._encode_clusters(part_embs, clusters)
        gru_input = cluster_embs.mean(0, keepdim=True)
        if gru_hidden is None: gru_hidden = torch.zeros(1, 1, self.hidden_dim, device=device)
        else: gru_hidden = gru_hidden.unsqueeze(0).unsqueeze(0)
        _, nh = self.gru(gru_input.unsqueeze(0), gru_hidden)
        h = nh.squeeze(0).squeeze(0)
        se = self.step_emb(torch.tensor(min(step_idx, 19), device=device))
        M = len(clusters)
        pf = []
        for i in range(M):
            for j in range(i+1, M):
                pf.append(torch.cat([cluster_embs[i], cluster_embs[j], h, se]))
        if not pf: return torch.zeros(1, device=device), h
        return self.pair_scorer(torch.stack(pf)).squeeze(-1), h


# ─── Training: manual_step_groups.connections as labels ──────────────────────

def _train_one_epoch(model, optimizer, records, device, args):
    """Train on all manual steps, using connections as pair labels."""
    model.train()
    total_loss, total_steps = 0.0, 0

    for rec in records:
        part_feats = _part_features(rec, _load_step_svg_data(rec))
        num_parts = int(rec["num_parts"])
        clusters_state: Dict[int, Cluster] = {p: frozenset([p]) for p in range(num_parts)}
        manual_steps = sorted(rec.get("manual_step_groups") or [], key=lambda s: s["step_id"])
        gru_hidden = None

        for si, sg in enumerate(manual_steps):
            # Parse step state: which clusters exist at this step?
            step_parts: List[Cluster] = []
            for p_str in sg["parts"]:
                ps = _parse_part_set(p_str)
                # Find the current cluster matching this part set
                matched = None
                part_set = set(ps)
                for k, c in clusters_state.items():
                    if set(c) == part_set:
                        matched = c; break
                if matched is None:
                    # Parts are singletons that never appeared before
                    matched = frozenset(ps)
                step_parts.append(matched)

            if len(step_parts) < 2: continue

            # Parse connections as positive pairs
            conn_set: Set[Tuple[Cluster, Cluster]] = set()
            for conn in sg["connections"]:
                a = _parse_part_set(conn[0]); b = _parse_part_set(conn[1])
                ca = frozenset(a) if frozenset(a) in set(step_parts) else None
                cb = frozenset(b) if frozenset(b) in set(step_parts) else None
                if ca is None or cb is None: continue
                key = (ca, cb) if tuple(sorted(ca)) <= tuple(sorted(cb)) else (cb, ca)
                conn_set.add(key)

            clusters_list = sorted(set(step_parts), key=lambda c: (len(c), tuple(sorted(c))))
            scores, new_hidden = model(clusters_list, part_feats, gru_hidden, step_idx=si)
            gru_hidden = new_hidden.detach()

            # Labels
            M = len(clusters_list)
            labels = torch.zeros(M * (M - 1) // 2, device=device)
            idx = 0
            for i in range(M):
                for j in range(i + 1, M):
                    key = (clusters_list[i], clusters_list[j]) if tuple(sorted(clusters_list[i])) <= tuple(sorted(clusters_list[j])) else (clusters_list[j], clusters_list[i])
                    if key in conn_set:
                        labels[idx] = 1.0
                    idx += 1

            n_pos = labels.sum().clamp_min(1)
            pos_weight = (len(labels) - n_pos) / n_pos
            weights = torch.where(labels > 0.5, pos_weight, 1.0)
            loss = nn.functional.binary_cross_entropy_with_logits(scores, labels, weight=weights)
            (loss / max(len(manual_steps), 1)).backward()
            total_loss += float(loss.item())
            total_steps += 1

            # Merge connected clusters (teacher forcing)
            if conn_set:
                edges = [(a, b) for a, b in conn_set]
                comps = connected_components(clusters_list, edges)
                for comp in comps:
                    if len(comp) >= 2:
                        parent = frozenset().union(*comp)
                        for c in comp:
                            # Update clusters_state
                            for k, v in list(clusters_state.items()):
                                if v == c: del clusters_state[k]
                        clusters_state[max(parent)] = parent

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return total_loss / max(total_steps, 1)


# ─── Inference ──────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_tree(record, model, device, threshold: float):
    """Predict connections at each step → CC group → merge."""
    num_parts = int(record["num_parts"])
    part_feats = _part_features(record, _load_step_svg_data(record))
    current = {frozenset([p]) for p in range(num_parts)}
    merges = []
    gru_hidden = None

    for step_idx in range(num_parts):
        if len(current) <= 1: break
        clusters = sorted(current, key=lambda c: (len(c), tuple(sorted(c))))
        scores, new_hidden = model(clusters, part_feats, gru_hidden, step_idx=step_idx)
        probs = torch.sigmoid(scores).cpu().numpy()
        gru_hidden = new_hidden.detach()

        M = len(clusters)
        edges = []
        idx = 0
        for i in range(M):
            for j in range(i + 1, M):
                if probs[idx] >= threshold:
                    edges.append((clusters[i], clusters[j]))
                idx += 1

        if not edges:
            best_idx = int(np.argmax(probs))
            idx = 0
            for i in range(M):
                for j in range(i + 1, M):
                    if idx == best_idx: pi, pj = i, j; break
                    idx += 1
            edges.append((clusters[pi], clusters[pj]))

        comps = connected_components(clusters, edges)
        for comp in comps:
            if len(comp) >= 2:
                parent = frozenset().union(*comp)
                merges.append([cluster_token(c) for c in sorted(comp, key=lambda x: (len(x), tuple(sorted(x))))])
                for c in comp: current.discard(c)
                current.add(parent)

    return step_tree_from_child_specs(merges, num_parts)


@torch.no_grad()
def tune_threshold(model, records, device):
    best_t, best_s = 0.5, -1.0
    for t in np.linspace(0.15, 0.90, 16):
        rows = []
        for rec in records:
            gt = build_tree_from_list(rec["assembly_tree"])
            pred = predict_tree(rec, model, device, float(t))
            rows.append(eval_tree(gt, pred))
        s = average_metrics(rows)["hard"]["f1"]
        if s > best_s: best_s = s; best_t = float(t)
    return best_t


@torch.no_grad()
def evaluate(model, records, device, threshold: float):
    rows = []
    for rec in records:
        gt = build_tree_from_list(rec["assembly_tree"])
        pred = predict_tree(rec, model, device, threshold)
        rows.append(eval_tree(gt, pred))
    return average_metrics(rows)


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    fit, val, test = split_records(records, args.val_fraction, args.seed, val_seed=args.split_seed)

    model = StepPlanner(hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val, best_state = -1.0, None

    for epoch in range(1, args.epochs + 1):
        train_loss = _train_one_epoch(model, optimizer, fit, device, args)
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            t = tune_threshold(model, val or fit, device)
            val_m = evaluate(model, val or fit, device, t)
            vh = val_m["hard"]["f1"]
            if vh > best_val:
                best_val = vh
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"epoch {epoch:3d}  loss={train_loss:.4f}  val_hard={vh:.4f}  best={best_val:.4f}", flush=True)

    if best_state: model.load_state_dict(best_state)
    model.eval()
    t = tune_threshold(model, val or fit, device)
    val_m = evaluate(model, val or fit, device, t)
    test_m = evaluate(model, test, device, t)
    all_m = evaluate(model, records, device, t)

    report = {"model": "step_planner_v2",
              "splits": {"fit": len(fit), "val": len(val), "test": len(test)},
              "threshold": t,
              "tree_metrics": {"val": val_m, "test": test_m, "all": all_m},
              "config": vars(args)}
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    torch.save({"model_state": model.cpu().state_dict(), "config": vars(args)}, args.model_output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
