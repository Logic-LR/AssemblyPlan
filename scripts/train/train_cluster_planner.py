#!/usr/bin/env python3
"""Metric-learning tree planner: learn features where connected parts cluster together.

Inference: K-Means (k=2,3) + silhouette → recursive top-down tree (like original paper).
Training: connections from manual_step_groups → contrastive: pull connected pairs close.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse, json, random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from eval.evaluate_paper_tree_metrics import (
    average_metrics, build_tree_from_list, eval_tree, Node,
)
from train.train_tree_planner_baseline import split_records
from train.train_tree_grpo import _part_to_svg_color, _parse_part_set, _load_step_svg_data


# ─── CLI ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="experiments/svg_assembly/datasets/tree_generation_dataset.json")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--emb-dim", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--margin", type=float, default=0.5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="experiments/svg_assembly/reports/cluster_planner_report.json")
    p.add_argument("--model-output", default="experiments/svg_assembly/reports/cluster_planner_model.pt")
    return p.parse_args()


# ─── Per-Part Features ──────────────────────────────────────────────────────

def _part_features(record: dict) -> Dict[int, np.ndarray]:
    """Per-part: shape_idx(1) + svg_geometry(7) = 8 dims."""
    feats: Dict[int, np.ndarray] = {}
    step_data = _load_step_svg_data(record)
    part_color = _part_to_svg_color(record) if step_data else {}
    for token in record.get("part_tokens") or []:
        pid = int(token["part_id"])
        shape_idx = float(np.argmax(token.get("shape_distribution") or [0.25]*4))
        svg_feat = np.zeros(7, dtype=np.float32)
        color = part_color.get(pid)
        if color:
            for instances in step_data.values():
                for inst in instances:
                    if str(inst.get("id", "")).lower() == color:
                        c = np.asarray(inst.get("center", [0,0]), dtype=np.float32)/200.0
                        ax = np.asarray(inst.get("principal_axis", [[0,0],[1,0]]), dtype=np.float32)
                        axv = ax[1]-ax[0]; an = float(np.linalg.norm(axv)) or 1.0; axv = axv/an
                        bbox = np.asarray(inst.get("bbox", [0,0,1,1]), dtype=np.float32)
                        svg_feat = np.concatenate([
                            c, axv,
                            [float(inst.get("axis_length",100))/200.0,
                             float(inst.get("elongation",1.0))/4.0,
                             float(np.linalg.norm(bbox[2:]-bbox[:2]))/400.0]
                        ]).astype(np.float32)
                        break
        feats[pid] = np.concatenate([[shape_idx], svg_feat]).astype(np.float32)
    return feats


# ─── Model: Feature Projector ───────────────────────────────────────────────

class PartProjector(nn.Module):
    """Project part features to embedding space for clustering."""
    def __init__(self, input_dim: int = 8, emb_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─── Training: Contrastive Loss ─────────────────────────────────────────────

def _train_one_epoch(model, optimizer, records, device, args):
    """Contrastive: pull connected pairs together, push unconnected apart."""
    model.train()
    total_loss, total_pairs = 0.0, 0

    for rec in records:
        part_feats = _part_features(rec)
        part_ids = sorted(part_feats.keys())
        if len(part_ids) < 2: continue

        # Build embeddings
        feat_tensor = torch.from_numpy(np.stack([part_feats[p] for p in part_ids])).float().to(device)
        embs = model(feat_tensor)  # [P, emb_dim]

        # Collect all connection pairs across all manual steps
        # For multi-part clusters, use the centroid of their part embeddings
        positive_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for sg in rec.get("manual_step_groups") or []:
            conns = sg.get("connections") or []
            for conn in conns:
                a_parts = _parse_part_set(conn[0])
                b_parts = _parse_part_set(conn[1])
                a_idxs = [part_ids.index(p) for p in a_parts if p in part_ids]
                b_idxs = [part_ids.index(p) for p in b_parts if p in part_ids]
                if not a_idxs or not b_idxs: continue
                a_centroid = embs[torch.tensor(a_idxs, device=device)].mean(0)
                b_centroid = embs[torch.tensor(b_idxs, device=device)].mean(0)
                positive_pairs.append((a_centroid, b_centroid))

        if not positive_pairs: continue

        # Build set of connected (part_i, part_j) for negative sampling
        pos_part_pairs = set()
        for sg in rec.get("manual_step_groups") or []:
            for conn in sg.get("connections") or []:
                a_parts = _parse_part_set(conn[0]); b_parts = _parse_part_set(conn[1])
                for pa in a_parts:
                    for pb in b_parts:
                        if pa in part_ids and pb in part_ids:
                            pi, pj = part_ids.index(pa), part_ids.index(pb)
                            pos_part_pairs.add((min(pi,pj), max(pi,pj)))

        # Sample negative pairs
        all_pairs = [(i, j) for i in range(len(part_ids)) for j in range(i+1, len(part_ids))]
        random.shuffle(all_pairs)
        neg_pairs = [(i,j) for i,j in all_pairs if (i,j) not in pos_part_pairs]
        neg_pairs = neg_pairs[:len(positive_pairs) * 3]
        if not neg_pairs: continue  # all pairs are connected, skip this object

        # Contrastive loss
        pos_dists = torch.stack([torch.norm(a - b) for a, b in positive_pairs])
        neg_dists = torch.stack([torch.norm(embs[i] - embs[j]) for i, j in neg_pairs])

        loss = pos_dists.mean() + torch.clamp(args.margin - neg_dists, min=0).mean()
        loss.backward()
        total_loss += float(loss.item())
        total_pairs += 1

    if total_pairs > 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return total_loss / max(total_pairs, 1)


# ─── Inference: Recursive Clustering (same as original paper) ───────────────

@torch.no_grad()
def _cluster_tree(part_ids: list, embs: np.ndarray, max_k: int = 3) -> Node:
    """Recursive top-down tree via K-Means clustering."""
    n = len(part_ids)
    if n == 0: return None
    if n == 1: return Node.leaf(part_ids[0])

    # Try k=1,2,...,min(max_k, n) and pick best silhouette
    best_score, best_labels = -1, None
    for k in range(1, min(max_k, n) + 1):
        if k == 1:
            best_score = 0.0; best_labels = np.zeros(n, dtype=int); continue
        if k >= n: continue  # silhouette needs k < n
        km = KMeans(n_clusters=k, random_state=0, n_init=5)
        labels = km.fit_predict(embs)
        if len(set(labels)) < 2: continue
        score = silhouette_score(embs, labels, metric='euclidean')
        if score > best_score:
            best_score = score; best_labels = labels

    if best_labels is None or len(set(best_labels)) <= 1:
        return Node.parent([Node.leaf(p) for p in part_ids])

    # Pick best cluster → subassembly, recurse on rest
    cluster_scores = []
    for c in range(len(set(best_labels))):
        idxs = np.where(best_labels == c)[0]
        if len(idxs) <= 1: cluster_scores.append(-1); continue
        ce = embs[idxs]
        dist = np.mean(np.linalg.norm(ce - ce.mean(0), axis=1))
        cluster_scores.append(-dist)

    best_cluster = int(np.argmax(cluster_scores))
    best_idxs = np.where(best_labels == best_cluster)[0]
    other_idxs = np.where(best_labels != best_cluster)[0]

    children = [Node.leaf(part_ids[i]) for i in best_idxs]

    if len(other_idxs) > 0:
        other_node = _cluster_tree([part_ids[i] for i in other_idxs], embs[other_idxs])
        if other_node: children.append(other_node)

    return Node.parent(children)


@torch.no_grad()
def predict_tree(record, model, device):
    part_feats = _part_features(record)
    part_ids = sorted(part_feats.keys())
    feat_tensor = torch.from_numpy(np.stack([part_feats[p] for p in part_ids])).float().to(device)
    embs = model(feat_tensor).cpu().numpy()
    return _cluster_tree(part_ids, embs)


@torch.no_grad()
def evaluate(model, records, device):
    rows = []
    for rec in records:
        gt = build_tree_from_list(rec["assembly_tree"])
        pred = predict_tree(rec, model, device)
        rows.append(eval_tree(gt, pred))
    return average_metrics(rows)


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    fit, val, test = split_records(records, args.val_fraction, args.seed, val_seed=args.split_seed)

    model = PartProjector(input_dim=8, emb_dim=args.emb_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val, best_state = -1.0, None

    for epoch in range(1, args.epochs + 1):
        train_loss = _train_one_epoch(model, optimizer, fit, device, args)
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

    report = {"model": "cluster_planner",
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
