"""Data loading and feature engineering for assembly plan generation.

Loads from tree_generation_dataset.json + simplified_svg, applies log1p
fixes to extreme ratio features, builds connection graphs, and provides
training examples.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_JSON = BASE_DIR / "experiments" / "svg_assembly" / "datasets" / "tree_generation_dataset.json"
SIMPLIFIED_SVG_DIR = BASE_DIR / "experiments" / "svg_assembly" / "simplified_svg"

# geometry_feature dims that are extreme ratios (ext0/ext2, sorted0/sorted2)
RATIO_LOG1P_DIMS = [7, 8]
# svg_feature_mean dim 16 is always 0 — dead feature
SVG_DROP_DIMS = [16]

SHAPE_TYPES = ["elongated_bar", "plate_like", "irregular", "point_or_line"]

# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------

def fix_geometry_feature(geom: List[float]) -> np.ndarray:
    """Apply log1p to extreme ratio dims, return 13-dim vector (drop nothing)."""
    g = np.array(geom, dtype=np.float32)
    for d in RATIO_LOG1P_DIMS:
        g[d] = math.log1p(max(0.0, g[d]))
    return g  # still 14-dim


def fix_svg_feature(svg_mean: List[float], svg_count: int) -> np.ndarray:
    """Drop dead dim 16, return 16-dim vector."""
    s = np.array(svg_mean, dtype=np.float32)
    # Remove dead dimensions
    keep = [i for i in range(len(s)) if i not in SVG_DROP_DIMS]
    return s[keep]  # 16-dim


def shape_dist_vector(shape_dist: Any) -> np.ndarray:
    """Convert shape_distribution to fixed 4-dim vector."""
    if isinstance(shape_dist, list):
        return np.array(shape_dist[:4], dtype=np.float32)
    if isinstance(shape_dist, dict):
        return np.array([float(shape_dist.get(st, 0.0)) for st in SHAPE_TYPES], dtype=np.float32)
    return np.zeros(4, dtype=np.float32)


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

class PartFeatures:
    """Per-part feature vector."""
    __slots__ = ("part_id", "geometry", "svg", "shape_dist", "raw_geometry", "graph_feats")

    def __init__(self, part_id: int, geometry: np.ndarray, svg: np.ndarray,
                 shape_dist: np.ndarray, raw_geometry: np.ndarray,
                 graph_feats: np.ndarray):
        self.part_id = part_id
        self.geometry = geometry      # 14-dim (log1p fixed)
        self.svg = svg                # 16-dim (dead dim dropped)
        self.shape_dist = shape_dist  # 4-dim
        self.raw_geometry = raw_geometry  # 14-dim original (for diagnostics)
        self.graph_feats = graph_feats   # 4-dim: degree, neighbor_deg, clustering, log_degree

    @property
    def feature_vec(self) -> np.ndarray:
        """Combined feature: geometry(14) + svg(16) + shape_dist(4) + graph(4) = 38-dim."""
        return np.concatenate([self.geometry, self.svg, self.shape_dist, self.graph_feats])


class ConnectionGraph:
    """Undirected graph of part connections."""
    def __init__(self, num_parts: int, edges: List[Tuple[int, int]]):
        self.num_parts = num_parts
        self.edges = edges
        self.adj: List[List[int]] = [[] for _ in range(num_parts)]
        self._adj_sets: List[frozenset] = [frozenset() for _ in range(num_parts)]
        for a, b in edges:
            self.adj[a].append(b)
            self.adj[b].append(a)
        self._adj_sets[a] = frozenset(self.adj[a])
        self._adj_sets[b] = frozenset(self.adj[b])
        # Precompute graph features
        self._graph_feats = self._compute_graph_features()

    def _compute_graph_features(self) -> np.ndarray:
        """Compute graph-theoretic features for each node.

        Returns [N, 4] array with columns:
          degree, neighbor_deg_mean, local_clustering_coef, log1p(degree)
        """
        n = self.num_parts
        degs = np.array([len(self.adj[i]) for i in range(n)], dtype=np.float32)

        # Neighbor mean degree
        neighbor_deg_mean = np.zeros(n, dtype=np.float32)
        for i in range(n):
            if degs[i] > 0:
                neighbor_deg_mean[i] = np.mean([degs[j] for j in self.adj[i]])
            else:
                neighbor_deg_mean[i] = 0.0

        # Local clustering coefficient
        clustering = np.zeros(n, dtype=np.float32)
        for i in range(n):
            neighbors = self.adj[i]
            d = len(neighbors)
            if d < 2:
                clustering[i] = 0.0
                continue
            edges_between = 0
            nset = frozenset(neighbors)
            for u in neighbors:
                for v in self.adj[u]:
                    if v != i and v in nset:
                        edges_between += 1
            # Each edge counted twice (u→v and v→u), so divide by 2
            edges_between //= 2
            clustering[i] = (2.0 * edges_between) / (d * (d - 1))

        log_deg = np.log1p(degs)

        return np.stack([degs, neighbor_deg_mean, clustering, log_deg], axis=-1)

    def get_graph_features(self, part_id: int) -> np.ndarray:
        """Return 4-dim graph features for a part."""
        return self._graph_feats[part_id]

    @property
    def edge_index(self) -> np.ndarray:
        """COO format edge_index [2, E] for GNN, with reverse edges."""
        if not self.edges:
            return np.zeros((2, 0), dtype=np.int64)
        src, dst = [], []
        for a, b in self.edges:
            src.extend([a, b])
            dst.extend([b, a])
        return np.array([src, dst], dtype=np.int64)

    @property
    def adjacency_matrix(self) -> np.ndarray:
        """Dense adjacency matrix [N, N]."""
        adj = np.zeros((self.num_parts, self.num_parts), dtype=np.float32)
        for a, b in self.edges:
            adj[a, b] = 1.0
            adj[b, a] = 1.0
        return adj


class SVGSpatialStep:
    """SVG spatial features for one assembly step.

    Instances are indexed by position (0, 1, 2, ...), NOT by stroke color,
    because the same part may use different stroke colors across steps.
    The mapping from instance position to GT part ID comes from the gt.parts
    field in simplified_instances.json.
    """
    def __init__(self, step_id: int, instance_features: List[np.ndarray],
                 gt_part_ids: List[str], connections: List[Tuple[str, str]]):
        self.step_id = step_id
        # instance_features[i] corresponds to gt_part_ids[i]
        self.instance_features = instance_features  # list of feature vectors
        self.gt_part_ids = gt_part_ids               # e.g. ["0", "2", "1"] or ["0,1,2", "3"]
        self.connections = connections




def _extract_spatial_features(inst: Dict[str, Any]) -> np.ndarray:
    """Extract a fixed-size spatial feature vector from a simplified SVG instance.

    Returns 10-dim vector:
    [center_x_norm, center_y_norm, bbox_w_norm, bbox_h_norm,
     axis_length_norm, axis_width_norm, log1p(elongation),
     log1p(hull_area), log1p(poly_area), shape_type_onehot... actually keep it small]
    """
    bbox = inst.get("bbox") or [0, 0, 0, 0]
    bw = max(0.0, bbox[2] - bbox[0])
    bh = max(0.0, bbox[3] - bbox[1])
    # Normalize by typical canvas size
    canvas_w, canvas_h = 793.701, 1122.52
    center = inst.get("center") or [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]

    al = float(inst.get("axis_length") or 0)
    aw = float(inst.get("axis_width") or 0)
    elong = float(inst.get("elongation") or 1.0)

    hull = inst.get("convex_hull") or []
    poly = inst.get("simplified_polygon") or []
    hull_area = _polygon_area(hull)
    poly_area = _polygon_area(poly)

    shape_onehot = [1.0 if inst.get("shape_type") == st else 0.0 for st in SHAPE_TYPES]

    return np.array([
        center[0] / canvas_w,
        center[1] / canvas_h,
        bw / canvas_w,
        bh / canvas_h,
        al / math.hypot(canvas_w, canvas_h),
        aw / math.hypot(canvas_w, canvas_h),
        math.log1p(max(0.0, elong)),
        math.log1p(max(0.0, hull_area)),
        math.log1p(max(0.0, poly_area)),
        *shape_onehot,
    ], dtype=np.float32)


def _polygon_area(pts: List[List[float]]) -> float:
    """Shoelace formula for polygon area."""
    if len(pts) < 3:
        return 0.0
    area = 0.0
    n = len(pts)
    for i in range(n):
        j = (i + 1) % n
        area += pts[i][0] * pts[j][1]
        area -= pts[j][0] * pts[i][1]
    return abs(area) / 2.0


# ---------------------------------------------------------------------------
# Object-level record
# ---------------------------------------------------------------------------

class ObjectRecord:
    """One IKEA object with all features and ground truth."""
    def __init__(
        self,
        category: str,
        name: str,
        split: str,
        num_parts: int,
        part_features: List[PartFeatures],
        graph: ConnectionGraph,
        assembly_tree: Any,  # nested list
        tree_actions: List[Dict[str, Any]],
        equivalence: Dict[str, List[str]],
        svg_steps: List[SVGSpatialStep],
    ):
        self.category = category
        self.name = name
        self.split = split
        self.num_parts = num_parts
        self.part_features = part_features
        self.graph = graph
        self.assembly_tree = assembly_tree
        self.tree_actions = tree_actions
        self.equivalence = equivalence
        self.svg_steps = svg_steps

    def feature_matrix(self) -> np.ndarray:
        """[N, D] part feature matrix."""
        return np.stack([pf.feature_vec for pf in self.part_features])


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(
    dataset_json: Path = DATASET_JSON,
    simplified_svg_dir: Path = SIMPLIFIED_SVG_DIR,
) -> List[ObjectRecord]:
    """Load and fix the full dataset."""
    records = json.loads(dataset_json.read_text(encoding="utf-8"))
    objects = []

    for rec in records:
        category = rec["category"]
        name = rec["name"]
        num_parts = rec["num_parts"]

        # Connection graph (build first to compute graph features)
        edges = [tuple(e) for e in (rec.get("connection_relation") or [])]
        graph = ConnectionGraph(num_parts, edges)

        # Part features with log1p fixes + graph features
        part_features = []
        for pt in rec["part_tokens"]:
            pid = int(pt["part_id"])
            geom = fix_geometry_feature(pt["geometry_feature"])
            svg = fix_svg_feature(pt["svg_feature_mean"], pt.get("svg_feature_count", 0))
            sd = shape_dist_vector(pt.get("shape_distribution", {}))
            raw_geom = np.array(pt["geometry_feature"], dtype=np.float32)
            gf = graph.get_graph_features(pid)
            part_features.append(PartFeatures(pid, geom, svg, sd, raw_geom, gf))

        # Assembly tree (keep nested list format)
        tree = rec["assembly_tree"]

        # Tree actions (postorder merge steps)
        actions = rec.get("tree_actions_postorder") or []

        # Equivalence
        equiv = rec.get("geometric_equivalence_relation") or {}

        # SVG spatial features per step (from simplified_svg)
        svg_steps = _load_svg_steps(simplified_svg_dir, category, name)

        objects.append(ObjectRecord(
            category=category,
            name=name,
            split=rec.get("split", "train"),
            num_parts=num_parts,
            part_features=part_features,
            graph=graph,
            assembly_tree=tree,
            tree_actions=actions,
            equivalence=equiv,
            svg_steps=svg_steps,
        ))

    return objects


def _load_svg_steps(svg_dir: Path, category: str, name: str) -> List[SVGSpatialStep]:
    """Load simplified SVG steps for one object.

    IMPORTANT: Instance identity is by POSITION (index in instances list),
    matching gt.parts by position. Stroke color is NOT a stable identity
    across steps — the same part can have different colors in different steps.
    """
    obj_dir = svg_dir / category / name
    if not obj_dir.exists():
        return []

    steps = []
    for step_dir in sorted(obj_dir.iterdir()):
        if not step_dir.is_dir() or not step_dir.name.startswith("step_"):
            continue
        json_file = step_dir / "simplified_instances.json"
        if not json_file.exists():
            continue

        data = json.loads(json_file.read_text(encoding="utf-8"))
        instances = data.get("instances", [])
        gt = data.get("gt", {})
        gt_parts = gt.get("parts", [])
        gt_connections = gt.get("connections") or []

        # Extract features by position (NOT by stroke color)
        instance_features = [_extract_spatial_features(inst) for inst in instances]

        # Parse connections to part ID pairs
        connections = [(c[0], c[1]) for c in gt_connections]

        step_id = data.get("step_id", int(step_dir.name.split("_")[1]))
        steps.append(SVGSpatialStep(
            step_id=step_id,
            instance_features=instance_features,
            gt_part_ids=gt_parts,
            connections=connections,
        ))

    return sorted(steps, key=lambda s: s.step_id)


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

def split_dataset(objects: List[ObjectRecord]) -> Tuple[List[ObjectRecord], List[ObjectRecord]]:
    """Split by object-level split field."""
    train = [o for o in objects if o.split == "train"]
    test = [o for o in objects if o.split == "test"]
    return train, test


# ---------------------------------------------------------------------------
# Training example extraction
# ---------------------------------------------------------------------------

def extract_merge_examples(record: ObjectRecord) -> List[Dict[str, Any]]:
    """Extract pairwise merge training examples from GT tree actions.

    For each merge step in the postorder traversal, we generate:
    - Positive pair: the two child groups that should merge
    - Negative pairs: all other active cluster pairs

    Returns list of dicts with keys:
      step_idx, cluster_a, cluster_b, label, active_clusters, record
    """
    actions = record.tree_actions
    if not actions:
        return []

    examples = []
    # Start with each part as its own cluster
    active: List[frozenset] = [frozenset([i]) for i in range(record.num_parts)]

    for step_idx, action in enumerate(actions):
        children = action["children"]
        parent = action["parent"]

        if len(children) < 2:
            # Single-child merge (shouldn't happen normally), just update active
            child_fs = [frozenset(c) for c in children]
            for cf in child_fs:
                if cf in active:
                    active.remove(cf)
            active.append(frozenset(parent))
            continue

        # All pairs of children that merge together
        # For k-ary merge (k>2), we treat it as the model needs to pick
        # the correct pair — we enumerate all child pairs as positive
        child_fs = [frozenset(c) for c in children]
        positive_pairs = set()
        for i in range(len(child_fs)):
            for j in range(i + 1, len(child_fs)):
                positive_pairs.add((child_fs[i], child_fs[j]))

        # Generate all pair comparisons from current active clusters
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                ca, cb = active[i], active[j]
                label = 1.0 if (ca, cb) in positive_pairs or (cb, ca) in positive_pairs else 0.0
                examples.append({
                    "step_idx": step_idx,
                    "cluster_a": ca,
                    "cluster_b": cb,
                    "label": label,
                    "active_clusters": list(active),
                    "record": record,
                })

        # Update active clusters: remove children, add parent
        for cf in child_fs:
            if cf in active:
                active.remove(cf)
        active.append(frozenset(parent))

    return examples


# ---------------------------------------------------------------------------
# Cluster representation
# ---------------------------------------------------------------------------

def cluster_feature_vec(cluster: frozenset, part_features: List[PartFeatures]) -> np.ndarray:
    """Aggregate part features into a cluster representation.

    Returns: [mean, max, min, size, log1p(size)] concatenated for each
    feature sub-block (geometry + svg + shape_dist).
    """
    feats = np.stack([part_features[p].feature_vec for p in sorted(cluster)])
    mean = feats.mean(axis=0)
    mx = feats.max(axis=0)
    mn = feats.min(axis=0)
    size = np.array([len(cluster), math.log1p(len(cluster))], dtype=np.float32)
    return np.concatenate([mean, mx, mn, size])  # D*3 + 2


def pair_feature_vec(
    ca: frozenset, cb: frozenset, part_features: List[PartFeatures],
    global_context: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build pair feature for two clusters.

    Standard pattern: [repr_a, repr_b, |a-b|, a*b, repr_union]
    Optionally appended with global context vector.
    """
    ra = cluster_feature_vec(ca, part_features)
    rb = cluster_feature_vec(cb, part_features)
    union = frozenset(ca | cb)
    ru = cluster_feature_vec(union, part_features)

    parts = [ra, rb, np.abs(ra - rb), ra * rb, ru]
    feat = np.concatenate(parts)

    if global_context is not None:
        feat = np.concatenate([feat, global_context])

    return feat


def global_context_features(active_clusters: List[frozenset], part_features: List[PartFeatures]) -> np.ndarray:
    """Compute global context from current set of active clusters.

    Features: cluster_count, log1p(count), total_parts,
    mean/std/max/min of cluster sizes,
    mean/std of cluster_repr vectors.
    """
    sizes = np.array([len(c) for c in active_clusters], dtype=np.float32)
    n = len(active_clusters)
    total = float(sizes.sum())

    # Aggregate cluster reprs
    reprs = np.stack([cluster_feature_vec(c, part_features) for c in active_clusters])
    repr_mean = reprs.mean(axis=0)
    repr_std = reprs.std(axis=0)

    ctx = np.array([
        n, math.log1p(n), total,
        sizes.mean(), sizes.std(), sizes.max(), sizes.min(),
    ], dtype=np.float32)

    return np.concatenate([ctx, repr_mean, repr_std])


# ---------------------------------------------------------------------------
# Pair feature dimensionality
# ---------------------------------------------------------------------------

def pair_feature_dim(include_context: bool = True) -> int:
    """Return the dimensionality of pair_feature_vec output."""
    # feature_vec dim: 14(geom) + 16(svg) + 4(shape) + 4(graph) = 38
    feat_dim = 38
    repr_dim = feat_dim * 3 + 2  # mean + max + min + size + log_size
    pair_dim = repr_dim * 5       # repr_a, repr_b, diff, product, union
    if include_context:
        ctx_base = 7  # count, log_count, total, mean_size, std_size, max_size, min_size
        ctx_dim = ctx_base + repr_dim * 2  # + repr_mean + repr_std
        pair_dim += ctx_dim
    return pair_dim
