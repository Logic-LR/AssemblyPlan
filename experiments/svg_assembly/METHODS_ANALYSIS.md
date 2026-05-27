# Assembly Tree Generation: Methods, Results, and Analysis

Last updated: 2026-05-27

---

## 1. Background

### Task

Given N primitive parts (geometry + per-part SVG features), predict the assembly tree.

```
Input:  [part_0, part_1, ..., part_N]  (geometry + SVG shape + spatial features)
Output: assembly tree  e.g. [[0, 1, 2], 3]  (merge {0,1,2} first, then add 3)
```

### Data

- 102 IKEA furniture objects (73 train, 29 test)
- 754 primitive parts, 404 tree actions
- 393 manual steps with per-step simplified SVG instances
- Each manual step: `parts` (active clusters) + `connections` (explicit connection edges)
- GT assembly trees + tree_actions_postorder

### Metrics

- **Simple F1**: Do the predicted non-leaf node part-sets match GT?
- **Hard F1**: Do the predicted node children (as sets, ignoring order) match GT?

### Original IKEA-Manual Paper Approach

```
Method:  DGCNN extracts 1024-dim point-cloud features per part
         → K-Means (k=2,3) on features → silhouette score picks best k
         → Best cluster = subassembly → recurse on remaining parts
         → Top-down recursive tree building

Model:   0 learned parameters (purely feature-based clustering)
Training: none (unsupervised)
Input:   per-part point cloud → DGCNN → 1024-dim feature
Output:  assembly tree (via recursive clustering)

Key insight: Shape similarity implicitly encodes assembly hierarchy.
             Four table legs look similar → cluster together first.
             Different-shaped parts cluster at higher levels.

Limitation: Doesn't use manual step SVGs at all.
```

---

## 2. All Methods We Attempted

### Method 1: BCE context MLP (BEST)

```
Script:   train_tree_planner_context.py
Model:    MLP(938→192→192→1), 218K params
Input:    pair_feat = [rep(A), rep(B), |A-B|, A*B, rep(A∪B), global_ctx], 938-dim
           global_ctx = [M, log(M), total_parts, mean/std/cluster_sizes] + mean/std(all cluster_reprs)
Training: pool ALL pairs from ALL steps of ALL objects → 4500 samples
          BCE loss: positive = child-child pairs in GT tree action
          Global standardization of input features
Inference: greedy: score all pairs → threshold → connected-components grouping → merge
           threshold tuned on val set (16 values, best Hard F1)
```

**Results:**

| Feature Mode | Test Simple | Test Hard | All Hard |
|---|---|---|---|
| geometry | 0.412 | 0.109 | 0.335 |
| svg_geometry | 0.407 | 0.108 | 0.423 |
| svg_geometry_composite | **0.570** | **0.332** | **0.673** |

**5-seed split variance (svg_geometry_composite):**
```
s0=0.332  s1=0.303  s2=0.254  s3=0.136  s4=0.346
mean=0.274 ± 0.085
```

**Problems:**

1. **66% flat tree predictions** (19/29 test objects). Model outputs single-step merge of all parts.
2. **Same pair gets same score regardless of step.** Test on applaro_2:
   - pair (0,2) at step 0: score 0.964
   - pair (0,2) at step 1: score 0.987 ← nearly identical
   - 665-dim local pair features are IDENTICAL at both steps
   - Only 273-dim global_ctx differs, and the difference is tiny (mean abs diff = 0.40)
3. **Feature redundancy.** |A-B| and A*B produce 266 dimensions of pairwise products that are mostly noise.
4. **Labels are coarse.** All child-child pairs in a GT tree action get label=1, even pairs that aren't directly connected in the manual SVG.
5. **Composite tokens are 2x cheat.** svg_geometry_composite (0.332) vs svg_geometry (0.108). Composite tokens encode manual subassembly annotations — the model is told the answer.
6. **Score compression at inference.** At test time, all pair probabilities cluster at 0.4-0.5. The model doesn't produce differentiated scores.

**Why the high score is misleading:**

```
BCE scores 0.332 because:
  - A few objects get lucky (correct merges by chance)
  - 19/29 objects get flat trees → Hard F1 = 0.00
  - The average is inflated by the lucky few
```

---

### Method 2: GRPO with SVG-derived Rewards

```
Script:   train_tree_grpo.py
Model:    Same ContextMergeMLP (218K params)
          + reference model (frozen copy for KL penalty)
Training: Group Relative Policy Optimization (GRPO)
          Per object: sample K=8-16 trees, compute reward, normalize within group
          Policy gradient with clipped importance sampling
          KL penalty towards reference model
Reward:   r = w1 * SVG_coherence + w2 * spatial_SVG + w3 * GT_F1
          SVG_coherence: fraction of non-leaf nodes appearing in any manual step
          spatial_SVG: per-step SVG geometry (proximity + axis alignment + connection candidates)
Inference: Same greedy CC decoder as BCE
```

**Variants:**

| Variant | Description | Result |
|---|---|---|
| Warm-start from BCE | BCE model → GRPO fine-tune, β=0.05 | Test Hard 0.306 |
| From-scratch K=8, τ=3.0 | Random init, high temperature exploration | Test Hard 0.170 |
| From-scratch K=16, τ=1.5 | More samples, moderate temperature | Test Hard 0.258 |
| From-scratch + spatial reward | Added spatial SVG geometry to reward | Test Hard 0.281 |
| Warm-start + spatial | BCE init + spatial reward | Test Hard 0.254 |

**Multi-seed split-seed runs (svg_geometry_composite):**

| Experiment | Test Hard |
|---|---|
| BCE 100% label | 0.274 ± 0.085 (n=5) |
| BCE 10% label | 0.184 ± 0.044 (n=3) |
| BCE 25% label | 0.218 ± 0.049 (n=3) |
| GRPO 10% label | **0.237** (+29% over BCE) |
| GRPO 25% label | **0.274** (+5% over BCE) |
| GRPO SVG-only (0% GT, pure SVG reward) | ~0.346 (3-seed mean) |

**Problems:**

1. **Exploration failure in warm-start.** BCE-trained logits are near-deterministic (sigmoid ≈ 0 or 1).
   - At τ=1.0, only 15-20% of sampled trees are unique
   - At τ=5.0, only 44% unique; τ=8.0 → 52% unique
   - Entropy regularization (λ=0.1) barely helps (20% → 22%)
   - Top-K sampling makes it worse (restricts to top model choices)
   
2. **Overfitting in from-scratch.** Random init → high entropy → good exploration.
   - Val Hard reaches 0.509 (beats BCE's 0.442)
   - Test Hard only 0.281 (massive gap)
   - Model finds strategies that work on the 62 train objects but don't generalize

3. **Reward signal issues.**
   - SVG coherence: many different trees get the same reward → no discriminative power
   - Spatial SVG: GT trees score 0.17-0.33, random trees 0.0 → valid signal but weak
   - Pearson r(SVG_coherence, GT_F1) = 1.0 on small objects, but very few unique trees sampled

4. **Train-test inconsistency.** GRPO trains the policy to maximize reward, but inference uses greedy argmax.
   The reward signal is discarded at test time.

5. **Spatial features absent at inference.** Model trained with per-step SVG spatial features (center_delta, axis_alignment, proximity) but inference passes empty spatial_map={}. The model never learned to function without this crutch.

6. **Per-object optimizer step causes instability.** Loss fluctuates wildly (1.2 → 141.5) because optimizer.step() is called per object rather than per minibatch.

---

### Method 3: Transformer Set-Context Tree Decoder

```
Script:   train_tree_decoder.py
Model:    SetContextTreeDecoder
          input_proj: Linear→LayerNorm→ReLU
          encoder: 2-layer TransformerEncoder (128-dim, 4 heads)
          pair_scorer: Linear(512→128→64→1)
Training: Teacher forcing through GT tree actions
          At each step: encode all current clusters with transformer
          Score all pairs → BCE loss (positive = children pairs)
Inference: Same greedy CC decoder
```

**Result: Underperforms flat MLP on 73-object dataset.**
(Specific numbers not recorded — model was abandoned early.)

**Problems:**

1. **Insufficient data.** 73 objects × ~4 actions = ~300 training states.
   Transformer attention needs thousands of examples to learn meaningful patterns.
   With 300 states, attention weights are essentially random.

2. **Unnecessary complexity for the task.** Self-attention over 4-12 clusters provides minimal benefit
   over a flat MLP when the dataset is this small.

---

### Method 4: Step-Conditioned Sequential Planner (v1)

```
Script:   train_step_planner.py (v1)
Model:    StepMergePlanner
          PartEncoder: shape_emb(4→16) + spatial_proj(9→32→16) → 32-dim per part
          ClusterEncoder: MultiheadAttention(32, 4 heads) → Linear(32→hidden_dim)
          GRU(256→256): tracks progress through steps
          step_emb(20→32): explicit step counter
          pair_scorer: Linear(773→256→128→1)
          Total: 640K params
Training: Teacher forcing through GT tree_actions_postorder (~250 states)
          At each step: encode clusters + GRU state + step_idx + optional spatial features
          BCE loss on pairs (positive = children of current action)
          Random 50% spatial feature dropout (to force GRU to learn state)
Inference: GRU tracks state → score pairs → greedy CC merge
```

**Results:**
- Best val Hard: 0.127 (epoch 40, then degrades)
- Test Hard: 0.022
- After adding step_emb: score std improved from 0.07 → 0.21, but still flat trees

**Problems:**

1. **Only 250 training states.** 62 objects × ~4 tree actions = ~250 states.
   Each state appears only once per object. GRU can't learn meaningful state transitions
   from 250 examples.

2. **Inference without spatial features.** Trained WITH per-step SVG spatial features
   (center_delta, distance, axis_alignment). At inference, spatial_map={}.
   Model learned to rely on spatial features as a crutch — without them, scores collapse.

3. **50% spatial dropout insufficient.** Even with dropout, the GRU doesn't learn
   to encode "which step am I at?" from cluster features alone.

4. **Activeness head didn't help.** Added prediction of "which clusters are active in this step?"
   Training labels: children of current tree action. But the extra loss term didn't
   improve discrimination.

---

### Method 5: Step-Conditioned Sequential Planner (v2 — connections)

```
Script:   train_step_planner.py (v2, rewritten)
Model:    StepPlanner (simpler than v1)
          shape_emb(4→16) + spatial_proj(8→32→16)
          cluster_attn → GRU(256) + step_emb(20→32)
          pair_scorer: Linear(800→256→128→1)
Training: 393 manual steps, each step's CONNECTIONS as positive pair labels
          Teacher forcing through manual step sequence
          BCE loss on pairs
Inference: GRU + step_emb → score pairs → CC merge
```

**Results (5 epochs):**
- Test Hard: 0.021 (still flat trees)
- Loss: 0.254 → 0.178 (decreasing)

**Problems:**

1. **Connections don't encode hierarchy.** All manual steps' connections look structurally
   similar to the model — pairs of clusters with an edge between them.
   The model learns "these are connected" but not "these connect at step 0, these at step 3."

2. **Still only 393 states across 62 objects.** Not enough diversity for GRU generalization.

3. **GRU input is too compressed.** The GRU receives mean(cluster_embeddings) — a single vector.
   It can't distinguish "6 singletons" from "2 large clusters" from this mean alone.

---

### Method 6: Clustering + Contrastive Learning

```
Script:   train_cluster_planner.py
Model:    PartProjector: Linear(8→128→128→64), 50K params
Training: Contrastive loss on connections
          Positive: pull connected part-pairs close in embedding space
          Negative: push unconnected part-pairs apart (> margin)
          Hinge loss: pos_dist.mean() + clamp(margin - neg_dist, 0).mean()
Inference: K-Means(k=2,3) + silhouette score → recursive top-down tree
          (Same algorithm as the original IKEA-Manual paper)
```

**Results (30 epochs):**
- Test Hard: 0.035 (same as geometry-only greedy baseline)
- Contrastive ratio: connected pairs 2.5x closer than unconnected
- Loss decreasing: 12.46 → 0.60

**Problems:**

1. **Contrastive loss flattens hierarchy.** All connections across all steps are treated equally.
   Part {0,2,3,4} connect in step 0, then connect to {5} in step 1, then to {1} in step 2.
   All 6 parts end up close together → K-Means can't find meaningful splits.

2. **Shape similarity vs connectivity.** The original paper uses DGCNN features that encode
   SHAPE similarity (four table legs look alike → cluster together).
   Our contrastive embeddings encode CONNECTIVITY (all connected parts are close).
   These are fundamentally different — connectivity doesn't encode assembly hierarchy.

3. **K-Means is non-parametric.** No way to inject step order or manual knowledge into the
   clustering process. Silhouette score is a poor proxy for assembly tree quality.

---

## 3. Summary: Root Causes

### Why all our methods struggle

| Root Cause | Which Methods Affected | Explanation |
|---|---|---|
| **Step order not used** | Methods 1-6 | All methods pool or average across steps. The model never learns "merge {0,1,2} first, THEN {0,1,2,3}". |
| **Training data too small** | Methods 3-5 | 62 objects × ~4 tree actions = ~250 states. GRU/transformer need thousands of states to learn meaningful transitions. |
| **Features encode "what" not "when"** | Methods 1-6 | Part features tell the model which parts are similar, but not which step they should merge at. |
| **Inference-training mismatch** | Methods 2, 4, 5 | Training uses step SVGs (spatial context), inference doesn't. Model learns the wrong dependency. |
| **Flat cross-entropy can't encode hierarchy** | Methods 1, 4, 5 | BCE per pair treats all steps equally. A pair that should merge at step 1 looks the same as a pair that should merge at step 3. |
| **Label granularity** | Method 1 | All child-child pairs in a GT tree action are positive, even if they're not directly connected in the manual SVG. |
| **Greedy CC decoder is too simple** | Methods 1-5 | Single threshold controls everything. If scores are compressed, CC produces either flat tree (low threshold) or nothing (high threshold). No middle ground. |

### What we have that works

1. **Composite tokens are powerful.** They directly encode manual subassembly knowledge. Test Hard 0.332 vs 0.162 without them. But they're "cheating" — the manual tells you which parts form subassemblies.

2. **GRPO helps when labels are scarce.** At 10% GT labels, GRPO + SVG reward gives +30% over BCE alone. This is a genuine contribution — SVG can substitute for missing GT tree labels.

3. **SVG spatial reward is a valid signal.** GT trees score 0.17-0.33, random trees 0.0. The signal exists — we just haven't figured out how to optimize it effectively.

4. **Step-conditioned models eliminate flat trees.** Sequential GRU models produce 0% flat tree rate (vs BCE's 66%). They produce wrong trees, but at least they produce trees.

### What the original paper gets right

1. **Shape similarity naturally encodes hierarchy.** Similar parts cluster at lower levels.
2. **Recursive clustering natively builds trees.** No sequential prediction needed.
3. **No learned parameters means no overfitting.** Works with any number of objects.
4. **K-Means + silhouette is simple and robust.** No threshold tuning, no gradient issues.

---

## 4. Code and Results Index

### Training Scripts

| Script | Model | Key Idea |
|---|---|---|
| `train_tree_planner_baseline.py` | Logistic regression | Simplest baseline: logistic scorer + greedy CC |
| `train_tree_planner_nn.py` | MLP(777→192→192→1) | Flat MLP pair scorer, no global context |
| `train_tree_planner_context.py` | MLP(938→192→192→1) | + global context features. BEST BCE model |
| `train_tree_decoder.py` | Transformer encoder | Set-context via self-attention. FAILED (data too small) |
| `train_tree_grpo.py` | Same MLP + context | GRPO: policy gradient with SVG rewards |
| `train_step_planner.py` | GRU + step_emb | Sequential: teacher forcing through tree actions. v2: manual_step_groups.connections |
| `train_cluster_planner.py` | PartProjector + K-Means | Contrastive embedding learning + original clustering algorithm |

### Key Experiment Reports

| Report | Method | Best Test Hard |
|---|---|---|
| `context_planner_svg_geometry_composite_report.json` | BCE context MLP + composite | **0.332** |
| `tree_planner_nn_geometry_report.json` | BCE flat MLP geometry | 0.162 (best no-composite) |
| `grpo_svg_geometry_composite_report.json` | GRPO warm-start | 0.306 |
| `grpo_spatial_scratch_report.json` | GRPO from-scratch + spatial | 0.281 |
| `splitseed_bce100_s0.json` | BCE with split-seed control | 0.332 (seed 0) |
| `splitseed_grpo_svgonly_s0.json` | GRPO SVG-only, 0% GT | 0.244 (seed 0) |
| `label_ratio_sgc_full_summary.md` | Label-ratio comparison | GRPO > BCE at 10%, 25% labels |

---

## 5. Open Questions

1. **Can we use manual step order without sequential prediction?** Step embedding + BCE training retains BCE's generalization while adding step awareness. Not tested yet.

2. **Can we get better part features?** Original paper uses DGCNN (1024-dim point cloud features). We use handcrafted 36-dim features. Could part_images/ (754 rendered OBJ views) train a better feature extractor?

3. **Is the greedy CC decoder the bottleneck?** All our models use the same decoder. Would a learned decoder (RNN/pointer network) help even with current features?

4. **How much does data augmentation help?** 102 objects is small. Can we generate synthetic assembly sequences from the 393 manual steps?
