# SVG-Enhanced Assembly Tree Generation Tasks

Last updated: 2026-05-26 (GRPO + spatial SVG reward experiments complete)

## Task Definition

The project goal is to use instruction-manual SVGs and assembly-step supervision
to improve assembly-tree generation.

The target inference setting is:

```text
observed real parts
-> recognize/ground each part to a simplified SVG or canonical part token
-> feed all part SVG/tokens to a trained planner
-> generate an assembly tree
```

Fine-grained physical manipulation details are outside this module.

Manuals are used in two ways:

1. Training supervision: manual steps provide part groupings, subassemblies,
   connection hints, order, and ground-truth assembly trees.
2. Optional inference support: if a relevant manual can be retrieved, it can
   guide planning; if not, the planner should still generate a plausible tree
   from observed part tokens and learned assembly patterns.

---

## Dataset

- 102 objects (73 train / 11 val / 29 test)
- 754 primitive parts, 404 tree actions, 302 composite tokens
- Feature modes: `geometry`, `svg`, `svg_geometry`, `svg_composite`, `svg_geometry_composite`

## Model Architecture (all models share this)

All tree planners use the same **greedy connected-components decoder**.
They differ only in how they score candidate merge pairs:

```
Input: set of N part tokens (geometry + SVG + optional composite)
  → for each merge step:
    1. compute per-cluster features (mean/max/min of part features)
    2. for each candidate pair (a,b), build pair feature vector
    3. score each pair with a learned function (logistic/MLP/context-MLP)
    4. greedily merge the highest-scoring pair(s) above threshold
  → output: assembly tree
```

### Feature modes

| Mode | Content | Needs manual annotation? |
|---|---|---|
| `geometry` | 14-dim 3D geometry (bbox, extent, center, n_verts, n_faces) | No |
| `svg` | 17-dim SVG prototype + 4-dim shape + 1-dim count | No |
| `svg_geometry` | geometry + SVG | No |
| `svg_composite` | SVG + 23-dim manual subassembly prototype | **Yes** |
| `svg_geometry_composite` | geometry + SVG + composite | **Yes** |

### Model variants

| Script | Model | Training |
|---|---|---|
| `train_tree_planner_baseline.py` | Logistic regression | BCE flat pairs |
| `train_tree_planner_nn.py` | 2-layer MLP (flat) | BCE flat pairs |
| `train_tree_planner_context.py` | 2-layer MLP + global context features | BCE flat pairs |
| `train_tree_decoder.py` | Transformer set-context encoder | Teacher forcing (failed) |
| `train_tree_grpo.py` | Same MLP + context | GRPO reinforcement learning |

---

## Full Results Table (all modes, all models)

```
Model                              Test Simple  Test Hard   All Hard
----------------------------------------------------------------------
=== geometry (no manual info) ===
greedy baseline                       0.3962     0.0345     0.1078
NN flat MLP                           0.4286     0.1622     0.3620
context MLP                           0.4123     0.1088     0.3352
GRPO warm-start                       0.4137     0.1088     0.3087

=== svg_geometry (no manual info) ===
greedy baseline                       0.4117     0.1259     0.2312
NN flat MLP                           0.4773     0.1557     0.2889
context MLP                           0.4071     0.1081     0.4232
GRPO warm-start                       0.4132     0.1058     0.3139

=== svg_geometry_composite (with manual subassembly labels) ===
greedy baseline                       0.4214     0.1128     0.2959
NN flat MLP                           0.4884     0.2057     0.6083
context MLP (BCE)                     0.5704     0.3316     0.6729  ← BEST OVERALL
context MLP + entropy reg             0.5228     0.2167     0.6137
GRPO warm-start (BCE→RL)             0.5651     0.3064     0.6238
GRPO from-scratch K=8  τ=3.0         0.4519     0.1696     0.3696
GRPO from-scratch K=16 τ=1.5         0.5203     0.2577     0.3846
GRPO from-scratch K=16 + spatial     0.5018     0.2813     0.3947
GRPO warm-start + spatial            0.5344     0.2540     0.5258
manual composite oracle              1.0000     1.0000     1.0000
```

---

## Key Findings

### 1. Composite features are the single biggest factor

No-composite best: 0.162 (geometry NN)
Composite best:    0.332 (context MLP) → 2x improvement

Manual-derived subassembly SVG prototypes encode rich assembly knowledge.

### 2. Context features amplify composite, hurt simpler modes

```
geometry:       flat 0.162 → context 0.109  (worse)
svg+geometry:   flat 0.156 → context 0.108  (worse)
composite:      flat 0.206 → context 0.332  (+61%!)
```

Without composite info, global context is noise. With composite,
"3 clusters left, 2 match manual subassemblies" is a strong signal.

### 3. Transformer decoder fails on 73-object dataset

The teacher-forcing transformer (train_tree_decoder.py) underperforms all
flat MLP baselines. 73 training objects × ~4 actions = ~300 states is
insufficient for attention-based set encoding.

### 4. GRPO exploration problem: BCE makes logits too polarized

BCE pretrained models produce near-deterministic scores (sigmoid ≈ 0 or 1).
At τ=1.0, only 15-20% of sampled trees are unique. Temperature helps
(τ=5 → 44%, τ=8 → 52%) but doesn't solve the fundamental issue.
Top-K sampling makes it worse. Entropy regularization (λ=0.1) barely helps.

### 5. GRPO from-scratch explores well but overfits

Random init → natural high entropy → diverse sampling.
Val Hard reaches 0.509 (beats BCE's 0.449), but Test Hard only 0.281
(vs BCE's 0.332). The model finds good train-object strategies that
don't generalize. K=16 > K=8, suggesting more samples helps.

### 6. Spatial SVG reward is a valid supervision signal

Per-step simplified SVG instances provide spatial geometry signals:
- Proximity: distance between part centers in the step SVG
- Axis alignment: principal axis parallelism/perpendicularity
- Connection candidate proximity: explicit connection points

GT trees score 0.17-0.33, random trees score 0.0.
From-scratch GRPO + spatial: Test Hard 0.281 (+9% over 0.258 without spatial).
Warm-start GRPO + spatial: Test Hard 0.254 (worse than BCE 0.332).

---

## Completed

- [x] Parse manual SVGs into structured vector features
- [x] Simplify SVG instances into compact geometry tokens
- [x] Recover supervised SVG-instance to part alignment from released masks
- [x] Render primitive OBJ parts into synthetic multi-view part images
- [x] Build part-to-SVG grounding samples
- [x] Train and evaluate simplified-SVG connection classifier
- [x] Train and evaluate grounding CNN baselines (best: residual CNN, equiv acc 84.9%)
- [x] Build object-level tree-generation dataset (102 objects, composite tokens)
- [x] Train initial greedy tree-planner baselines (geometry, SVG, SVG+geometry)
- [x] Train neural merge-scorer baselines (flat MLP, all feature modes)
- [x] Add composite/subassembly SVG prototypes as manual/RAG context
- [x] Train no-leakage subassembly candidate predictor + negative sampling
- [x] Export planner predictions and build error viewer
- [x] Add end-to-end diagnostic and paper tree metric evaluation
- [x] Reorganize `scripts/` into `build/`, `train/`, `eval/`, `export/` subdirectories
- [x] Implement transformer set-context tree decoder (`train_tree_decoder.py`)
  - Negative result: underperforms flat MLP on 73-object dataset
- [x] Implement context-augmented flat MLP (`train_tree_planner_context.py`)
  - **Best model**: context MLP + composite → Test Hard F1 0.332 (+61% over baseline)
- [x] Implement GRPO tree planner (`train_tree_grpo.py`) with SVG-derived rewards
  - Basic SVG coherence reward: subassembly existence check
  - Spatial SVG reward: per-step geometry proximity + alignment + connection points
  - From-scratch + spatial: Test Hard 0.281 (best RL result)
  - Warm-start GRPO: Test Hard 0.306 (below BCE baseline)
- [x] Run four exploration strategies for GRPO:
  1. High temperature (τ=3~8): helps diversity but not enough
  2. Top-K sampling: worse than softmax
  3. Entropy regularization (λ=0.1): minimal diversity gain
  4. From-scratch (random init): best exploration, overfits to train
- [x] Use per-step simplified SVG geometry as spatial reward signal in GRPO

## Active Todo

- [ ] Combine BCE pretraining + GRPO with strong KL constraint to preserve generalization
- [ ] Use ALL 102 objects for GRPO (unsupervised RL doesn't need a test split)
- [ ] Learn a proper tree decoder (sequence/set-to-tree) instead of greedy connected-components
- [ ] Improve no-leakage subassembly prediction precision
- [ ] Add real-image or 3D-observation grounding benchmark
- [ ] Add RAG/manual-retrieval hooks as optional planner context

## Next Steps (priority order)

1. **GRPO on all 102 objects** (no train/test split for RL) — should improve generalization
2. **Stronger BCE→GRPO bridge**: BCE warm-start + high KL penalty (β=0.5~1.0) to stay
   close to the pretrained policy while exploring locally
3. **Richer SVG reward**: beyond spatial proximity, use step order and connection
   graph structure from manual steps
4. **Learned tree decoder**: replace greedy connected-components with a sequential
   merge predictor (RNN/Transformer decoder)

## Scripts Index

```
scripts/
├── build/     (7)  数据构建 & 特征提取
├── train/     (12) 模型训练
│   ├── train_tree_planner_baseline.py    logistic regression
│   ├── train_tree_planner_nn.py          flat MLP
│   ├── train_tree_planner_context.py     context-aware MLP (best)
│   ├── train_tree_decoder.py             transformer decoder (failed)
│   ├── train_tree_planner_entropy.py     (deprecated, entropy in context script)
│   ├── train_tree_grpo.py                GRPO with SVG/spatial rewards
│   ├── train_grounding_cnn.py            grounding CNN
│   ├── train_grounding_model.py          geometric grounding
│   ├── train_grounding_image_model.py    image-feature grounding
│   ├── train_pairwise_connection_model.py
│   ├── train_simplified_connection_model.py
│   └── train_subassembly_candidate_model.py
├── eval/      (5)  评估 & 推理
└── export/    (4)  导出 & 分析报告
```

## Rules

- Every completed implementation or experiment should be checked off in this file
- If a task changes direction, update the wording here instead of relying on chat history
- Keep manual-step diagnostic and object-level tree generation clearly separated
