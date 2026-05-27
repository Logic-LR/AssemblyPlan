# 装配树生成：方法、结果与分析

最后更新：2026-05-27

---

## 1. 背景

### 任务

给定 N 个 primitive 零件（几何 + per-part SVG 特征），预测装配树。

```
输入：[part_0, part_1, ..., part_N]（几何 + SVG 形状 + 空间特征）
输出：装配树，如 [[0, 1, 2], 3]（先合并 {0,1,2}，再加 3）
```

### 数据

- 102 个 IKEA 家具对象（73 train, 29 test）
- 754 个 primitive 零件，404 个 tree action
- 393 个 manual step，每个 step 有简化 SVG 实例
- 每个 manual step：`parts`（活跃 cluster）+ `connections`（明确的连接边）
- GT 装配树 + tree_actions_postorder

### 评估指标

- **Simple F1**：预测的非叶节点 parts 集合是否与 GT 一致？
- **Hard F1**：预测的非叶节点 children 集合（无序）是否与 GT 一致？

### 原版 IKEA-Manual 论文做法

```
方法：DGCNN 提取 per-part 的 1024 维点云特征
      → 对特征做 K-Means（k=2,3）→ silhouette score 选最佳 k
      → 最佳簇 = 子装配体 → 递归处理剩余零件
      → 自顶向下递归建树

模型：0 个学习参数（纯特征聚类）
训练：无（无监督）
输入：per-part 点云 → DGCNN → 1024 维特征
输出：装配树（通过递归聚类）

核心洞察：形状相似度天然编码了装配层次。
          四条桌腿长得像 → 先聚在一起。
          形状不同的零件在更高层聚合。

局限性：完全没有用到 manual step SVG。
```

---

## 2. 我们尝试过的所有方法

### 方法 1：BCE context MLP（当前最优）

```
脚本：   train_tree_planner_context.py
模型：   MLP(938→192→192→1)，218K 参数
输入：   pair_feat = [rep(A), rep(B), |A-B|, A*B, rep(A∪B), global_ctx]，938 维
         global_ctx = [M, log(M), total_parts, cluster 大小均值/标准差]
                    + 所有 cluster_repr 的均值/标准差
训练：   将所有 step、所有对象的全部 pair 混在一起 → 4500 个样本
         BCE loss：GT tree action 中的 child-child pair = 正例
         对输入特征做全局标准化
推理：   贪心：对所有 pair 打分 → threshold → CC 分组 → 合并
         threshold 在 val 上调优（16 个候选值，选最佳 Hard F1）
```

**结果：**

| 特征模式 | Test Simple | Test Hard | All Hard |
|---|---|---|---|
| geometry | 0.412 | 0.109 | 0.335 |
| svg_geometry | 0.407 | 0.108 | 0.423 |
| svg_geometry_composite | **0.570** | **0.332** | **0.673** |

**5 seed 方差（svg_geometry_composite）：**
```
s0=0.332  s1=0.303  s2=0.254  s3=0.136  s4=0.346
均值=0.274 ± 0.085
```

**问题：**

1. **66% 的预测是 flat tree**（19/29 个 test 对象）。模型输出单步全合并。
2. **同一 pair 在不同 step 得分几乎一样。** 在 applaro_2 上的测试：
   - pair (0,2) 在 step 0：得分 0.964
   - pair (0,2) 在 step 1：得分 0.987 ← 几乎一致
   - 665 维局部 pair 特征在两步**完全相同**
   - 只有 273 维 global_ctx 不同，且差异极小（平均绝对差 = 0.40）
3. **特征冗余。** |A-B| 和 A*B 产生了 266 维的 cross-product，大部分是噪声。
4. **标签粗糙。** GT tree action 中所有 child-child pair 都标为正例，即使是 manual SVG 中没有直接连接的 pair。
5. **Composite token 是 2 倍外挂。** svg_geometry_composite（0.332）vs svg_geometry（0.108）。Composite token 直接编码了手工子装配体标注——模型被剧透了答案。
6. **推理时分数压缩。** 测试时所有 pair 概率集中在 0.4-0.5，模型无法产生有区分度的分数。

**为什么高分是误导性的：**

```
BCE 的 0.332 分数是因为：
  - 少数对象碰巧对了（运气好的 merge）
  - 19/29 个对象得到 flat tree → Hard F1 = 0.00
  - 被少数幸运对象拉高了均值
```

---

### 方法 2：GRPO + SVG Reward

```
脚本：   train_tree_grpo.py
模型：   同 ContextMergeMLP（218K 参数）
         + 参考模型（冻结副本，用于 KL 惩罚）
训练：   Group Relative Policy Optimization（GRPO）
         每个对象：采样 K=8-16 棵树，计算 reward，组内归一化
         策略梯度 + clipped importance sampling
         KL 惩罚朝向参考模型
Reward： r = w1 * SVG_coherence + w2 * spatial_SVG + w3 * GT_F1
         SVG_coherence：非叶节点出现在 manual step 中的比例
         spatial_SVG：per-step SVG 几何（空间距离 + 轴对齐 + 连接候选点）
推理：   与 BCE 相同的贪心 CC decoder
```

**变体：**

| 变体 | 描述 | Test Hard |
|---|---|---|
| BCE warm-start | BCE 模型 → GRPO 微调，β=0.05 | 0.306 |
| from-scratch K=8, τ=3.0 | 随机初始化，高温探索 | 0.170 |
| from-scratch K=16, τ=1.5 | 更多样本，中等温度 | 0.258 |
| from-scratch + spatial | 加入 spatial SVG 几何 reward | 0.281 |
| warm-start + spatial | BCE 初始化 + spatial reward | 0.254 |

**Multi-seed split-seed 实验（svg_geometry_composite）：**

| 实验 | Test Hard |
|---|---|
| BCE 100% label | 0.274 ± 0.085（n=5） |
| BCE 10% label | 0.184 ± 0.044（n=3） |
| BCE 25% label | 0.218 ± 0.049（n=3） |
| **GRPO 10% label** | **0.237**（比 BCE 高 29%） |
| **GRPO 25% label** | **0.274**（比 BCE 高 5%） |
| GRPO SVG-only（0% GT，纯 SVG reward） | ~0.346（3 seed 均值） |

**问题：**

1. **Warm-start 探索失败。** BCE 训练的 logits 接近确定性（sigmoid ≈ 0 或 1）。
   - τ=1.0 时，只有 15-20% 的采样树是唯一的
   - τ=5.0 时，44% 唯一；τ=8.0 → 52% 唯一
   - Entropy regularization（λ=0.1）几乎没用（20% → 22%）
   - Top-K 采样反而更差（限制在模型最偏好的几个选择上）

2. **From-scratch 过拟合。** 随机初始化 → 高熵 → 好探索。
   - Val Hard 达到 0.509（超过 BCE 的 0.442）
   - Test Hard 只有 0.281（巨大 gap）
   - 模型找到了在 62 个 train 对象上有效的策略，但不泛化

3. **Reward 信号问题。**
   - SVG coherence：很多不同的树得到相同的 reward → 无区分力
   - Spatial SVG：GT 树得分 0.17-0.33，随机树 0.0 → 信号有效但弱
   - 在小对象上 Pearson r(SVG_coherence, GT_F1) = 1.0，但只能采出极少数唯一树

4. **训练-推理不一致。** GRPO 训练 policy 最大化 reward，但推理用贪心 argmax。reward 信号在测试时被丢弃。

5. **推理时缺少 spatial 特征。** 训练时有 per-step SVG 空间特征（center_delta、轴对齐、距离），但推理传入空 spatial_map={}。模型从未学会不依赖这个拐杖。

6. **Per-object 参数更新导致训练不稳定。** Loss 剧烈波动（1.2 → 141.5），因为 optimizer.step() 是对每个对象而非每个 minibatch 调用的。

---

### 方法 3：Transformer Set-Context Tree Decoder

```
脚本：   train_tree_decoder.py
模型：   SetContextTreeDecoder
         input_proj: Linear→LayerNorm→ReLU
         encoder: 2-layer TransformerEncoder（128-dim, 4 heads）
         pair_scorer: Linear(512→128→64→1)
训练：   Teacher forcing 走过 GT tree action
         每一步：用 transformer 编码当前所有 cluster
         对所有 pair 打分 → BCE loss（正例 = children pairs）
推理：   与 BCE 相同的贪心 CC decoder
```

**结果：在 73 个对象的数据集上不如 flat MLP。**
（未记录具体数字——模型很早就被放弃了。）

**问题：**

1. **数据不足。** 73 个对象 × ~4 action = ~300 个训练状态。
   Transformer attention 需要数千个样本才能学到有意义的 pattern。
   300 个状态下 attention 权重基本是随机的。

2. **对这个任务来说过于复杂。** 4-12 个 cluster 上的 self-attention 在这个数据量下比 flat MLP 没有任何优势。

---

### 方法 4：Step-Conditioned Sequential Planner（v1）

```
脚本：   train_step_planner.py（v1）
模型：   StepMergePlanner
         PartEncoder: shape_emb(4→16) + spatial_proj(9→32→16) → 32 维 per part
         ClusterEncoder: MultiheadAttention(32, 4 heads) → Linear(32→hidden_dim)
         GRU(256→256)：追踪装配进度
         step_emb(20→32)：显式 step 计数器
         pair_scorer: Linear(773→256→128→1)
         总计：640K 参数
训练：   Teacher forcing 走过 GT tree_actions_postorder（~250 个状态）
         每一步：编码 cluster + GRU 状态 + step_idx + 可选 spatial 特征
         BCE loss on pairs（正例 = 当前 action 的 children）
         随机 50% spatial 特征 dropout（强迫 GRU 学习状态追踪）
推理：   GRU 追踪状态 → 对 pair 打分 → 贪心 CC merge
```

**结果：**
- 最佳 val Hard：0.127（epoch 40，之后退化）
- Test Hard：0.022
- 加入 step_emb 后：分数标准差从 0.07 提升到 0.21，但仍然是 flat tree

**问题：**

1. **只有 250 个训练状态。** 62 个对象 × ~4 tree action = ~250 个状态。每个状态在每个对象中只出现一次。GRU 无法从 250 个样本中学会有意义的状态转移。

2. **推理时没有 spatial 特征。** 训练时有 per-step SVG spatial 特征（center_delta、距离、轴对齐），推理时 spatial_map={}。模型学会依赖 spatial 特征，没有它就崩溃。

3. **50% spatial dropout 不够。** 即使加了 dropout，GRU 也无法从 cluster 特征中学到"我在第几步"。

4. **Activeness head 没有帮助。** 增加了"哪些 cluster 在这一步活跃？"的预测。训练标签：当前 tree action 的 children。但是额外的 loss 项没有提升区分度。

---

### 方法 5：Step-Conditioned Sequential Planner（v2 — connections）

```
脚本：   train_step_planner.py（v2，重写）
模型：   StepPlanner（比 v1 更简单）
         shape_emb(4→16) + spatial_proj(8→32→16)
         cluster_attn → GRU(256) + step_emb(20→32)
         pair_scorer: Linear(800→256→128→1)
训练：   393 个 manual step，每个 step 的 CONNECTIONS 作为正例 pair 标签
         Teacher forcing 走过 manual step 序列
         BCE loss on pairs
推理：   GRU + step_emb → 对 pair 打分 → CC merge
```

**结果（5 epoch）：**
- Test Hard：0.021（仍然是 flat tree）
- Loss：0.254 → 0.178（在下降）

**问题：**

1. **Connections 不编码层次。** 所有 manual step 的 connection 在模型看来结构相同——都是一对 cluster 之间有一条边。模型学会了"这些是连接的"但没学会"这些在 step 0 连接、那些在 step 3 连接"。

2. **仍然只有 393 个状态，跨 62 个对象。** 不够 GRU 泛化。

3. **GRU 输入太压缩。** GRU 接收 mean(cluster_embeddings)——一个单一向量。它无法区分"6 个单例"和"2 个大 cluster"。

---

### 方法 6：聚类 + 对比学习

```
脚本：   train_cluster_planner.py
模型：   PartProjector: Linear(8→128→128→64)，50K 参数
训练：   在 connections 上做对比学习
         正例：拉近有连接的 part-pair 在嵌入空间中的距离
         负例：推远无连接的 part-pair（超过 margin）
         Hinge loss: pos_dist.mean() + clamp(margin - neg_dist, 0).mean()
推理：   K-Means(k=2,3) + silhouette score → 递归自顶向下建树
         （与 IKEA-Manual 原版论文相同的算法）
```

**结果（30 epoch）：**
- Test Hard：0.035（与 geometry-only greedy baseline 相同）
- 对比比率：有连接的 pair 比无连接的近 2.5 倍
- Loss 在下降：12.46 → 0.60

**问题：**

1. **对比 loss 扁平化了层次。** 所有 step 的所有 connection 被平等对待。零件 {0,2,3,4} 在 step 0 连接，然后在 step 1 连接到 {5}，在 step 2 连接到 {1}。全部 6 个零件最终都靠得很近 → K-Means 找不到有意义的分组。

2. **形状相似度 vs 连接关系。** 原版论文用的是 DGCNN 特征，编码的是**形状相似度**（四条桌腿长得像 → 聚在一起）。我们的对比嵌入编码的是**连接关系**（有连接的零件都靠近）。这两种信息完全不同——连接关系不编码装配层次。

3. **K-Means 没有参数可调。** 无法将 step 顺序或 manual 知识注入聚类过程。Silhouette score 作为装配树质量的 proxy 效果很差。

---

## 3. 总结：根因分析

### 为什么所有方法都有困难

| 根因 | 受影响方法 | 解释 |
|---|---|---|
| **Step 顺序未被使用** | 方法 1-6 | 所有方法都把 steps 混在一起或平均掉。模型从未学到"先合并 {0,1,2}，再合并 {0,1,2,3}"。 |
| **训练数据太少** | 方法 3-5 | 62 个对象 × ~4 tree action = ~250 个状态。GRU/transformer 需要数千个状态才能学到有意义的转移。 |
| **特征编码了"什么"而非"何时"** | 方法 1-6 | Part 特征告诉模型哪些零件是相似的，但不告诉模型它们该在哪个 step 合并。 |
| **推理-训练不一致** | 方法 2, 4, 5 | 训练时用了 step SVG（spatial context），推理时没有。模型学了错误的依赖。 |
| **Flat BCE 无法编码层次** | 方法 1, 4, 5 | 每 pair 的 BCE 把所有 step 平等对待。应该在 step 1 合并的 pair 看起来和应该在 step 3 合并的 pair 一样。 |
| **标签粒度粗** | 方法 1 | GT tree action 中所有 child-child pair 都为正例，即使是 manual SVG 中没有直接连接的 pair。 |
| **贪心 CC decoder 过于简单** | 方法 1-5 | 单一 threshold 控制一切。如果分数被压缩，CC 要么产生 flat tree（低 threshold）要么什么都不合（高 threshold）。没有中间地带。 |

### 有效的东西

1. **Composite token 很强。** 它们直接编码了手工子装配体知识。Test Hard 0.332 vs 0.162（不用 composite）。但它们是"作弊"——manual 直接告诉你哪些零件组成子装配体。

2. **GRPO 在标签稀缺时有用。** 在 10% GT label 时，GRPO + SVG reward 比纯 BCE 高 30%。这是一个真正的贡献——SVG 可以替代缺失的 GT tree label。

3. **SVG spatial reward 是有效的信号。** GT 树得分 0.17-0.33，随机树 0.0。信号存在——只是我们还没找到有效优化它的方法。

4. **Step-conditioned 模型消灭了 flat tree。** Sequential GRU 模型的 flat tree 率为 0%（BCE 是 66%）。它们产出的是错误的树，但至少是树。

### 原版论文做对了什么

1. **形状相似度天然编码层次。** 相似零件在底层聚类，不同零件在高层聚合。
2. **递归聚类原生建树。** 不需要 sequential 预测。
3. **零参数 = 零过拟合。** 任意数量对象都能工作。
4. **K-Means + silhouette 简单鲁棒。** 不需要 threshold 调优，没有梯度问题。

---

## 4. 代码与结果索引

### 训练脚本

| 脚本 | 模型 | 核心思路 |
|---|---|---|
| `train_tree_planner_baseline.py` | Logistic regression | 最简 baseline：logistic scorer + 贪心 CC |
| `train_tree_planner_nn.py` | MLP(777→192→192→1) | Flat MLP pair scorer，无全局上下文 |
| `train_tree_planner_context.py` | MLP(938→192→192→1) | + 全局上下文特征。BEST BCE 模型 |
| `train_tree_decoder.py` | Transformer encoder | Set-context via self-attention。失败（数据太少） |
| `train_tree_grpo.py` | Same MLP + context | GRPO：策略梯度 + SVG reward |
| `train_step_planner.py` | GRU + step_emb | Sequential TE：teacher forcing 过 tree action。v2：manual_step_groups.connections |
| `train_cluster_planner.py` | PartProjector + K-Means | 对比嵌入学习 + 原版聚类算法 |

### 关键实验报告

| 报告 | 方法 | 最佳 Test Hard |
|---|---|---|
| `context_planner_svg_geometry_composite_report.json` | BCE context MLP + composite | **0.332** |
| `tree_planner_nn_geometry_report.json` | BCE flat MLP geometry | 0.162（最佳无 composite） |
| `grpo_svg_geometry_composite_report.json` | GRPO warm-start | 0.306 |
| `grpo_spatial_scratch_report.json` | GRPO from-scratch + spatial | 0.281 |
| `splitseed_bce100_s0.json` | BCE with split-seed control | 0.332（seed 0） |
| `splitseed_grpo_svgonly_s0.json` | GRPO SVG-only，0% GT | 0.244（seed 0） |
| `label_ratio_sgc_full_summary.md` | Label-ratio 对比 | GRPO > BCE at 10%，25% labels |

---

## 5. 待解决问题

1. **能否在不做 sequential prediction 的情况下利用 step 顺序？** step embedding + BCE 训练保留了 BCE 的泛化能力，同时加入了 step 感知。尚未测试。

2. **能否获得更好的 part 特征？** 原版论文使用 DGCNN（1024 维点云特征）。我们使用手动的 36 维特征。part_images/（754 张渲染图）能否训练出更好的特征提取器？

3. **贪心 CC decoder 是否是瓶颈？** 我们所有模型都用相同的 decoder。换成一个学习的 decoder（RNN/pointer network）即使在当前特征下能否有帮助？

4. **数据增强有多大帮助？** 102 个对象很小。能否从 393 个 manual step 生成合成的装配序列？
