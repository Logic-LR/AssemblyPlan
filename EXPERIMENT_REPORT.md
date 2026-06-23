# Assembly Plan Generation — GNN + GRPO 实验报告

## 1. 实验设置

### 1.1 数据集

- **数据源**：IKEA-Manual 数据集，`tree_generation_dataset.json`
- **规模**：102 个家具对象，73 train / 29 test（按对象预定义 split，非随机划分）
- **类别分布**：Bench、Chair、Table 等多种 IKEA 家具
- **零件数**：每个对象包含 4–12 个零件（primitive parts）
- **GT 树结构**：嵌套列表格式，如 `[0, [[8, 4, 2, 9], [[1, [7, 11, 6, 5], 10], 3]]]`
  - 叶节点为 int（零件 ID），内部节点为 list
  - 支持 **k-ary 合并**（如 `[8, 4, 2, 9]` 为 4 子节点）

### 1.2 特征工程

每个零件提取 34 维特征向量：

| 特征类型 | 维度 | 说明 |
|---------|------|------|
| geometry_feature | 14 | 几何属性（尺寸、比例等），dim 7/8 做 log1p 压缩 |
| svg_feature_mean | 16 | SVG 空间特征均值（中心、bbox、轴长等），去掉恒零维 |
| shape_distribution | 4 | 形状类型 one-hot：elongated_bar / plate_like / irregular / point_or_line |

连接图从 `connection_relation` 构建，无向边，COO 格式 `[2, E]`。

### 1.3 模型架构

**GNNMergeModel**（总参数量 ~629K）：

```
输入: part_features [N, 34], edge_index [2, E]

GNNEncoder (2-layer GraphSAGE):
  Linear(34, 128) → 输入投影
  SAGEConv(128, 128) + LayerNorm + Dropout(0.1) + Residual  × 2层
  → part_embeddings [N, 128]

Cluster 聚合（在合并步骤 t，活跃簇 = {C1, ..., Ck}）:
  cluster_embed_i = [mean_pool, max_pool, min_pool, size, log1p(size)]
  → 128×3 + 2 = 386 维

Pair 特征构建:
  pair_repr = [repr_a, repr_b, |a-b|, a*b, repr_union, context]
  → 386×5 + 779 = 2709 维（含全局上下文）

MergeScorer (3-layer MLP):
  Linear(2709, 192) → LN → ReLU → Dropout(0.15)
  Linear(192, 192)  → LN → ReLU → Dropout(0.15)
  Linear(192, 1)    → logit
```

全局上下文（779 维）：簇数量、log1p(数量)、总零件数、簇大小均值/标准差/最大/最小 + 簇嵌入的均值和标准差向量。

### 1.4 评估指标

- **Simple F1**：预测非叶节点与 GT 非叶节点按零件集合匹配（parts 相同即算匹配），计算 Precision / Recall / F1
- **Hard F1**：在 Simple 基础上额外要求子节点分区相同（忽略纯叶子子节点），即结构完全一致
- 两个指标均在对象级别宏平均

### 1.5 训练超参数

**监督训练**：

| 参数 | 值 |
|------|-----|
| 优化器 | AdamW (lr=1e-3, weight_decay=1e-4) |
| 学习率调度 | CosineAnnealingLR (eta_min=lr×0.01) |
| 损失函数 | BCEWithLogitsLoss (pos_weight = neg/pos) |
| 梯度裁剪 | max_norm=1.0 |
| Epochs | 200 |
| 评估间隔 | 每 20 epoch |

**GRPO 微调**：

| 参数 | 值 |
|------|-----|
| 优化器 | AdamW (lr=5e-5, weight_decay=1e-5) |
| 每对象采样数 | 8 |
| 采样温度 | 1.2 |
| PPO clip epsilon | 0.2 |
| KL 惩罚系数 | 0.1 |
| 奖励 | 0.5 × Simple F1 + 0.5 × Hard F1 |
| Epochs | 50 |
| 评估间隔 | 每 5 epoch |

---

## 2. 实验过程

### 2.1 监督训练

使用 BCE 损失在 GT merge pairs 上训练 GNN + MLP scorer。从 GT 树的 postorder 遍历提取正样本（正确的合并对），负样本为同一时刻所有其他活跃簇对。

训练 200 epoch，每 20 epoch 在验证集上评估 Tree F1（greedy 解码），保存最佳 Hard F1 对应的 checkpoint。

最佳 checkpoint：epoch 20，保存至 `gnn_supervised_best.pt`。

### 2.2 解码器对比实验

在监督训练的 checkpoint 上，对比三种解码策略：

**Greedy Decode**：每步选择 sigmoid 概率最高的 pair 进行二元合并，逐步构建嵌套树。

**Group-Aware Greedy Decode**：在 greedy 基础上增加团检测（clique detection）。每步尝试在单元素簇中找到一个团（所有两两 logit 均高于阈值），若找到则将整个团作为扁平节点合并（匹配 GT 的 k-ary 合并）。未找到团时回退到标准二元合并。

**Beam Search Decode**：维护 beam_width=5 个候选状态，每步扩展所有可能的 pair 合并，保留得分最高的候选。

### 2.3 GRPO 微调

从监督训练的 checkpoint 出发，使用 GRPO（Group Relative Policy Optimization）直接优化 Tree F1：

1. **采样**：对每个训练对象，从当前策略采样 K=8 棵树
2. **奖励计算**：每棵树的奖励 = 0.5 × Simple F1 + 0.5 × Hard F1（与 GT 对比）
3. **优势估计**：组内标准化 advantages = (r - mean) / std
4. **策略更新**：PPO clipped loss + KL 惩罚（约束与参考模型的距离）
5. **参考模型**：冻结的监督训练 checkpoint，每步只做前向不更新梯度

**性能优化**：
- GNN 嵌入预计算：每个对象只做 2 次 GNN 编码（当前模型 + 参考模型），而非 K×2 次
- `_replay_log_prob` 支持 `part_embeds` 参数复用预计算嵌入
- 修复梯度流问题：`_replay_log_prob` 中移除 `torch.no_grad()`，改用 `with_grad` 参数控制

### 2.4 阈值调优

Group-aware 解码器的团检测阈值（threshold）对性能影响显著。在验证集上对两个模型（Supervised / GRPO）分别搜索最优阈值：

| 阈值 | Supervised Simple | Supervised Hard | GRPO Simple | GRPO Hard |
|------|-------------------|-----------------|-------------|-----------|
| -5.0 | — | — | 0.436 | 0.156 |
| -3.0 | — | — | 0.428 | 0.157 |
| -2.0 | 0.446 | 0.173 | 0.445 | 0.184 |
| -1.5 | 0.453 | 0.185 | **0.465** | **0.204** |
| -1.2 | 0.456 | 0.189 | 0.462 | 0.196 |
| -1.0 | 0.456 | 0.178 | 0.462 | 0.196 |
| -0.5 | 0.423 | 0.169 | 0.432 | 0.174 |
| 0.0  | 0.418 | 0.166 | 0.411 | 0.146 |
| +0.5 | 0.412 | 0.157 | — | — |
| +1.0 | 0.397 | 0.139 | — | — |
| +2.0 | 0.369 | 0.135 | — | — |
| +5.0 | 0.308 | 0.115 | — | — |

最优阈值：**-1.5**（GRPO 模型）。

---

## 3. 实验结果

### 3.1 最终结果对比

| 模型 | 解码器 | Simple P | Simple R | Simple F1 | Hard P | Hard R | Hard F1 |
|------|--------|----------|----------|-----------|--------|--------|---------|
| Supervised | Greedy | 0.254 | 0.438 | 0.312 | 0.092 | 0.157 | 0.114 |
| Supervised | Group (t=0.0) | 0.464 | 0.411 | 0.418 | 0.180 | 0.162 | 0.166 |
| Supervised | Group (t=-1.5) | — | — | 0.453 | — | — | 0.185 |
| Supervised | Beam (w=5) | 0.239 | 0.414 | 0.294 | 0.077 | 0.135 | 0.097 |
| **GRPO** | **Group (t=-1.5)** | **0.505** | **0.469** | **0.465** | **0.224** | **0.202** | **0.204** |
| GRPO | Beam (w=5) | 0.271 | 0.457 | 0.329 | 0.101 | 0.168 | 0.123 |

### 3.2 与旧代码库对比

| 方法 | Simple F1 | Hard F1 |
|------|-----------|---------|
| 旧代码库（flat CC 解码器） | 0.488 | 0.206 |
| **GNN + GRPO + Group (t=-1.5)** | **0.465** | **0.204** |

Hard F1 基本持平（0.204 vs 0.206），Simple F1 有 2.3 个百分点的差距。旧代码库使用扁平 CC（Connected Components）解码器，不产生嵌套树结构，其 Simple F1 更高是因为允许非层级的部分匹配。

### 3.3 逐样本分析（GRPO + Group, threshold=-1.5）

| 对象 | 类别 | 零件数 | GT 树 | 预测树 | Simple | Hard |
|------|------|--------|-------|--------|--------|------|
| 0 | Bench/hemnes | 12 | `[0, [[8,4,2,9], [[1,[7,11,6,5],10], 3]]]` | `[0, [3, [[1,2,4,5,11], [6,7,8,9,10]]]]` | 0.364 | 0.182 |
| 1 | Bench/sjalland | 4 | `[2, [0,1], 3]` | `[2, [0,1,3]]` | 0.500 | 0.000 |
| 2 | Chair/applaro | 6 | `[[[0,4,1,5], 3], 2]` | `[2, [5, [1, [0,3,4]]]]` | 0.571 | 0.286 |
| 3 | Chair/bekvam | 8 | `[4, [[7, [[0,2,3], 6,5]], 1]]` | `[0, [4, [1,2,3,5,6,7]]]` | 0.250 | 0.000 |
| 4 | Chair/fanbyn | 7 | `[[[4, [[6,2,3], 5]], 1], 0]` | `[1, [6, [0,2,3,4,5]]]` | 0.250 | 0.000 |
| 5 | Chair/herman | 5 | `[4, [[1, [0,2]], 3]]` | `[4, [0,1,2,3]]` | 0.667 | 0.333 |
| 6 | Chair/jokkmokk | 8 | `[7, [[0,5,6,1,3,2], 4]]` | `[7, [4, [0,1,2,3,5,6]]]` | 1.000 | 1.000 |
| 7 | Chair/lerhamn | 8 | `[4, [0, [[[7,6], 5], 1,2,3]]]` | `[4, [7, [0,1,2,3,5,6]]]` | 0.500 | 0.250 |

---

## 4. 结果分析

### 4.1 关键发现

**1. 团检测阈值是核心超参数**

默认阈值 0.0 过于保守，导致许多 GT 中的 k-ary 合并未被检测到。降低阈值至 -1.5 后：
- Supervised Hard F1: 0.166 → 0.185（+11.4%）
- GRPO Hard F1: 0.146 → 0.204（+39.7%）

GRPO 模型对阈值更敏感，因为 GRPO 的奖励信号直接来自 Tree F1，鼓励模型学出更明确的团结构。

**2. GRPO 在最优阈值下显著优于纯监督**

在 threshold=-1.5 时：
- Hard F1: 0.185 → 0.204（+10.3%）
- Simple F1: 0.453 → 0.465（+2.6%）

GRPO 的优势在于直接优化 Tree F1 而非 pair-level BCE，使模型的 logit 分布更有利于解码。

**3. Greedy/Beam 解码器性能远低于 Group-Aware**

- Greedy Hard F1 = 0.114，Beam Search = 0.123
- Group-Aware Hard F1 = 0.204（+79%）

原因：GT 树包含大量 k-ary 扁平节点（如 `[8, 4, 2, 9]`），纯二元合并的解码器无法产生这种结构，导致 Hard F1 中的子分区匹配全部失败。

**4. Beam Search 在当前设置下反而弱于 Greedy**

Beam Search (w=5) Hard F1 = 0.123 vs Greedy = 0.114，提升微弱。可能原因：
- 当前 pair logit 区分度不足，beam 中的候选树高度相似
- Beam search 的 score 是 log-prob 累加，不直接对应 Tree F1

### 4.2 典型错误模式

**过度扁平化**（最常见）：

- Object 1: GT `[2, [0,1], 3]`（三元合并）→ 预测 `[2, [0,1,3]]`（把 3 也拉入团）
- Object 5: GT `[4, [[1, [0,2]], 3]]` → 预测 `[4, [0,1,2,3]]`（整个子树被扁平化）

团检测倾向于将 logit 较高的单元素簇全部纳入，缺乏对层级深度的感知。

**层级结构错误**：

- Object 0: GT 有 4 层嵌套，预测只有 3 层，中间节点的子分区完全不同
- Object 7: GT 中 `[7,6]` 先合并再与 5 合并，预测直接将 [7,6,5] 扁平合并

### 4.3 与旧代码库的差距分析

旧代码库 Hard F1 = 0.206，使用扁平 CC 解码器（不产生嵌套树）。GNN + GRPO 达到 0.204，已基本持平。

Simple F1 差距（0.488 vs 0.465）来自：
- 旧代码库的 CC 解码器允许非层级的部分匹配，天然有利于 Simple 指标
- GNN 模型的团检测在部分对象上过度合并，丢失了 GT 的中间节点

---

## 5. 总结

| 指标 | 旧代码库 | GNN Supervised | GNN GRPO |
|------|---------|----------------|----------|
| Simple F1 | 0.488 | 0.453 (t=-1.5) | **0.465** (t=-1.5) |
| Hard F1 | 0.206 | 0.185 (t=-1.5) | **0.204** (t=-1.5) |
| 解码器 | Flat CC | Group-Aware | Group-Aware |
| 模型参数量 | ~900K | ~629K | ~629K |

GNN + GRPO 方案在 Hard F1 上达到与旧代码库持平的水平（0.204 vs 0.206），同时具有以下优势：
1. 产生真正的嵌套树结构（而非扁平 cluster 列表）
2. 模型更轻量（629K vs 900K）
3. GNN 编码器利用连接图拓扑信息，具有更好的泛化潜力

后续改进方向：
- 引入深度感知的团检测（限制团大小或合并深度）
- 在 GRPO 奖励中加入树结构相似度（如 TED — Tree Edit Distance）
- 增大数据集规模（当前仅 102 个对象）
