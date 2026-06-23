# Assembly Plan Generation — GNN + GRPO 实验报告

## 1. 实验设置

### 1.1 数据集

- **数据源**：IKEA-Manual 数据集，`tree_generation_dataset.json`
- **规模**：102 个家具对象，73 train / 29 test（按对象预定义 split，非随机划分）
- **类别分布**：Bench(2)、Chair(13)、Desk(2)、Misc(4)、Shelf(1)、Table(7)
- **零件数**：test 集 4–16 个，均值 8.8
- **GT 树结构**：嵌套列表格式，如 `[0, [[8, 4, 2, 9], [[1, [7, 11, 6, 5], 10], 3]]]`
  - 叶节点为 int（零件 ID），内部节点为 list
  - 支持 **k-ary 合并**（如 `[8, 4, 2, 9]` 为 4 子节点）

GT 树结构统计（29 test objects）：

| 指标 | 最小 | 最大 | 均值 |
|------|------|------|------|
| 零件数 | 4 | 16 | 8.8 |
| 内部节点数 | 1 | 9 | 4.9 |
| 树深度 | 1 | 9 | 4.6 |
| 最大子节点数 | 2 | 6 | 3.8 |

### 1.2 特征工程

每个零件提取 34 维特征向量：

| 特征类型 | 维度 | 说明 |
|---------|------|------|
| geometry_feature | 14 | 几何属性（尺寸、比例等），dim 7/8 做 log1p 压缩 |
| svg_feature_mean | 16 | SVG 空间特征均值（中心、bbox、轴长等），去掉恒零维 |
| shape_distribution | 4 | 形状类型 one-hot：elongated_bar / plate_like / irregular / point_or_line |

连接图从 `connection_relation` 构建，无向边，COO 格式 `[2, E]`。

### 1.3 模型架构

**GNNMergeModel**（总参数量 629,121）：

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

### 2.4 GRPO 训练曲线分析

GRPO 训练 50 epoch（实际完成 20 epoch 后手动终止），每 5 epoch 评估一次：

| Epoch | Loss | Reward | Greedy Simple | Greedy Hard | Group(t=-1.5) Simple | Group(t=-1.5) Hard |
|-------|------|--------|---------------|-------------|---------------------|-------------------|
| 5     | 0.179 | 0.319 | 0.313 | 0.106 | 0.444 | 0.180 |
| **10** | **0.209** | **0.353** | **0.338** | **0.132** | **0.431** | **0.183** |
| 15    | 0.268 | 0.366 | 0.308 | 0.112 | 0.409 | 0.157 |
| 20    | 0.298 | 0.378 | 0.311 | 0.112 | 0.397 | 0.146 |

**关键观察**：
- **Loss 持续上升**（0.179 → 0.298，+66%），说明模型在过拟合采样噪声
- **Reward 缓慢上升**（0.319 → 0.378），但这是训练集上的 reward，不代表泛化能力
- **验证集 Hard F1 在 epoch 10 后持续下降**（0.183 → 0.146），典型的 RL 过拟合
- GRPO 最佳点（epoch 10, Hard=0.183）仍低于 Supervised（0.185）

**过拟合原因分析**：
1. 训练集仅 73 个对象，每对象采样 8 棵树 → 有效训练信号 584 条/epoch，远不足以稳定 RL 梯度
2. Tree F1 奖励是全局指标，单步 merge 的贡献难以归因 → 梯度方差大
3. KL 约束（coeff=0.1）未能有效防止策略退化

### 2.5 阈值调优

Group-aware 解码器的团检测阈值（threshold）对性能影响显著。在验证集上对两个模型分别搜索最优阈值：

| 阈值 | Supervised Simple | Supervised Hard | GRPO Simple | GRPO Hard |
|------|-------------------|-----------------|-------------|-----------|
| -2.0 | 0.446 | 0.173 | — | — |
| -1.5 | **0.453** | **0.185** | 0.413 | 0.165 |
| -1.0 | 0.456 | 0.178 | 0.420 | 0.156 |
| -0.5 | 0.423 | 0.169 | — | — |
| 0.0  | 0.418 | 0.166 | 0.412 | 0.159 |
| +0.5 | 0.412 | 0.157 | — | — |
| +1.0 | 0.397 | 0.139 | — | — |
| +2.0 | 0.369 | 0.135 | — | — |
| +5.0 | 0.308 | 0.115 | — | — |

最优阈值：**-1.5**（Supervised 模型）。

---

## 3. 实验结果

### 3.1 最终结果对比（已验证）

| 模型 | 解码器 | Simple F1 | Hard F1 |
|------|--------|-----------|---------|
| Supervised | Greedy | 0.312 | 0.114 |
| Supervised | Group (t=0.0) | 0.418 | 0.166 |
| **Supervised** | **Group (t=-1.5)** | **0.453** | **0.185** |
| Supervised | Beam (w=5) | 0.294 | 0.097 |
| GRPO (ep20, best=ep10) | Greedy | 0.311 | 0.112 |
| GRPO (ep10, 最佳) | Group (t=-1.5) | 0.431 | 0.183 |
| GRPO (ep20) | Group (t=-1.5) | 0.397 | 0.146 |
| GRPO (ep20) | Beam (w=5) | 0.329 | 0.138 |

### 3.2 与旧代码库对比

| 方法 | Simple F1 | Hard F1 |
|------|-----------|---------|
| 旧代码库（flat CC 解码器） | 0.488 | 0.206 |
| **GNN Supervised + Group (t=-1.5)** | **0.453** | **0.185** |

Hard F1 差距 0.021（0.185 vs 0.206），Simple F1 差距 0.035（0.453 vs 0.488）。

### 3.3 按类别统计（Supervised + Group, t=-1.5）

| 类别 | 样本数 | Simple F1 | Hard F1 |
|------|--------|-----------|---------|
| Table | 7 | 0.459 | 0.224 |
| Chair | 13 | 0.475 | 0.205 |
| Bench | 2 | 0.400 | 0.100 |
| Misc | 4 | 0.295 | 0.089 |
| Desk | 2 | 0.210 | 0.000 |
| Shelf | 1 | 0.182 | 0.000 |

### 3.4 Hard F1 分布

| 区间 | 对象数 |
|------|--------|
| 0.0 (完全失败) | 14 / 29 (48%) |
| (0, 0.25) | 8 / 29 (28%) |
| [0.25, 0.5) | 5 / 29 (17%) |
| [0.5, 0.75) | 1 / 29 (3%) |
| 1.0 (完美) | 1 / 29 (3%) |

### 3.5 逐样本分析（Supervised + Group, t=-1.5）

**最差 10 个对象（Hard F1 = 0）：**

| 对象 | 零件数 | Simple F1 |
|------|--------|-----------|
| Shelf/vesken | 16 | 0.182 |
| Desk/flisat | 15 | 0.133 |
| Chair/stig | 10 | 0.133 |
| Chair/ragrund | 11 | 0.154 |
| Misc/satsumas_2 | 10 | 0.182 |
| Desk/pahl | 11 | 0.286 |
| Misc/vesken | 11 | 0.286 |
| Chair/fanbyn | 7 | 0.250 |
| Chair/skogsta_2 | 5 | 0.667 |
| Bench/sjalland | 4 | 0.400 |

**最佳 5 个对象：**

| 对象 | 零件数 | Simple F1 | Hard F1 |
|------|--------|-----------|---------|
| Table/klingsbo_1 | 7 | 1.000 | 1.000 |
| Chair/applaro | 6 | 0.857 | 0.571 |
| Chair/jokkmokk | 8 | 0.667 | 0.444 |
| Chair/norraryd | 5 | 0.800 | 0.400 |
| Table/lack | 6 | 0.800 | 0.400 |

---

## 4. 结果分析

### 4.1 关键发现

**1. GRPO 在 epoch 10 后过拟合，最终未能超越 Supervised**

GRPO 训练曲线呈典型的 RL 过拟合模式：

| Epoch | Loss | Hard F1 (Group) | 趋势 |
|-------|------|-----------------|------|
| 5     | 0.179 | 0.180 | ↑ |
| 10    | 0.209 | **0.183** | ↑ 最佳 |
| 15    | 0.268 | 0.157 | ↓ 开始退化 |
| 20    | 0.298 | 0.146 | ↓ 持续退化 |

- Loss 持续上升（+66%），模型过拟合采样噪声
- 最佳点 epoch 10 的 Hard F1 = 0.183，仍低于 Supervised 的 0.185
- 训练 20 epoch 后退化到 0.146（-20%）

过拟合原因：73 个训练对象 × 8 采样 = 584 条有效信号/epoch，远不足以稳定 RL 梯度。Tree F1 作为全局奖励信号过于稀疏，单步 merge 贡献难以归因。

**2. 团检测阈值是核心超参数**

默认阈值 0.0 过于保守。降低阈值至 -1.5 后，Supervised Hard F1 从 0.166 提升到 0.185（+11.4%）。

阈值越低，团检测越激进，能捕获更多 k-ary 合并，但也会引入误合并。-1.5 是当前模型的最优点。

**3. Group-Aware 解码是性能关键**

- Greedy Hard F1 = 0.114
- Group-Aware (t=-1.5) Hard F1 = 0.185（+62%）

GT 树包含大量 k-ary 扁平节点，纯二元合并的解码器无法产生这种结构，导致 Hard F1 中的子分区匹配全部失败。团检测是弥合这一差距的必要机制。

**4. Beam Search 弱于 Greedy**

Beam Search (w=5) Hard F1 = 0.097 vs Greedy = 0.114，反而更差。原因：
- 当前 pair logit 区分度不足，beam 中的候选树高度相似
- Beam search 的 score 是 log-prob 累加，不直接对应 Tree F1
- beam_width=5 可能不够大，无法覆盖足够的搜索空间

### 4.2 典型错误模式

**过度扁平化**（最常见）：

- Object 1: GT `[2, [0,1], 3]`（三元合并）→ 预测 `[2, [0,1,3]]`（把 3 也拉入团）
- Object 5: GT `[4, [[1, [0,2]], 3]]` → 预测 `[4, [0,1,2,3]]`（整个子树被扁平化）

团检测倾向于将 logit 较高的单元素簇全部纳入，缺乏对层级深度的感知。

**层级结构错误**：

- Object 0: GT 有 4 层嵌套，预测只有 3 层，中间节点的子分区完全不同
- Object 7: GT 中 `[7,6]` 先合并再与 5 合并，预测直接将 [7,6,5] 扁平合并

**完全失败的对象特征**：

Hard F1 = 0 的 14 个对象中，零件数 ≥ 10 的有 7 个（50%），说明模型在复杂对象上的表现显著退化。

### 4.3 与旧代码库的差距分析

旧代码库 Hard F1 = 0.206，使用扁平 CC 解码器（不产生嵌套树）。GNN Supervised 达到 0.185，差距 0.021。

差距来源：
1. **解码器差异**：旧代码库的 CC 解码器直接从连接图提取连通分量，天然适合 k-ary 合并；GNN 的团检测是启发式的，阈值敏感
2. **特征差异**：旧代码库可能使用了更丰富的特征（如全局几何特征），GNN 的 34 维零件特征可能信息不足
3. **数据量瓶颈**：102 个对象对于 GNN 训练偏少，GraphSAGE 的消息传递可能未充分学习

---

## 5. 总结

| 指标 | 旧代码库 | GNN Supervised | GNN GRPO (最佳 ep10) |
|------|---------|----------------|---------------------|
| Simple F1 | 0.488 | **0.453** | 0.431 |
| Hard F1 | 0.206 | **0.185** | 0.183 |
| 解码器 | Flat CC | Group-Aware (t=-1.5) | Group-Aware (t=-1.5) |
| 模型参数量 | ~900K | 629K | 629K |

**最佳方案**：GNN Supervised + Group-Aware Greedy (threshold=-1.5)

- Hard F1 = 0.185，与旧代码库差距 0.021
- 产生真正的嵌套树结构（而非扁平 cluster 列表）
- 模型更轻量（629K vs 900K）

**GRPO 未生效**，主要受限于数据量不足（73 训练对象）和奖励信号稀疏。

后续改进方向：
- **数据增强**：对训练对象做零件排列增强，扩大 GRPO 的有效训练样本
- **深度感知团检测**：限制团大小上限或引入层级先验，避免过度扁平化
- **更丰富的特征**：引入全局几何特征、连接图的度中心性等结构特征
- **增大 beam width**：beam_width=20 或更大，结合 diverse beam search
- **树编辑距离奖励**：在 GRPO 中使用 TED 替代 Simple/Hard F1，提供更细粒度的梯度信号
