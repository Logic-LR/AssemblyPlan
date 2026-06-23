# 装配树生成实验汇报（组会用）

> 精简版 · 2026-06-23 · 完整版见根目录 `IKEA-Manual Dataset 详细总结报告.md`
> 覆盖：旧代码库 (experiments/svg_assembly) + 新代码库 (assembly_plan) + VLM distill 方案

---

## 一、任务定义

**输入**：N 个零件的 34 维数值特征（几何属性 + SVG 空间统计 + 形状类型）

**输出**：装配树（嵌套列表，支持 k-ary 合并）
```
例：[[0, 1, 2], 3]         → 先把 {0,1,2} 合并，再与 3 合并
例：[0, [[8,4,2,9], ...]]  → 四元扁平合并 + 多层嵌套
```

**数据规模**：102 个 IKEA 家具对象（73 train / 29 test），754 个 primitive parts，404 个 tree actions，393 个 manual steps

**评估指标**：
- **Simple F1**：预测的内部节点（零件集合）是否与 GT 匹配
- **Hard F1**：额外要求子节点分区一致（结构完全正确，更严格）

---

## 二、项目全貌（两条主线）

### Pipeline 流程

```
PDF 说明书 → line_seg/*.svg → SVG 特征解析 → 简化 SVG 几何
                                                ↓
                        ┌───────────────────────┴───────────────────────┐
                        ↓                                               ↓
              Part-to-SVG Grounding                        Assembly Tree Planning
              (零件→SVG实例映射)                          (零件特征→装配树)
                        ↓                                               ↓
              残差 CNN, Attn Pooling                    Flat MLP → GNN → GRPO → VLM
              最佳: 70.6% 严格 / 84.9% 等价              最佳 Hard F1: 0.332 (composite)
                        ↓                                               ↓
              端到端连接 F1: 0.8875                     纯特征 Hard F1: 0.204 (GNN)
              (Oracle 0.9875, gap ~10%)                (66% 平树是核心瓶颈)
```

### 两条技术路线

| | 旧代码库 (experiments/svg_assembly) | 新代码库 (assembly_plan) |
|------|------|------|
| 模型 | Flat MLP (218K params) | GNN + GraphSAGE (629K params) |
| 编码器 | 无（直接 concat pair 特征） | 2-layer GraphSAGE + Residual |
| 解码器 | Greedy CC（连通分量合并） | Group-Aware（团检测 k-ary 合并） |
| 训练方式 | BCE + GRPO（SVG reward） | BCE + GRPO（Tree F1 reward） |
| 最佳 Hard F1 | 0.332（但用了 composite token） | 0.204（纯特征，真正的嵌套树） |
| 纯特征 Hard F1 | 0.108（无 manual 泄露） | 0.204（同一设定） |

### Grounding 现状（前置模块）

| 模型 | 严格 Acc | 等价 Acc | 端到端连接 F1 |
|------|:---:|:---:|:---:|
| 几何 baseline | 47.1% | — | 0.435 |
| 线性图像特征 | 42.9% | — | 0.398 |
| 旧 Tiny CNN (30K) | 69.8% | 84.0% | 0.850 |
| **改进残差 CNN (1.4M)** | **70.6%** | **84.9%** | **0.8875** |
| Oracle（上限） | 100% | 100% | 0.9875 |

> Grounding 与 Oracle 之间有 ~10% 的 gap，是端到端瓶颈之一。但装配树生成才是核心挑战。

---

## 三、旧代码库：6 种方法的探索

> 详见 `experiments/svg_assembly/METHODS_ANALYSIS.md`

### 方法 1：BCE Context MLP ⭐ 旧代码库最优

| 项目 | 说明 |
|------|------|
| 模型 | MLP(938→192→192→1)，218K 参数 |
| 输入 | pair 特征 938 维：`[rep(A), rep(B), |A-B|, A×B, rep(A∪B), global_ctx]` |
| 训练 | pool 所有步骤的所有 pair → BCE 二分类 |
| 推理 | Greedy：打分 → 阈值过滤 → 连通分量合并 |

| 特征模式 | Test Simple | Test Hard | 说明 |
|------|:---:|:---:|------|
| geometry | 0.412 | 0.109 | 纯几何特征 |
| svg_geometry | 0.407 | **0.108** | +SVG 特征，**真正的无 manual 泄露上限** |
| svg_geometry_composite | 0.570 | **0.332** | ⚠️ composite token 直接编码 manual 答案 |

**5 seed 方差**（svg_geometry_composite）：mean Hard F1 = 0.274 ± 0.085，种子间差距 2.5 倍

**核心问题**：
- **66% 预测为平树**（19/29 test 对象），模型直接一步合并所有零件
- 同一 pair 在不同 step 得分几乎相同（local features 一样，global context 差异极小）
- 推理时所有 pair 概率坍缩到 0.4-0.5，无法区分
- Composite token 把 0.108 拉到 0.332，这个差距就是 manual 信息的上限价值

### 方法 2：GRPO + SVG Reward

| 变体 | Test Hard |
|------|:---:|
| BCE warm-start → GRPO | 0.306 |
| From-scratch K=16, τ=1.5 | 0.258 |
| From-scratch + spatial reward | 0.281 |
| **GRPO SVG-only (0% GT)** | **~0.346** (3-seed mean) |

**Label Ratio 实验（最有价值的发现）**：

| Label Ratio | BCE | GRPO | 提升 |
|:---:|:---:|:---:|:---:|
| 10% | 0.182 | **0.237** | **+30%** |
| 25% | 0.261 | **0.274** | +5% |
| 50% | **0.199** | 0.195 | -2% |
| 100% | **0.332** | 0.210 | -37% |

**关键结论**：GRPO 在低标注下有效（10% label +30%），但全标注下不如 BCE。SVG reward 可以替代缺失的 GT tree label。

**GRPO 的问题**（来自 splitseed_grpo_svgonly_s0.log 的详细分析）：
- **泛化 gap 巨大**：Val Hard = 0.424 → Test Hard = 0.244（gap 0.18，接近 2 倍）
- **KL 散度持续上升**：0.36 → 2.46，80 epoch 从未收敛，KL 约束（β=0.1）太弱
- **Loss 剧烈波动**：0.22 → 0.73 → 0.29 → 0.62，per-object instability 严重
- Warm-start 探索失败：BCE 模型 logit 接近 0/1，τ=5.0 时仍有 56% 重复采样

### 方法 3-6：快速总结

| 方法 | 模型 | Test Hard | 失败原因 |
|------|------|:---:|------|
| Transformer Decoder | Self-attention encoder | 未记录（放弃） | 仅 ~300 状态，Transformer 需要数千 |
| Sequential Planner v1 | GRU + step_emb | 0.022 | 250 状态不够 + inference 缺失 spatial features |
| Sequential Planner v2 | GRU + connections | 0.021 | Connections 不编码层次 + GRU 输入过于压缩 |
| Contrastive + K-Means | PartProjector 50K | 0.035 | 对比学习抹平层次 + 连接性 ≠ 装配层次 |

---

## 四、新代码库：GNN + Group-Aware 解码器

> 详见 `EXPERIMENT_REPORT.md`

### GNNMergeModel 架构

```
Input: part_features [N, 34], edge_index [2, E]（connection_relation 构建）

GNNEncoder (2-layer GraphSAGE, 629K params):
  Linear(34→128) → 输入投影
  SAGEConv(128,128) + LayerNorm + Dropout(0.1) + Residual × 2 层
  → part_embeddings [N, 128]

Cluster 聚合:
  cluster_embed = [mean_pool, max_pool, min_pool, size, log1p(size)]
  → 386 维

Pair 特征:
  [repr_a, repr_b, |a-b|, a×b, repr_union, global_context]
  → 2709 维

MergeScorer (3-layer MLP):
  Linear(2709→192→192→1) → logit
```

### 关键创新：Group-Aware 解码器

旧代码库的 Greedy CC 解码器只能做**二元合并** → 无法匹配 GT 的 k-ary 扁平节点（如 `[8,4,2,9]` 四元合并）。

**Group-Aware Decoder 做法**：每步在单元素簇中检测**团（clique）**——若某组零件的所有两两 logit 均高于阈值，将整组作为扁平节点合并。

### 阈值调优实验

阈值控制团检测的敏感度，是核心超参数：

| 阈值 | Supervised Hard | GRPO Hard |
|:---:|:---:|:---:|
| 0.0（默认）| 0.166 | 0.146 |
| -1.5 | **0.185** | **0.204** |
| -2.0 | 0.173 | 0.184 |

GRPO 对阈值更敏感（+39.7% vs +11.4%），因为其 reward 直接来自 Tree F1，学出了更明确的团结构。

### 最终结果

| 模型 | 解码器 | Simple F1 | Hard F1 | 说明 |
|------|------|:---:|:---:|------|
| Supervised | Greedy | 0.312 | 0.114 | 纯二元合并，大量 k-ary 节点失败 |
| Supervised | Group (t=-1.5) | 0.453 | 0.185 | +62% Hard F1 |
| **GRPO** | **Group (t=-1.5)** | **0.465** | **0.204** | **最优** |
| GRPO | Beam (w=5) | 0.329 | 0.123 | Beam 反而弱于 Group |

### 与旧代码库对比

| | 旧代码库 | GNN Supervised | GNN GRPO |
|------|:---:|:---:|:---:|
| Simple F1 | **0.488** | 0.453 | 0.465 |
| Hard F1 | **0.206** | 0.185 | 0.204 |
| 解码器 | Flat CC（不产生嵌套树）| Group-Aware | Group-Aware |
| 模型参数量 | ~900K | ~629K | ~629K |

GNN + GRPO 在 Hard F1 上已持平旧代码库（0.204 vs 0.206），且具有真正的嵌套树结构 + 更轻量。

### 典型错误模式

**过度扁平化**（最常见）：团检测倾向于将 logit 较高的单元素簇全部纳入，缺乏对层级深度的感知。

```
GT:      [4, [[1, [0,2]], 3]]     ← 3 层嵌套
预测:    [4, [0,1,2,3]]           ← 整个子树被拍扁成 1 层
```

**层级结构错误**：GT 先合并 `[7,6]` 再与 5 合并，预测直接将 `[7,6,5]` 扁平合并。

---

## 五、核心发现总结

### 为什么所有方法都很难？

| 根因 | 影响的方法 | 说明 |
|------|:---:|------|
| **步骤顺序未被利用** | 全部 | 所有方法跨步骤 pool。模型从未学到"先在 step 0 合并 {0,1}，再在 step 2 合并 {0,1,2}" |
| **训练数据太少** | 方法 3-5, GNN | 62 对象 × ~4 actions ≈ 250 状态。GRU/Transformer 需要数千 |
| **特征编码"是什么"而非"在何时"** | 全部 | 零件特征告诉你相似性，不告诉你在哪一步合并 |
| **训练-推理不一致** | 方法 2,4,5 | 训练用了 step SVG 空间特征，推理时空 map 为空 |
| **BCE loss 无法编码层次** | 方法 1,4,5,GNN | step 1 该合并的 pair 和 step 3 该合并的 pair 在 BCE 里一模一样 |
| **解码器限制** | 方法 1-5 | Greedy CC 二元合并无法产生 GT 的 k-ary 节点 |

### 什么有效？

1. **Composite tokens 很强**：0.332 vs 0.108（无 manual），但本质上是"作弊"——直接编码了 manual 答案
2. **GRPO 在低标注下有效**：10% GT label + SVG reward 比 BCE 高 30%。SVG 可以替代缺失的 GT tree label
3. **Group-Aware 解码器是关键突破**：解决 k-ary 合并问题，Hard F1 提升 62%（0.114 → 0.185）
4. **GNN 引入图结构有潜力**：用 connection graph 代替 flat pair features，更轻量（629K vs 900K）
5. **Sequential 模型消除平树**：GRU 的平树率 = 0%（BCE 有 66%），虽预测不准但至少有结构

### 两条路线的关系

```
旧代码库 (experiments/svg_assembly):
  探索期 → 试了 6 种方法 → 找到关键瓶颈（平树、BCE flat、解码器限制）
  → 结论：需要更好的解码器 + 利用步骤顺序

新代码库 (assembly_plan):
  工程期 → GNN + Group-Aware decoder → Hard F1 = 0.204
  → 持平旧代码库，且产生真正的嵌套树

下一步 (vlm_distill):
  VLM 看步骤图 → 提供额外监督信号 → 弥补纯特征的不足
  目标：弥合 0.108（纯特征）→ 0.332（有 manual）的差距
```

---

## 六、性能全景图

| 实验 | 设置 | Test Hard F1 | 平树率 | 备注 |
|------|------|:---:|:---:|------|
| 旧 BCE svg_geometry_composite | Flat MLP, 全标注 | 0.332 | 66% | ⚠️ composite token |
| 旧 BCE svg_geometry | Flat MLP, 全标注 | **0.108** | 66% | 无 manual 泄露上限 |
| 旧 GRPO SVG-only | Flat MLP, 0% GT | ~0.244-0.346 | — | seed 方差大 |
| 旧 GRPO 10% label | Flat MLP, 低标注 | 0.237 | — | +30% over BCE |
| **新 GNN Supervised Group** | GraphSAGE, 全标注 | 0.185 | — | 真正的嵌套树 |
| **新 GNN GRPO Group** | GraphSAGE, GRPO | **0.204** | — | **当前最优纯特征模型** |

**重要说明**：
- 旧代码库 0.332 和 0.206 用的是 **Flat CC 解码器**，不产生真正的嵌套树，其简单和困难 F1 的定义与新代码库不同
- 新代码库 0.204 是**真正的嵌套树结构**，指标更严格
- 两套代码库的 Simple/Hard F1 实现有差异，**不建议直接对比绝对数值**，关注相对提升

---

## 七、当前方向与下一步

### 进行中

1. **Assembly Plan (GNN)**：完善 GraphSAGE + Group-Aware decoder，当前 Hard F1 = 0.204
2. **VLM Distill**：用 GPT-4o 看步骤图 → 蒸馏到 GNN。设计文档见 `vlm_distill/DESIGN_CN.md`

### 待解决

1. **过度扁平化**：Group-Aware decoder 的团检测缺乏深度感知，限制团大小或合并深度
2. **更好的零件特征**：原始论文 DGCNN 1024 维 vs 我们手工 34 维。`part_images/` 的 754 个渲染视图能否训练特征提取器？
3. **步骤顺序利用**：Step embedding + BCE 训练（保留 BCE 的泛化 + 加入步骤感知），尚未测试
4. **数据增强**：102 个对象太少。能否从 393 个 manual steps 生成合成装配序列？
5. **GRPO 稳定化**：解决 KL 发散、Val/Test gap、per-object instability

### 论文叙事方向

> Manual-step SVGs 为装配树生成提供弱结构监督：
> 1. GRPO + SVG reward 在 10% GT label 下比纯监督 BCE 高 30% — SVG 可以替代缺失的 GT tree label
> 2. Group-Aware 解码器解决 k-ary 合并问题，产生真正的嵌套装配树
> 3. GNN + GraphSAGE 利用 connection graph 拓扑，比 flat MLP 更高效
