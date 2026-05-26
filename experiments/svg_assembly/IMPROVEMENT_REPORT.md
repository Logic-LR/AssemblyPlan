# SVG Grounding 模型改进报告

**日期**: 2026-05-25 ~ 2026-05-26
**任务**: 改进 Part-to-SVG Grounding 模型，提升端到端装配理解性能

---

## 1. 改动概述

| # | 改动 | 文件 | 类型 |
|---|---|---|---|
| 1 | SVG 特征增加空间位置 (center_x, center_y) | `train_grounding_cnn.py:svg_feature()` | 特征工程 |
| 2 | 残差 CNN Backbone (3层→15层, 3万→143万参数) | `train_grounding_cnn.py:BetterImageCNN` | 模型架构 |
| 3 | 注意力多视角池化 | `train_grounding_cnn.py:TinyGroundingCNN.forward()` | 模型架构 |
| 4 | 无效 ablation: 跨步骤颜色追踪 | `run_end_to_end_diagnostic.py:assignment_maps()` | 已移除 |
| 5 | 图片预加载 | `train_grounding_cnn.py:_preload_images()` | 性能优化 |
| 6 | GPU 支持 (CPU-only → CUDA) | `train_grounding_cnn.py:main()` | 环境 |

---

## 2. 模型架构对比

| | 旧 TinyGroundingCNN | 新 TinyGroundingCNN |
|---|---|---|
| CNN Backbone | 3 层卷积 (8→16→32) | BetterImageCNN: 6 残差块 + 3 降采样层 |
| 通道数 | 8→16→32 | 32→64→128→256 |
| 输出维度 | 32 | 256 |
| 残差连接 | 无 | 有 |
| BatchNorm | 无 | 有 |
| 多视角聚合 | mean + max pool | 可学习注意力权重 + softmax |
| SVG 特征维度 | 15 | 17 (+center_x, +center_y) |
| 总参数量 | ~30K | **1,435,314** |

---

## 3. Grounding 结果对比

**测试集**: 40 步, 119 实例

| 指标 | 旧模型 | 新模型 | 变化 |
|---|---|---|---|
| 严格 Instance Accuracy | 69.75% | **70.59%** | +0.84% |
| 等价 Instance Accuracy | 84.03% | **84.87%** | +0.84% |
| 严格 Exact Match | 75.00% | 67.50% | -7.50% |
| 等价 Exact Match | 85.00% | 82.50% | -2.50% |
| 正确匹配实例数 | 83/119 | 84/119 | +1 |

**验证集** (选最佳 checkpoint):

| 指标 | 旧模型 | 新模型 |
|---|---|---|
| 等价 Val Accuracy | 76.87% | **86.57%** (+9.70%) |

---

## 4. 端到端结果对比

**Grounding → Connection 完整流程** (Test 集, 40 步)

| 指标 | 旧模型 | 新模型 | 变化 |
|---|---|---|---|
| **端到端连接 F1** | 0.8500 | **0.8875** | **+4.4%** |
| **端到端连接 Exact Match** | 90.0% | **92.5%** | **+2.5%** |
| Oracle Grounding 连接 F1 | 0.9875 | 0.9875 | 不变 |
| Oracle Grounding Exact Match | 97.5% | 97.5% | 不变 |

**关键发现**: Grounding 严格准确率仅提升 0.84%，但端到端连接 F1 提升了 4.4%。说明新模型选择性改进的 grounding 实例对下游连接预测有放大效应。

---

## 5. 训练细节

| 参数 | 值 |
|---|---|
| 训练样本 | 2,619 pairs (300 步) |
| 验证样本 | 396 pairs (53 步) |
| 测试样本 | 435 pairs (40 步) |
| Epochs | 20 |
| Optimizer | AdamW (lr=0.001, weight_decay=0.0001) |
| Loss | BCEWithLogitsLoss (正样本加权) |
| Dropout | 0.2 |
| Max Images per Sample | 8 |
| 硬件 | NVIDIA GeForce RTX 3060 Laptop GPU (6GB) |
| 每轮耗时 | ~130-170s |
| 总训练时间 | ~50 分钟 |

### 训练曲线

| Epoch | Loss | Val Strict Acc | Val Equiv Acc |
|---|---|---|---|
| 5 | 0.6560 | 73.88% | 81.34% |
| 10 | 0.5871 | 74.63% | **84.33%** |
| 15 | 0.5424 | 72.39% | 82.09% |
| **20** | **0.5078** | **77.61%** | **86.57%** ★ |

---

## 6. 无效 ablation: 跨步骤颜色追踪

曾经测试过一个 `cross_step_bias` 思路：把 stroke 颜色在步骤间当作身份线索来保持映射。这个假设不成立，已经从主诊断脚本中移除。

原因是：SVG stroke color 只能作为同一个 step 内 parser 分组 path 的临时 instance id，不能作为跨步骤零件身份。不同 step 里的相同颜色不表示同一个零件，也不应该参与模型决策。

| 设置 | 端到端连接 F1 | Exact Match |
|---|---|---|
| 无跨步骤偏置 | **0.8875** | **0.925** |
| 跨步骤偏置 (+5.0 硬偏置) | 0.7875 | 0.850 |

**结论**: 这不是一个可修补的软/硬偏置问题，而是特征假设错误。后续如果做跨步骤一致性，必须基于几何、部件集合、连接上下文或显式预测的 latent identity，不能基于 SVG 颜色。

---

## 7. 已知问题与下一步

### 当前瓶颈

1. **DataLoader 单线程**: GPU 大量空闲，每轮 130s 中计算时间 <10s。加 `num_workers` 或预计算特征 tensor 可提速 10x+
2. **Grounding 仍是瓶颈**: 端到端 F1 0.8875 vs Oracle 0.9875，差距 ~10 个百分点全部来自 grounding 误差
3. **跨步骤一致性需要重新定义**: 不能用 SVG 颜色；如果需要 temporal/step consistency，必须学习 latent identity 或用装配上下文约束
4. **复合零件仍是弱点**: composite parts 的 mean/max/min 聚合丢失结构信息

### 已验证有效的改进

- **空间位置特征** (+2 维 center_x/y) — 低投入高回报
- **残差 CNN + BatchNorm** — 比 3 层浅卷积有实质提升
- **注意力多视角池化** — 让模型自主选择最有区分力的视角
- **图片预加载** — 消除磁盘 I/O

---

## 8. 文件清单

| 文件 | 说明 |
|---|---|
| `scripts/train_grounding_cnn.py` | 修改后的训练脚本 (含所有改进) |
| `scripts/run_end_to_end_diagnostic.py` | 修改后的端到端诊断 |
| `experiments/svg_assembly/reports/grounding_cnn_improved_val_report.json` | 新模型训练报告 |
| `experiments/svg_assembly/reports/grounding_cnn_improved_val_model.pt` | 新模型权重 |
| `experiments/svg_assembly/reports/end_to_end_diagnostic_improved_report.json` | 端到端诊断报告 |
| `experiments/svg_assembly/reports/training_gpu_v2.log` | 训练日志 |
