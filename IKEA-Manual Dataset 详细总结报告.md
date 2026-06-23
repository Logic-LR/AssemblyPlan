# IKEA-Manual Dataset 项目详细技术总结报告

> 生成时间：2026-05-27
> 覆盖目录：`D:\Docu\LLM-RAG\dataset`
> 排除目录：`.venv`、`__MACOSX`

---

## 目录

1. [项目概述](#1-项目概述)
2. [数据集结构详解](#2-数据集结构详解)
3. [Code 模块详解](#3-code-模块详解)
4. [SVG Assembly 实验模块详解](#4-svg-assembly-实验模块详解)
5. [Part-to-SVG Grounding 详细分析](#5-part-to-svg-grounding-详细分析)
6. [装配树生成：所有方法与结果](#6-装配树生成所有方法与结果)
7. [GRPO 强化学习详解](#7-grpo-强化学习详解)
8. [实验报告数据汇总](#8-实验报告数据汇总)
9. [关键发现与根因分析](#9-关键发现与根因分析)
10. [文件依赖关系与项目时间线](#10-文件依赖关系与项目时间线)
11. [待解决问题与下一步计划](#11-待解决问题与下一步计划)

---

## 1. 项目概述

### 1.1 项目目标

本项目围绕 **IKEA-Manual Dataset** 展开，核心目标是：**利用宜家家具说明书中的 SVG 矢量图和装配步骤监督信息，改进装配树（Assembly Tree）生成任务**。

具体推理流程为：

```
观察到的真实零件
→ 识别/接地（grounding）：将每个零件映射为一个简化 SVG 或规范化的零件 token
→ 将所有零件 token 输入训练好的规划器（planner）
→ 生成装配树
```

### 1.2 数据集规模

| 项目 | 数量 |
|---|---|
| 家具对象总数 | 102 |
| 训练/验证/测试划分 | 73 / 11 / 29 |
| 原始零件（primitive parts）总数 | 754 |
| 装配树动作（tree actions）总数 | 404 |
| 说明书步骤（manual steps）总数 | 393 |
| 复合/子装配 token（composite tokens） | 302 |
| Grounding 样本数 | 1056（原始 754 + 复合 302） |
| Grounding 训练/测试样本 | 937 / 119 |

### 1.3 技术路线概览

项目技术路线分为两条主线：

1. **Part-to-SVG Grounding**：将观察到的零件（图像/几何）映射到说明书 SVG 实例
2. **Assembly Tree Planning**：基于零件 token（几何 + SVG 特征）预测装配树

---

## 2. 数据集结构详解

### 2.1 目录结构

```
D:\Docu\LLM-RAG\dataset\
├── README.md                  # 数据集核心说明
├── main_data.json             # 核心数据 JSON（所有对象标注）
├── pdfs\                     # 说明书 PDF 文件
├── code\                     # 实验代码
│   ├── README.md
│   ├── manual_generation\    # 说明书生成实验
│   ├── part_assembly\        # 零件装配（基于 PartNet 的 3D 装配）
│   ├── resources\
│   └── utils\
├── line_seg\                 # SVG 矢量线条分割结果
├── seg\                      # 像素级分割
├── parts\                    # 装配零件分解
├── scripts\                  # SVG 装配实验的核心脚本
│   ├── README_svg_features.md  # SVG 特征构建详细说明
│   ├── build\                # 数据构建和特征提取脚本（7 个）
│   ├── train\                # 模型训练脚本（12 个）
│   ├── eval\                 # 评估和推理脚本（5 个）
│   └── export\               # 导出和分析报告脚本（4 个）
└── experiments\
    └── svg_assembly\        # SVG 感知装配理解实验
        ├── README.md
        ├── PROJECT_TASKS.md  # 项目任务追踪
        ├── IMPROVEMENT_REPORT.md  # Grounding 改进报告
        ├── METHODS_ANALYSIS.md
        ├── METHODS_ANALYSIS_CN.md
        ├── METHODS_ANALYSIS_CN_20260527_105611_616.md
        ├── reports\           # 所有实验报告 JSON/Markdown
        ├── datasets\         # 构建的数据集
        ├── simplified_svg\   # 简化 SVG 几何输出
        └── predicted_assembly_trees\  # 预测装配树输出
```

### 2.2 `main_data.json` 数据结构

`main_data.json` 是每个对象的核心标注文件，结构如下：

**对象级字段：**

| 字段 | 类型 | 说明 |
|---|---|---|
| `category` | string | 家具类别（Bench, Chair, Table 等） |
| `name` | string | 对象名称（如 `applaro`） |
| `steps` | list | 每个装配步骤的标注（见下方） |
| `connection_relation` | list | 原始零件之间的连接关系 |
| `geometric_equivalence_relation` | list | 几何等价关系（如四条桌腿等价） |
| `assembly_tree` | list | 树形结构的装配计划 |
| `parts_ct` | int | 原始零件数量 |

**步骤级字段（`steps` 中每个元素）：**

| 字段 | 类型 | 说明 |
|---|---|---|
| `parts` | list[string] | 本步骤涉及的装配零件（如 `['0,1,2', '3']`） |
| `connections` | list[list] | 成对连接关系（如 `[['0,1,2', '3']]`） |
| `page_id` | int | 本步骤在说明书中的页码 |
| `step_id` | int | 步骤 ID（本对象内） |
| `masks` | list | 每个零件的掩码（用 `pycocotools.masks.decode` 解码为 numpy 数组） |
| `intrinsics` | list | 每个零件的内在矩阵 |
| `extrinsics` | list | 每个零件的外在矩阵 |
| `part_segmentation_split` | string | 分割任务的划分（`train`/`val`/`test`） |
| `step_id_global` | int | 在整个数据集中的步骤索引 |

### 2.3 数据文件说明

- **`line_seg/`**：存放从 PDF 说明书中提取的 SVG 矢量图，每个 step 一个 SVG 文件，stroke 颜色代表视觉实例 ID
- **`seg/`**：像素级分割掩码，与 `main_data.json` 中的 `masks` 字段对应
- **`parts/`**：零件分解结果
- **`pdfs/`**：原始说明书 PDF

---

## 3. Code 模块详解

### 3.1 `code/README.md`

`code/` 目录包含两个主要实验方向：

1. **Manual Plan Generation**：运行 `bash manual_generation/test.sh` 启动实验
2. **Part Assembly**：参见 `part_assembly/` 子目录下的 `README.md`

### 3.2 `code/part_assembly/README.md` — Generative 3D Part Assembly

本模块是 NeurIPS 2020 论文 **"Generative 3D Part Assembly via Dynamic Graph Learning"** 的官方实现，由 Jialei Huang、Guanqi Zhan 等人完成。

**核心思想**：通过动态图神经网络，以零件点云为输入，进行 5 次迭代的图消息传递，实现从粗到细的零件装配优化。

**动态图学习框架要点**：
- 图结构从零件位姿估计中推理出来，同时位姿也随更新后的零件关系而演化（双重迭代优化）
- 每隔一个时间步，将几何等价的零件（如椅子的两个扶手）聚合为单一节点，在稀疏节点集合上进行图学习，再反池化回密集节点集合

**文件结构**：
```
data/
  partnet_dataset/        # PartNet 数据（如需重新制备数据则需要）
prepare_data/
  Chair.{train,val,test}.npy  # 数据列表文件
  prepare_shape.py              # 制备数据
  prepare_contact_points.py     # 制备接触点数据
exps/
  utils/                       # 工具函数
  dynamic_graph_learning/
    logs/              # 检查点和 tensorboard 文件
    models/            # 模型文件
    scripts/           # 训练/测试脚本
    data_dynamic.py    # 数据加载
    test_dynamic.py    # 测试代码
    train_dynamic.py   # 训练代码
```

**依赖环境**：Ubuntu 16.04 + CUDA 10.0 + GCC 7.5 + Python 3.7 + PyTorch 1.1

**快速开始**：
```bash
conda env create -f environment.yaml
. activate PartAssembly
cd exps/utils/chd
python setup.py build
# 训练
cd exps/dynamic_graph_learning/scripts/
./train_dynamic.sh
# 测试
./test_dynamic.sh
```

**与本项目的关系**：此模块是 3D 零件装配的另一种技术路线（基于点云和动态图），与本项目关注的 SVG 感知装配树生成是并列关系，共享 PartNet 数据集基础。

### 3.3 `code/part_assembly/exps/utils/` 子模块

| 子模块 | README 内容 | 用途 |
|---|---|---|
| `cd/` | Chamfer Distance CUDA 实现，运行 `python setup.py install` 安装 | 点云距离计算 |
| `chamfer/` | 同上（重复） | 点云距离计算 |
| `emd/` | Earth Mover Distance CUDA 实现，运行 `python setup.py install` 安装 | 点云匹配计算 |

---

## 4. SVG Assembly 实验模块详解

### 4.1 实验 Pipeline 全流程

`experiments/svg_assembly/README.md` 定义了当前实验的完整 pipeline：

**Step 1：SVG 特征解析**
```powershell
python scripts\build\build_svg_features.py
```
将 `line_seg/**/*.svg` 解析为结构化矢量特征，输出到 `svg_features/<category>/<name>/step_<id>.json`。

每个输出 JSON 包含：
- `canvas`：SVG 画布宽高和 viewBox
- `instances`：每个 stroke 颜色对应一个视觉实例（注意：颜色不是语义零件 ID，不能跨步骤匹配）
- `sampled_points`：从原始 SVG path 中采样得到的点
- `bbox`、`center`、`principal_axis`、`axis_length`、`axis_width`：每个视觉实例的几何特征
- `spatial_relations`：实例间 pairwise 的 bbox 重叠、bbox 距离、中心偏移、采样点距离
- `gt`：来自 `main_data.json` 的步骤级监督信息（`parts`、`connections`）
- `object_gt`：对象级 `assembly_tree`、最终连接关系、几何等价关系

**Step 2：构建索引和空间基线**
```powershell
python scripts\build\build_svg_feature_index.py
python scripts\eval\eval_spatial_connection_baseline.py
```

**Step 3：恢复 SVG 颜色实例到原始零件的对齐**
```powershell
python scripts\export\analyze_instance_mask_alignment.py
```
利用释放的掩码标注，恢复颜色实例到原始零件的监督信号。关键发现：**像素掩码和 RLE 掩码在所有 393 个步骤中完美对齐**，因此颜色实例到原始零件的监督是可以恢复的。

但**启发式对齐不完美**：基于 `area_desc` 的精确颜色到零件的对齐率仅约 **49.4%**。

**Step 4：训练成对连接分类器**
```powershell
python scripts\train\train_pairwise_connection_model.py --align mask
```
使用掩码对齐标签训练。结果：**在发布的 40 步测试集上，基于掩码对齐的成对分类器达到了完美的 test step top-k connection recovery（F1 = 1.0000）**。

仅使用简化 SVG 几何也能保留大部分连接信号：简化 step top-k test F1 = 0.9875，而原始采样 SVG 特征为 1.0000。

**Step 5：渲染 OBJ 原始零件为多视角图像**
```powershell
python scripts\build\render_part_images.py --views 4
```
输出：`part_images/<category>/<name>/<part_id>/view_*.png` 和 `part_images/part_index.jsonl`。
当前预处理渲染了 **754 个原始 OBJ 零件**的合成为多视角图像。

**Step 6：构建 Grounding 数据集**
```powershell
python scripts\build\build_grounding_dataset.py
```
输出：`experiments/svg_assembly/datasets/grounding_samples.jsonl`（1056 个样本：754 个原始样本 + 302 个复合/子装配样本）。

**Step 7：训练 Grounding 基线模型**

详见第 5 节。

**Step 8：构建对象级树生成数据集**
```powershell
python scripts\build\build_tree_generation_dataset.py
```
输出：`tree_generation_dataset.json`，每个对象记录包含：
- 原始零件 token
- 聚合的原始 SVG 原型
- 说明书衍生的复合/子装配 SVG 原型
- 目标装配树
- 后序遍历树动作
- 说明书步骤分组
- 几何等价元数据

**Step 9：训练对象级树规划器基线**

详见第 6 节。

### 4.2 环境配置

当前本地环境：
```powershell
.venv\Scripts\python.exe -m pip install -r requirements-ml.txt
```

已安装核心包：CPU PyTorch、torchvision、scikit-learn、numpy、Pillow、tqdm、matplotlib。

---

## 5. Part-to-SVG Grounding 详细分析

### 5.1 任务定义

**输入**：一个渲染的零件图像（或多视角图像）或零件几何特征
**输出**：该零件对应说明书 SVG 实例的 ID（即步骤中的哪个视觉实例对应哪个零件）

这是一个**将观察到的零件接地到说明书表示**的问题。

### 5.2 模型演进历程

#### 5.2.1 几何-only 基线（`train_grounding_model.py --primitive-only`）

- 仅使用简化 SVG 几何特征（bbox、中心、主轴等）
- 测试集精确分配准确率约 **0.471**
- 全样本分配准确率约 **0.529**
- 结论：简化 SVG 几何本身不足以实现鲁棒的 part-to-SVG grounding

#### 5.2.2 渲染图像轮廓特征（`train_grounding_image_model.py`）

- 使用渲染零件图像的特征向量
- 线性图像特征模型**未改善**全样本 grounding
- 原始-only 精确分配略有改善，但全样本分配下降
- 原因：渲染图像轮廓向量维度高，数据集小，容易过拟合

#### 5.2.3 Tiny CNN（`train_grounding_cnn.py`，旧版）

- 3 层卷积（8→16→32），~30K 参数
- 多视角聚合：mean + max pooling
- SVG 特征维度：15 维
- 测试集严格 Instance Accuracy 约 **69.75%**

#### 5.2.4 改进版残差 CNN（`train_grounding_cnn.py`，新版，详见 `IMPROVEMENT_REPORT.md`）

**模型架构改进**：

| 对比项 | 旧 TinyGroundingCNN | 新 TinyGroundingCNN |
|---|---|---|
| CNN Backbone | 3 层卷积 (8→16→32) | BetterImageCNN: 6 残差块 + 3 降采样层 |
| 通道数 | 8→16→32 | 32→64→128→256 |
| 输出维度 | 32 | 256 |
| 残差连接 | 无 | 有 |
| BatchNorm | 无 | 有 |
| 多视角聚合 | mean + max pool | 可学习注意力权重 + softmax |
| SVG 特征维度 | 15 | 17（+center_x, +center_y） |
| 总参数量 | ~30K | **1,435,314** |

**关键改进点**（来自 `IMPROVEMENT_REPORT.md`）：

1. **SVG 特征增加空间位置**（`center_x`, `center_y`）→ 低投入高回报
2. **残差 CNN + BatchNorm** → 比 3 层浅卷积有实质提升
3. **注意力多视角池化** → 让模型自主选择最有区分力的视角
4. **图片预加载** → 消除磁盘 I/O 瓶颈
5. **GPU 支持** → 利用 NVIDIA GeForce RTX 3060 Laptop GPU (6GB)

**训练细节**：

| 参数 | 值 |
|---|---|
| 训练样本 | 2,619 pairs (300 步) |
| 验证样本 | 396 pairs (53 步) |
| 测试样本 | 435 pairs (40 步) |
| Epochs | 20 |
| Optimizer | AdamW (lr=0.001, weight_decay=0.0001) |
| Loss | BCEWithLogitsLoss（正样本加权） |
| Dropout | 0.2 |
| Max Images per Sample | 8 |
| 硬件 | NVIDIA GeForce RTX 3060 Laptop GPU (6GB) |
| 每轮耗时 | ~130-170s |
| 总训练时间 | ~50 分钟 |

**训练曲线**：

| Epoch | Loss | Val Strict Acc | Val Equiv Acc |
|---|---|---|---|
| 5 | 0.6560 | 73.88% | 81.34% |
| 10 | 0.5871 | 74.63% | **84.33%** |
| 15 | 0.5424 | 72.39% | 82.09% |
| **20** | **0.5078** | **77.61%** | **86.57%** ★ |

**结果对比**：

测试集（40 步，119 实例）：

| 指标 | 旧模型 | 新模型 | 变化 |
|---|---|---|---|
| 严格 Instance Accuracy | 69.75% | **70.59%** | +0.84% |
| 等价 Instance Accuracy | 84.03% | **84.87%** | +0.84% |
| 严格 Exact Match | 75.00% | 67.50% | -7.50% |
| 等价 Exact Match | 85.00% | 82.50% | -2.50% |
| 正确匹配实例数 | 83/119 | 84/119 | +1 |

验证集（选最佳 checkpoint）：

| 指标 | 旧模型 | 新模型 |
|---|---|---|
| 等价 Val Accuracy | 76.87% | **86.57%** (+9.70%) |

**端到端结果对比**（Grounding → Connection 完整流程，Test 集，40 步）：

| 指标 | 旧模型 | 新模型 | 变化 |
|---|---|---|---|
| **端到端连接 F1** | 0.8500 | **0.8875** | **+4.4%** |
| **端到端连接 Exact Match** | 90.0% | **92.5%** | **+2.5%** |

> **关键发现**：Grounding 严格准确率仅提升 0.84%，但端到端连接 F1 提升了 4.4%。说明新模型选择性改进的 grounding 实例对下游连接预测有放大效应。

#### 5.2.5 等价标签残差 CNN

```powershell
python scripts\train\train_grounding_cnn.py --equivalence-labels --epochs 25
```

- 测试集严格 Instance Accuracy：**0.6975**
- 测试集等价 Instance Accuracy：**0.8403**（与之前持平）
- 测试集严格 Exact Match：75.00%
- 验证集等价 Accuracy：**76.87%**

#### 5.2.6 带数据增强的残差 CNN

```powershell
python scripts\train\train_grounding_cnn.py --augment --epochs 25
```

- 轻量图像增强**略微改善** pairwise F1，但**未改善**最终分配准确率
- 说明主要 grounding 差距不只是渲染器过拟合；复合/子装配表示和任务公式化可能更重要

### 5.3 当前最佳 Grounding 结果汇总

来自 `experiment_summary.md`：

| 模型 | pair_f1_test | assign_inst_acc_test | assign_equiv_acc_test | assign_exact_test | assign_inst_acc_val |
|---|---|---|---|---|---|
| geometry primitive | 0.4353 | 0.4706 | - | 0.6486 | - |
| geometry all | 0.4649 | 0.5294 | - | 0.5250 | - |
| linear image primitive | 0.4157 | 0.4941 | - | 0.7027 | - |
| linear image all | 0.3982 | 0.4286 | - | 0.4000 | - |
| tiny CNN primitive e25 | 0.4848 | 0.6471 | - | 0.7297 | - |
| **tiny CNN all e25** | **0.5726** | **0.7143** | - | **0.7250** | - |
| tiny CNN all val | 0.5480 | 0.6471 | - | 0.6750 | 0.7761 |
| tiny CNN all val aug | 0.5694 | 0.6387 | - | 0.6500 | 0.7761 |
| **tiny CNN all equiv-label val** | **0.6649** | **0.6975** | **0.8403** | **0.7500** | **0.7687** |
| **residual CNN all improved val** | **0.7053** | **0.7059** | **0.8487** | **0.6750** | **0.7761** |

**当前最强 Grounding 模型**：改进版残差 CNN（等价标签训练）
- 测试集严格 Instance Accuracy：**0.7059 (70.59%)**
- 测试集等价 Instance Accuracy：**0.8487 (84.87%)**
- 端到端连接 F1（测试集）：**0.8875**

### 5.4 无效 Ablation：跨步骤颜色追踪

曾经测试 `cross_step_bias` 思路：把 stroke 颜色在步骤间当作身份线索。

**结果**（来自 `IMPROVEMENT_REPORT.md`）：

| 设置 | 端到端连接 F1 | Exact Match |
|---|---|---|
| 无跨步骤偏置 | **0.8875** | **0.925** |
| 跨步骤偏置 (+5.0 硬偏置) | 0.7875 | 0.850 |

**结论**：SVG stroke color 只能作为同一个 step 内 parser 分组 path 的临时 instance id，**不能作为跨步骤零件身份**。不同 step 里的相同颜色不表示同一个零件。

### 5.5 已知问题与瓶颈

1. **Grounding 仍是端到端瓶颈**：端到端 F1 0.8875 vs Oracle 0.9875，差距 ~10 个百分点全部来自 grounding 误差
2. **DataLoader 单线程**：GPU 大量空闲，每轮 130s 中计算时间 <10s
3. **复合零件仍是弱点**：composite parts 的 mean/max/min 聚合丢失结构信息
4. **跨步骤一致性需要重新定义**：不能基于 SVG 颜色；如需要 temporal/step consistency，必须学习 latent identity 或用装配上下文约束

---

## 6. 装配树生成：所有方法与结果

### 6.1 任务定义

**输入**：N 个原始零件（几何 + 每个零件的 SVG 特征）
**输出**：装配树，如 `[[0, 1, 2], 3]`（先合并 {0,1,2}，再加 3）

### 6.2 评估指标

| 指标 | 定义 |
|---|---|
| **Simple F1** | 预测的非叶节点 parts 集合是否与 GT 一致？ |
| **Hard F1** | 预测的非叶节点 children 集合（无序）是否与 GT 一致？ |

Hard F1 比 Simple F1 严格得多，要求精确的层次结构匹配。

### 6.3 特征模式（Feature Modes）

| 模式 | 内容 | 需要手动标注？ |
|---|---|---|
| `geometry` | 14 维 3D 几何（bbox, extent, center, n_verts, n_faces） | 否 |
| `svg` | 17 维 SVG 原型 + 4 维形状 + 1 维计数 | 否 |
| `svg_geometry` | geometry + SVG | 否 |
| `svg_composite` | SVG + 23 维手动子装配原型 | **是** |
| `svg_geometry_composite` | geometry + SVG + composite | **是** |

> **重要**：`svg_geometry_composite` 中的 composite token 来自说明书中间步骤，本质上是"泄露"了子装配结构，在推理时不应可用（除非作为 RAG/manual 检索上下文）。

### 6.4 原版 IKEA-Manual 论文方法（对比基线）

```
方法：DGCNN 提取每个零件的 1024 维点云特征
     → 对特征做 K-Means（k=2,3）→ silhouette score 选最佳 k
     → 最佳簇 = 子装配体 → 递归处理剩余零件
     → 自顶向下递归建树

模型：0 个学习参数（纯特征聚类）
训练：无（无监督）
输入：per-part 点云 → DGCNN → 1024 维特征
输出：装配树（通过递归聚类）
```

**核心洞察**：形状相似度天然编码了装配层次。四条桌腿长得像 → 先聚在一起。形状不同的零件在更高层聚合。

**局限性**：完全没有用到 manual step SVG。

**本项目复现的 Paper Tree Metrics**（使用连接诱导树，利用 manual step SVG）：

| 指标 | Precision | Recall | F1 |
|---|---|---|---|
| Simple | 1.0000 | 0.9956 | 0.9972 |
| Hard | 0.9961 | 0.9935 | 0.9944 |

> **注意**：这些指标非常高，因为诊断使用了 manual step SVG；它们不应与论文的形状-only 装配计划生成任务直接比较。

### 6.5 方法 1：BCE Context MLP（当前最优 BCE 模型）

**脚本**：`train_tree_planner_context.py`

**模型架构**：MLP(938→192→192→1)，**218K 参数**

**输入特征**（938 维）：
- `rep(A)`：簇 A 的表示（均值/最大值/最小值聚合的零件特征）
- `rep(B)`：簇 B 的表示
- `|A-B|`：A 有 B 无的特征维度
- `A*B`：A 和 B 的特征逐维乘积
- `rep(A∪B)`：A∪B 的表示
- `global_ctx`：全局上下文（M, log(M), total_parts, cluster 大小均值/标准差, 所有 cluster_repr 的均值/标准差）

**训练**：
- 将所有步骤、所有对象的全部 pair 混在一起 → 4500 个样本
- BCE loss：GT tree action 中的 child-child pair = 正例
- 对输入特征做全局标准化

**推理**：
- 贪心：对所有 pair 打分 → threshold → 连通分量（Connected Components）分组 → 合并
- threshold 在 val 上调优（16 个候选值，选最佳 Hard F1）

**结果**：

| 特征模式 | Test Simple | Test Hard | All Hard |
|---|---|---|---|
| geometry | 0.412 | 0.109 | 0.335 |
| svg_geometry | 0.407 | 0.108 | 0.423 |
| **svg_geometry_composite** | **0.570** | **0.332** | **0.673** |

**5-seed 方差**（svg_geometry_composite）：
```
s0=0.332  s1=0.303  s2=0.254  s3=0.136  s4=0.346
均值=0.274 ± 0.085
```

**该方法的问题**（详见 `METHODS_ANALYSIS.md`）：

1. **66% 的预测是 flat tree**（19/29 个 test 对象）。模型输出单步全合并。
2. **同一 pair 在不同 step 得分几乎一样**。在 applaro_2 上测试：pair (0,2) 在 step 0 得分 0.964，在 step 1 得分 0.987。
3. **特征冗余**。|A-B| 和 A*B 产生了 266 维的 cross-product，大部分是噪声。
4. **标签粗糙**。GT tree action 中所有 child-child pair 都标为正例，即使是在 manual SVG 中没有直接连接的 pair。
5. **Composite token 是"作弊"**。svg_geometry_composite (0.332) vs svg_geometry (0.108)。Composite token 直接编码了手工子装配体标注。
6. **推理时分数压缩**。测试时所有 pair 概率集中在 0.4-0.5，模型无法产生有区分度的分数。

### 6.6 方法 2：GRPO 与 SVG-derived Rewards

**脚本**：`train_tree_grpo.py`

**模型**：Same ContextMergeMLP（218K 参数）+ 参考模型（冻结副本，用于 KL 惩罚）

**训练**：Group Relative Policy Optimization（GRPO）
- 每个对象：采样 K=8-16 棵树，计算 reward，组内归一化
- 策略梯度 + clipped importance sampling
- KL 惩罚朝向参考模型

**Reward 设计**：
```
r = w1 * SVG_coherence + w2 * spatial_SVG + w3 * GT_F1
```
- `SVG_coherence`：非叶节点出现在 manual step 中的比例
- `spatial_SVG`：per-step SVG 几何（空间距离 + 轴对齐 + 连接候选点）
- `GT_F1`：Ground Truth 树 F1（可选，可设为 0 做纯 SVG reward）

**变体结果**：

| 变体 | 描述 | Test Hard |
|---|---|---|
| BCE warm-start | BCE 模型 → GRPO 微调，β=0.05 | 0.306 |
| from-scratch K=8, τ=3.0 | 随机初始化，高温探索 | 0.170 |
| from-scratch K=16, τ=1.5 | 更多样本，中等温度 | 0.258 |
| from-scratch + spatial | 加入 spatial SVG 几何 reward | 0.281 |
| warm-start + spatial | BCE 初始化 + spatial reward | 0.254 |

**Label-Ratio 实验**（svg_geometry_composite，multi-seed）：

| 实验 | Test Hard |
|---|---|
| BCE 100% label | 0.274 ± 0.085 (n=5) |
| BCE 10% label | 0.184 ± 0.044 (n=3) |
| BCE 25% label | 0.218 ± 0.049 (n=3) |
| **GRPO 10% label** | **0.237** (+29% over BCE) |
| **GRPO 25% label** | **0.274** (+5% over BCE) |
| GRPO SVG-only（0% GT，纯 SVG reward） | ~0.346（3 seed 均值） |

**GRPO 的问题**：

1. **Warm-start 探索失败**。BCE 训练的 logits 接近确定性。τ=5.0 时只有 44% 的采样树是唯一的。
2. **From-scratch 过拟合**。Val Hard 达到 0.509（超过 BCE 的 0.442），但 Test Hard 只有 0.281。
3. **Reward 信号问题**。SVG coherence 无区分力；Spatial SVG 信号有效但弱（GT 树得分 0.17-0.33，随机树 0.0）。
4. **训练-推理不一致**。GRPO 训练 policy 最大化 reward，但推理用贪心 argmax，reward 信号在测试时被丢弃。
5. **推理时缺少 spatial 特征**。训练时有 per-step SVG 空间特征，推理时传入空 spatial_map={}。

### 6.7 方法 3：Transformer Set-Context Tree Decoder

**脚本**：`train_tree_decoder.py`

**模型**：SetContextTreeDecoder
- `input_proj`: Linear→LayerNorm→ReLU
- `encoder`: 2-layer TransformerEncoder（128-dim, 4 heads）
- `pair_scorer`: Linear(512→128→64→1)

**结果**：在 73 个对象的数据集上**不如 flat MLP**，很早就被放弃。

**问题**：
1. **数据不足**。73 对象 × ~4 actions = ~300 个训练状态。Transformer attention 需要数千个样本。
2. **对任务来说过于复杂**。4-12 个 cluster 上的 self-attention 在数据集这么小时没有任何优势。

### 6.8 方法 4：Step-Conditioned Sequential Planner (v1)

**脚本**：`train_step_planner.py`（v1）

**模型**：StepMergePlanner
- `PartEncoder`: shape_emb(4→16) + spatial_proj(9→32→16) → 32 维 per part
- `ClusterEncoder`: MultiheadAttention(32, 4 heads) → Linear(32→hidden_dim)
- `GRU(256→256)`：追踪装配进度
- `step_emb(20→32)`：显式 step 计数器
- `pair_scorer`: Linear(773→256→128→1)
- 总计：**640K 参数**

**训练**：Teacher forcing 走过 GT `tree_actions_postorder`（~250 个状态）

**结果**：
- 最佳 val Hard：0.127（epoch 40，之后退化）
- Test Hard：0.022
- 加入 step_emb 后：分数标准差从 0.07 提升到 0.21，但仍然是 flat tree

**问题**：
1. **只有 250 个训练状态**，不够 GRU 泛化
2. **推理时没有 spatial 特征**（训练时有，推理时为空）
3. **50% spatial dropout 不够**

### 6.9 方法 5：Step-Conditioned Sequential Planner (v2 — connections)

**脚本**：`train_step_planner.py`（v2，重写）

使用 393 个 manual step 的 `CONNECTIONS` 作为正例 pair 标签，而非 GT tree actions。

**结果**（5 epochs）：Test Hard 0.021（仍然是 flat tree）

**问题**：
1. **Connections 不编码层次**。所有 manual step 的 connection 在模型看来结构相同。
2. **GRU 输入太压缩**。GRU 接收 `mean(cluster_embeddings)`——一个单一向量，无法区分"6 个单例"和"2 个大 cluster"。

### 6.10 方法 6：聚类 + 对比学习

**脚本**：`train_cluster_planner.py`

**模型**：PartProjector: Linear(8→128→128→64)，50K 参数

**训练**：在 connections 上做对比学习
- 正例：拉近有连接的 part-pair 在嵌入空间中的距离
- 负例：推远无连接的 part-pair（超过 margin）
- Hinge loss: `pos_dist.mean() + clamp(margin - neg_dist, 0).mean()`

**推理**：K-Means(k=2,3) + silhouette score → 递归自顶向下建树（与原版论文相同算法）

**结果**（30 epochs）：
- Test Hard：0.035（与 geometry-only greedy baseline 相同）
- 对比比率：有连接的 pair 比无连接的近 2.5 倍

**问题**：
1. **对比 loss 扁平化了层次**。所有 step 的所有 connection 被平等对待。
2. **形状相似度 vs 连接关系**。原版论文用的是 DGCNN 特征（形状相似度）。我们的对比嵌入编码的是连接关系。这两种信息完全不同。
3. **K-Means 没有参数可调**。无法将 step 顺序或 manual 知识注入聚类过程。

### 6.11 完整结果汇总（所有模型，所有特征模式）

来自 `PROJECT_TASKS.md`：

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
context MLP (BCE)                     0.5704     0.3316     0.6729  -> BEST OVERALL
context MLP + entropy reg             0.5228     0.2167     0.6137
GRPO warm-start (BCE->RL)             0.5651     0.3064     0.6238
GRPO from-scratch K=8  tau=3.0         0.4519     0.1696     0.3696
GRPO from-scratch K=16 tau=1.5         0.5203     0.2577     0.3846
GRPO from-scratch K=16 + spatial     0.5018     0.2813     0.3947
GRPO warm-start + spatial            0.5344     0.2540     0.5258
manual composite oracle              1.0000     1.0000     1.0000
```

**子装配候选预测模型**（`train_subassembly_candidate_model.py`）：

| 模型 | candidate_F1_test | Simple F1 | Hard F1 |
|---|---|---|---|
| recall-heavy（无负采样） | 0.0174 | 0.2869 | 0.0675 |
| conservative（decoder threshold） | 0.0194 | 0.4063 | 0.0976 |
| **neg20（20 负样本/正样本）** | **0.0221** | **0.3952** | **0.1128** |

> 负采样版本有所改善，但候选集仍然太噪声，无法替代 manual/RAG composite context。

---

## 7. GRPO 强化学习详解

### 7.1 GRPO 算法原理

Group Relative Policy Optimization（GRPO）是本项目中用于装配树生成的一种**策略梯度强化学习方法**，专为**标签稀缺场景**设计。

**核心思想**：
- 每个对象采样 K 棵树（K=8-16）
- 计算每棵树的 reward
- 在组内对 reward 做归一化（减去组内均值，除以标准差）
- 用归一化 reward 作为优势估计，计算策略梯度
- 加入 clipped importance sampling 和 KL 惩罚（朝向参考模型）

### 7.2 Reward 设计

```
r = w1 * SVG_coherence + w2 * spatial_SVG + w3 * GT_F1
```

**SVG_coherence**：
- 定义：预测树中出现在任何 manual step 中的非叶节点比例
- 意图：鼓励预测的树结构与说明书步骤一致
- 问题：很多不同的树得到相同的 reward → 无区分力

**spatial_SVG**：
- 定义：per-step SVG 几何信号（proximity + axis alignment + connection candidates）
- GT 树得分范围：0.17-0.33
- 随机树得分：0.0
- 结论：信号存在，但较弱

**GT_F1**：
- 定义：预测树与 GT 树之间的 F1
- 可提供最强监督，但需要 GT 树标签
- 在 label-ratio 实验中可调节 `gt-label-ratio` 来控制 GT reward 权重

### 7.3 SVG-only GRPO（无 GT reward）

设置 `--reward-gt-f1 0 --reward-svg-coherence 0.4 --reward-spatial-svg 0.6 --gt-label-ratio 0`

**结果**（svg_geometry_composite）：

| 方法 | Seed | Test Hard |
|---|---|---|
| GRPO scratch | 0 | **0.423** |
| GRPO scratch | 1 | 0.365 |
| GRPO scratch | 2 | 0.250 |
| GRPO warm | 0 | 0.233 |

**3-seed 均值**：~0.346

> **结论**：纯 SVG/spatial reward 在 svg_geometry_composite 上达到约 0.346 Test Hard，高于 100% BCE context 基线（0.332）。这支持了"manual-step SVGs 包含可用装配结构"的论点。

**但**：结果方差大，应在固定数据划分和多 seed 下验证后再作为 headline 数字。

### 7.4 Label-Ratio 实验详细结果

来自 `label_ratio_sgc_full_summary.md`：

| label_ratio | method | Test Simple | Test Hard | All Hard |
|---|---|---|---|---|
| 10% | BCE context MLP | 0.466 | 0.182 | 0.270 |
| 10% | GRPO (gt=0.50, svg=0.20, spatial=0.30) | 0.487 | **0.237** | 0.333 |
| 25% | BCE context MLP | 0.535 | 0.261 | 0.386 |
| 25% | GRPO (gt=0.50, svg=0.20, spatial=0.30) | 0.552 | **0.274** | 0.376 |
| 50% | BCE context MLP | 0.495 | 0.199 | 0.442 |
| 50% | GRPO (gt=0.50, svg=0.20, spatial=0.30) | 0.491 | 0.195 | 0.410 |
| 100% | BCE context MLP | **0.570** | **0.332** | **0.673** |
| 100% | GRPO (gt=0.50, svg=0.20, spatial=0.30) | 0.516 | 0.210 | 0.546 |

**核心结论**：
- **GRPO 在标签稀缺时帮助最大**：10% labels 时 Test Hard 从 0.182 → 0.237（+30%）；25% labels 时从 0.261 → 0.274（+5%）
- **全监督时 GRPO 不优于 BCE**：100% labels 时 BCE 的 Test Hard 0.332 > GRPO 的 0.210
- GRPO 应定位为**弱监督/低标签方法**，而非全监督 BCE 的替代

### 7.5 无 Manual 对比（svg_geometry）

来自 `label_ratio_sg_bce_summary.md`：

| label_ratio | Test Simple | Test Hard | All Hard |
|---|---|---|---|
| 10% | 0.338 | 0.060 | 0.164 |
| 25% | 0.365 | 0.077 | 0.164 |
| 50% | 0.455 | 0.124 | 0.333 |
| 100% | 0.407 | 0.108 | 0.423 |

> **结论**：去掉 manual-derived composite 特征后，Test Hard 急剧下降。当前强结果主要依赖 manual-step 子装配上下文，而非单纯原始 SVG 形状。

---

## 8. 实验报告数据汇总

### 8.1 端到端诊断结果

来自 `experiment_summary.md`：

| 名称 | F1 | Exact Match |
|---|---|---|
| base predicted grounding | 0.8500 | 0.9000 |
| equiv-label predicted grounding | 0.8625 | 0.9250 |
| improved predicted grounding | 0.8875 | 0.9250 |
| oracle grounding | 0.9875 | 0.9750 |

### 8.2 等价感知指标

| 名称 | ground_strict | ground_equiv | conn_strict_f1 | conn_equiv_f1 |
|---|---|---|---|---|
| test/base-label | 0.6471 | 0.8403 | 0.8500 | 0.8875 |
| test/equiv-label | 0.6975 | 0.8403 | 0.8625 | 0.9000 |
| test/improved | 0.7059 | 0.8487 | 0.8875 | 0.9375 |
| all/export/base-label | 0.7150 | 0.7585 | 0.9238 | 0.9283 |
| all/export/equiv-label | 0.7566 | 0.8087 | 0.9492 | 0.9537 |
| all/export/improved | 0.7311 | 0.7879 | 0.9567 | 0.9626 |

> **test 行是更严格的泛化视图；all/export 行用于写出完整的 per-object 预测装配树。**

### 8.3 简化 SVG 连接预测结果

| 名称 | pair_f1_test | step_topk_f1_test | step_exact_test |
|---|---|---|---|
| sampled SVG + mask alignment | 0.9427 | 1.0000 | 1.0000 |
| simplified SVG + mask alignment | 0.9560 | 0.9875 | 0.9750 |

---

## 9. 关键发现与根因分析

### 9.1 所有方法都困难的根因

来自 `METHODS_ANALYSIS.md`：

| 根因 | 受影响方法 | 解释 |
|---|---|---|
| **Step 顺序未被使用** | 方法 1-6 | 所有方法都把 steps 混在一起或平均掉。模型从未学到"先合并 {0,1,2}，再合并 {0,1,2,3}"。 |
| **训练数据太少** | 方法 3-5 | 62 个对象 × ~4 tree action = ~250 个状态。GRU/transformer 需要数千个状态才能学到有意义的转移。 |
| **特征编码了"什么"而非"何时"** | 方法 1-6 | Part 特征告诉模型哪些零件是相似的，但不告诉模型它们该在哪个 step 合并。 |
| **推理-训练不一致** | 方法 2, 4, 5 | 训练时用了 step SVG（spatial context），推理时没有。模型学了错误的依赖。 |
| **Flat BCE 无法编码层次** | 方法 1, 4, 5 | 每 pair 的 BCE 把所有 step 平等对待。应该在 step 1 合并的 pair 看起来和应该在 step 3 合并的 pair 一样。 |
| **标签粒度粗** | 方法 1 | GT tree action 中所有 child-child pair 都为正例，即使是 manual SVG 中没有直接连接的 pair。 |
| **贪心 CC decoder 过于简单** | 方法 1-5 | 单一 threshold 控制一切。如果分数被压缩，CC 要么产生 flat tree（低 threshold）要么什么都不合（高 threshold）。没有中间地带。 |

### 9.2 有效的东西

1. **Composite token 很强**。它们直接编码了手工子装配体知识。Test Hard 0.332 vs 0.162（不用 composite）。
2. **GRPO 在标签稀缺时有用**。在 10% GT label 时，GRPO + SVG reward 比纯 BCE 高 30%。
3. **SVG spatial reward 是有效的信号**。GT 树得分 0.17-0.33，随机树 0.0。
4. **Step-conditioned 模型消灭了 flat tree**。Sequential GRU 模型的 flat tree 率为 0%（BCE 是 66%）。

### 9.3 原版论文做对了什么

1. **形状相似度天然编码层次**。
2. **递归聚类原生建树**，不需要 sequential 预测。
3. **零参数 = 零过拟合**。
4. **K-Means + silhouette 简单鲁棒**。

---

## 10. 文件依赖关系与项目时间线

### 10.1 文件间依赖关系

**数据依赖链**：
```
main_data.json + line_seg/*.svg
    → build_svg_features.py → svg_features/*.json
    → build_svg_feature_index.py → svg_features/index.jsonl
    → analyze_instance_mask_alignment.py → mask 对齐标签
    → train_pairwise_connection_model.py → 连接分类器

main_data.json + part_images/
    → build_grounding_dataset.py → grounding_samples.jsonl
    → train_grounding_cnn.py → grounding CNN 模型

main_data.json + svg_features/*.json
    → build_tree_generation_dataset.py → tree_generation_dataset.json
    → train_tree_planner_context.py / train_tree_grpo.py → 树规划器模型
```

**脚本依赖**：
- `scripts/build/` (7 个)：数据构建，是后续所有训练的前置
- `scripts/train/` (12 个)：训练脚本，依赖 build 的输出
- `scripts/eval/` (5 个)：评估脚本，依赖训练好的模型
- `scripts/export/` (4 个)：导出脚本，依赖评估结果

### 10.2 项目时间线（推断）

根据文件内容和最后更新时间推断：

1. **初始阶段**：数据集构建，`main_data.json` 标注，PDF 转 SVG（`line_seg/`）
2. **SVG 特征解析**：实现 `build_svg_features.py`，解析 SVG 为结构化特征
3. **连接预测**：实现成对连接分类器，验证 SVG 特征对连接预测的有效性
4. **Part-to-SVG Grounding**：实现多个版本的 grounding 模型（几何、图像、CNN、残差 CNN）
5. **装配树规划器**：实现多种规划器（greedy baseline → flat MLP → context MLP → GRPO → Transformer → step-conditioned → 对比学习）
6. **GRPO 与低标签实验**：实现 label-ratio 控制，验证 GRPO 在低标签场景的优势
7. **改进 Grounding CNN**（2026-05-25 ~ 2026-05-26）：残差 CNN、注意力池化、空间特征（记录于 `IMPROVEMENT_REPORT.md`）
8. **方法分析与总结**（2026-05-27）：撰写 `METHODS_ANALYSIS.md`、`PROJECT_TASKS.md`，系统总结所有方法的成败

---

## 11. 待解决问题与下一步计划

### 11.1 已完成事项

- [x] 解析 manual SVG 为结构化矢量特征
- [x] 简化 SVG 实例为紧凑几何 token
- [x] 从释放掩码恢复监督 SVG 实例到零件的对齐
- [x] 渲染原始 OBJ 零件为合成多视角零件图像
- [x] 构建 part-to-SVG grounding 样本
- [x] 训练和评估简化 SVG 连接分类器
- [x] 训练和评估 grounding CNN 基线（最佳：残差 CNN，等价准确率 84.9%）
- [x] 构建对象级树生成数据集（102 对象，composite tokens）
- [x] 训练初始贪心树规划器基线
- [x] 训练神经网络合并评分器基线
- [x] 加入复合/子装配 SVG 原型作为 manual/RAG 上下文
- [x] 训练无泄露子装配候选预测器 + 负采样
- [x] 导出规划器预测并构建错误查看器
- [x] 加入端到端诊断和论文树指标评估
- [x] 重组 `scripts/` 为 `build/`、`train/`、`eval/`、`export/` 子目录
- [x] 实现 Transformer set-context 树解码器（负结果）
- [x] 实现 context-augmented flat MLP（**最佳模型**：Test Hard F1 0.332）
- [x] 实现 GRPO 树规划器（SVG-derived rewards）
- [x] 运行 label-ratio 控制实验
- [x] 验证 svg_geometry_composite scratch SVG-only GRPO across seeds

### 11.2 活跃 Todo

- [ ] 用固定划分和多种子重复关键运行
- [ ] 跨种子验证 SVG/spatial reward 在标签稀缺时是否有帮助
- [ ] 结合 BCE 预训练 + GRPO 与强 KL 约束以保留泛化能力
- [ ] 测试全对象弱 GRPO 作为 ablation，但保留 held-out test split 用于论文指标
- [ ] **学习一个 proper 树解码器**（sequence/set-to-tree）而非贪心连通分量
- [ ] 改善无泄露子装配预测精度
- [ ] 加入真实图像或 3D 观察 grounding 基准
- [ ] 加入 RAG/manual-retrieval hooks 作为可选规划器上下文

### 11.3 下一步计划（优先级排序）

1. **在将 SVG-only GRPO 作为 headline 结果之前，跨种子验证**
2. **检查 reward 在标签稀缺时跨种子的有用性**：用 2-3 个 label seeds 重复 10%/25% 实验
3. **更强的 BCE→GRPO 桥接**：BCE warm-start + 高 KL 惩罚（beta=0.5~1.0）以在局部探索的同时保持接近预训练策略
4. **更丰富的 SVG reward**：除空间邻近度外，使用 step 顺序和来自 manual steps 的连接图结构
5. **学习树解码器**：用 sequential merge predictor（RNN/Transformer decoder）替代贪心连通分量

### 11.4 开放问题

1. **能否在不做 sequential prediction 的情况下利用 step 顺序？** step embedding + BCE 训练保留了 BCE 的泛化能力，同时加入了 step 感知。尚未测试。
2. **能否获得更好的 part 特征？** 原版论文使用 DGCNN（1024 维点云特征）。我们使用手动的 36 维特征。part_images/（754 张渲染图）能否训练出更好的特征提取器？
3. **贪心 CC decoder 是否是瓶颈？** 我们所有模型都用相同的 decoder。换成一个学习的 decoder（RNN/pointer network）即使在当前特征下能否有帮助？
4. **数据增强有多大帮助？** 102 个对象很小。能否从 393 个 manual step 生成合成的装配序列？

---

## 附录 A：脚本索引

```
scripts/
├── build/   (7)  数据构建和特征提取
│   ├── build_svg_features.py                # SVG 特征解析
│   ├── build_svg_feature_index.py           # 构建步骤级索引
│   ├── simplify_svg_instances.py           # 简化 SVG 几何
│   ├── render_part_images.py                # 渲染零件图像
│   ├── build_grounding_dataset.py          # 构建 grounding 数据集
│   ├── build_tree_generation_dataset.py    # 构建树生成数据集
│   └── build_tree_planner_error_viewer.py  # 构建错误查看器 HTML
├── train/   (12) 模型训练
│   ├── train_tree_planner_baseline.py      # 逻辑回归基线
│   ├── train_tree_planner_nn.py            #  flat MLP
│   ├── train_tree_planner_context.py       # 上下文感知 MLP（最佳）
│   ├── train_tree_decoder.py               # Transformer 解码器（失败）
│   ├── train_tree_grpo.py                 # GRPO 强化学习
│   ├── train_grounding_cnn.py             # Grounding CNN
│   ├── train_grounding_model.py           # 几何 grounding
│   ├── train_grounding_image_model.py     # 图像特征 grounding
│   ├── train_pairwise_connection_model.py  # 成对连接模型
│   ├── train_simplified_connection_model.py # 简化连接模型
│   └── train_subassembly_candidate_model.py # 子装配候选模型
├── eval/    (5)  评估和推理
│   ├── eval_spatial_connection_baseline.py     # 空间连接基线
│   ├── run_tree_planner_inference.py           # 规划器推理
│   ├── run_end_to_end_diagnostic.py           # 端到端诊断
│   ├── evaluate_composite_context_decoder.py   # 复合上下文解码器评估
│   └── evaluate_paper_tree_metrics.py         # 论文树指标评估
└── export/  (4)  导出和分析
    ├── export_tree_planner_predictions.py             # 导出规划器预测
    ├── export_tree_predictions_and_equivalence_report.py # 导出树预测和等价报告
    ├── summarize_svg_assembly_experiments.py         # 总结所有实验报告
    └── summarize_label_ratio_experiments.py          # 总结 label-ratio 实验
```

---

## 附录 B：核心数字速查

| 指标 | 最佳值 | 模型 | 备注 |
|---|---|---|---|
| Grounding 测试严格 Acc | **70.59%** | 残差 CNN (improved) | 等价 Acc 84.87% |
| 端到端连接 F1（测试） | **0.8875** | 残差 CNN grounding | Oracle 为 0.9875 |
| 装配树 Test Hard F1 | **0.332** | BCE context MLP + composite | 无 composite 最佳：0.162 |
| 装配树 All Hard F1 | **0.673** | BCE context MLP + composite | |
| GRPO 低标签增益（10%） | **+29%** | GRPO vs BCE | Test Hard: 0.237 vs 0.182 |
| Paper Tree Simple F1 | **0.9972** | 连接诱导树 + manual SVG | 不直接可比 |
| Paper Tree Hard F1 | **0.9944** | 连接诱导树 + manual SVG | 不直接可比 |
| 复合上下文 Oracle | **1.0000** | manual composite oracle | 确认泄露边界 |

---

*报告结束。所有数据来自 `D:\Docu\LLM-RAG\dataset` 目录下的 20 个 .md 文件，经深度阅读后整理。*
