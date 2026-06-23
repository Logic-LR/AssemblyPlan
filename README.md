# IKEA-Manual 装配树生成

从 IKEA 家具说明书中自动推理装配计划。给定 N 个零件（3D 几何 + SVG 空间特征），
预测层次化装配树 —— 例如 `[[0, 1, 2], 3]` 表示"先将 {0,1,2} 合并为子装配体，再与 3 组合"。

## 仓库结构

```
├── README.md                            ← 本文件
├── IKEA-Manual Dataset 详细总结报告.md    # 项目完整技术总结（11 章）
├── EXPERIMENT_REPORT.md                  # GNN + GRPO 实验报告 (assembly_plan)
├── main_data.json                        # 核心标注数据（步骤、连接、装配树）
│
├── code/                                 # 原始基线实现
│   ├── manual_generation/                # 说明书生成实验（DGCNN + K-Means）
│   └── part_assembly/                    # NeurIPS 2020: Generative 3D Part Assembly
│
├── scripts/                              # SVG 装配实验工具
│   ├── build/                            # 数据构建与特征提取（7 个脚本）
│   ├── train/                            # 模型训练（12 个脚本）
│   ├── eval/                             # 评估与推理（5 个脚本）
│   └── export/                           # 导出与分析（4 个脚本）
│
├── assembly_plan/                        # GNN + GraphSAGE 装配树规划器（当前主力）
│   ├── model.py                          # GNNMergeModel（GraphSAGE + MLP 打分器）
│   ├── decoder.py                        # Group-Aware k-ary 树解码器
│   ├── train.py / run.py                 # 训练 / 推理入口
│   └── EXPERIMENT_REPORT.md              # 详细实验结果
│
├── experiments/svg_assembly/             # 实验中心
│   ├── METHODS_ANALYSIS.md               # 6 种方法 + 根因分析（英文）
│   ├── METHODS_ANALYSIS_CN.md            # 同上（中文）
│   ├── EXPERIMENT_REPORT_CN.md           # 组会汇报稿
│   ├── IMPROVEMENT_REPORT.md             # Grounding CNN 改进报告
│   ├── datasets/                         # 构建的数据集
│   │   └── DATASET_FIELDS.md             # 每个特征维度的详细说明
│   └── reports/                          # 实验报告汇总
│
├── data/                                 # 原始 XML 数据样本
├── assembly_trees/                       # GT 装配树导出
└── line_seg/                             # SVG 线条分割（每步骤一个 SVG）
```

## 数据集

- **102 个 IKEA 家具对象**（73 train / 29 test）
- 754 个 primitive parts，404 个 tree actions，393 个 manual steps
- 每个零件的特征：**35 维**（geometry 14 + SVG spatial 17 + shape type 4）
- 详见 `experiments/svg_assembly/datasets/DATASET_FIELDS.md`

## 关键结果

| 模型 | Hard F1 | 备注 |
|------|:---:|------|
| BCE Context MLP + composite | 0.332 | 最佳监督方法，但 composite token 泄露 manual 答案 |
| BCE Context MLP（纯特征）| 0.108 | 无 manual 信息 |
| GNN + GRPO + Group decoder | **0.204** | 当前最佳，产生真正的嵌套树 |
| GRPO SVG-only（0% GT 标签）| ~0.346 | 3 seed 均值，方差大 |

BCE 方法的 66% 预测为平树（所有零件一步合并）。完整汇报见 `experiments/svg_assembly/EXPERIMENT_REPORT_CN.md`。

## 快速开始

### 环境安装

```powershell
# 根目录的通用环境（scripts/ 使用）
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-ml.txt

# assembly_plan 有独立环境（需要 torch_geometric 等）
python -m venv assembly_plan\.venv
assembly_plan\.venv\Scripts\python.exe -m pip install torch torch_geometric tqdm numpy
```

### 构建数据集（如已有 tree_generation_dataset.json 可跳过）

```powershell
# 1. 解析 SVG 为结构化特征
python scripts/build/build_svg_features.py

# 2. 简化 SVG 几何实例
python scripts/build/simplify_svg_instances.py

# 3. 渲染零件 3D 模型为多视角图像
python scripts/build/render_part_images.py --views 4

# 4. 构建 grounding 样本
python scripts/build/build_grounding_dataset.py

# 5. 构建对象级树生成数据集
python scripts/build/build_tree_generation_dataset.py
```

### 训练

```powershell
# 监督训练
assembly_plan\.venv\Scripts\python.exe -m assembly_plan.run --mode train

# GRPO 微调（需要先有监督 checkpoint）
assembly_plan\.venv\Scripts\python.exe -m assembly_plan.run --mode train_grpo --ckpt <checkpoint路径>

# 评估
assembly_plan\.venv\Scripts\python.exe -m assembly_plan.run --mode eval --ckpt <checkpoint路径>
```

## 文档索引

| 文档 | 内容 |
|------|------|
| `IKEA-Manual Dataset 详细总结报告.md` | 项目完整技术总结（11 章） |
| `experiments/svg_assembly/EXPERIMENT_REPORT_CN.md` | 组会汇报稿 |
| `experiments/svg_assembly/METHODS_ANALYSIS_CN.md` | 6 种方法详细分析与根因 |
| `experiments/svg_assembly/datasets/DATASET_FIELDS.md` | 训练数据每个字段的详细说明 |
| `EXPERIMENT_REPORT.md` | GNN + GRPO 实验报告（英文） |
| `assembly_plan/EXPERIMENT_REPORT.md` | GNN 模块详细实验结果（英文） |
