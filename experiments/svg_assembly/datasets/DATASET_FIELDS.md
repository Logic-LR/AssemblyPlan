# tree_generation_dataset.json — 字段说明文档

> 路径：`experiments/svg_assembly/datasets/tree_generation_dataset.json`
> 构建脚本：`scripts/build/build_tree_generation_dataset.py`

---

## 一、数据来源总览

```
main_data.json（人工标注）
    │
    ├── assembly_tree ───────────────┐
    ├── steps (parts, connections) ──┤
    ├── connection_relation ─────────┤
    └── geometric_equivalence ───────┤
                                     ▼
line_seg/*.svg                build_tree_generation_dataset.py
    │                                     │
    └→ build_svg_features.py              │
       → svg_features/*.json              │
          │                               │
          └→ simplify_svg_instances.py    │
             → simplified_svg/*.json      │
                │                         │
                └→ build_grounding_dataset.py
                   → grounding_samples.jsonl
                      │                  │
parts/*.obj              │                  │
    │                    │                  │
    └→ render_part_images.py              │
       → part_images/ + part_index.jsonl   │
                      │                  │
                      ▼                  ▼
               tree_generation_dataset.json
```

**一句话概括**：从 `main_data.json` 取 GT 标注（装配树、连接关系、步骤信息），从渲染的 3D 零件模型取几何特征，从说明书 SVG 取 2D 视觉特征，三者合并成对象级训练数据。

---

## 二、顶层字段

每个对象一条记录，102 条：

| 字段 | 类型 | 来源 | 说明 |
|------|------|------|------|
| `category` | string | main_data.json | 家具类别，如 `"Bench"`, `"Chair"` |
| `name` | string | main_data.json | 对象名称，如 `"applaro"` |
| `split` | string | 自动判定 | `"train"`(73) 或 `"test"`(29)。规则：只要该对象有任意一个 step 的 `part_segmentation_split == "test"`，整个对象就是 test |
| `num_parts` | int | main_data.json `parts_ct` | 零件总数 |
| `part_tokens` | list[dict] | 见 §三 | 每个零件的特征向量（模型输入） |
| `composite_tokens` | list[dict] | 见 §四 | 说明书衍生的子装配体特征（⚠️ 可选 manual 上下文） |
| `assembly_tree` | list | main_data.json | GT 装配树，嵌套列表格式 |
| `tree_actions_postorder` | list[dict] | 从 assembly_tree 自动生成 | 后序遍历拆解的训练动作（见 §五） |
| `manual_step_groups` | list[dict] | main_data.json `steps` | 每个说明书步骤的零件分组和连接关系 |
| `connection_relation` | list[list] | main_data.json | 零件间无向连接边（GNN 图输入） |
| `geometric_equivalence_relation` | dict | main_data.json | 几何等价零件分组 |

---

## 三、`part_tokens` — 零件特征（模型核心输入）

每个零件一条记录，包含三类特征，最终拼接为 **34 维向量**（部分模型使用 38 维，视 svg_feature_std 使用方式而定）：

```
最终模型输入 = geometry_feature(14) + svg_feature_mean(17) + shape_distribution(4)
            ≈ 34 维*
(* 部分模型还会加入 svg_feature_count=1，共 35 维)
```

### 3.1 `geometry_feature` [14 维] — 3D 几何属性

**来源**：`part_index.jsonl` 中 OBJ 3D 模型的包围盒和面数统计
**函数**：`scripts/train/train_grounding_cnn.py → primitive_geometry()`

```
输入: part = {
    "extent": [x, y, z],          # 3D bbox 尺寸（米）
    "num_faces": int,             # 三角面数
    "num_vertices": int,          # 顶点数
}
```

| 索引 | 计算方式 | 含义 | 示例(part 0 座板) |
|:---:|------|------|:---:|
| 0 | `extent.x` | 3D bbox X 尺寸（原始顺序）| 1.70 m |
| 1 | `extent.y` | 3D bbox Y 尺寸 | 0.09 m |
| 2 | `extent.z` | 3D bbox Z 尺寸 | 0.03 m |
| 3 | `sorted_ext[0]` | bbox 最长边（降序排列后）| 1.70 m |
| 4 | `sorted_ext[1]` | bbox 次长边 | 0.09 m |
| 5 | `sorted_ext[2]` | bbox 最短边 | 0.03 m |
| 6 | `sorted_ext[0] / sorted_ext[1]` | 最长/次长 比例 = 18.33 | 极度细长（板状） |
| 7 | `sorted_ext[1] / sorted_ext[2]` | 次长/最短 比例 = 3.00 | 中等扁平 |
| 8 | `sorted_ext[0] / sorted_ext[2]` | 最长/最短 比例 = 55.0 | 极端比例 |
| 9 | `extent.x × extent.y × extent.z` | bbox 体积 = 0.0049 m³ | 很小 |
| 10 | `log1p(num_faces)` | ln(1 + 面数) = ln(25) ≈ 3.22 | 面数对数压缩 |
| 11 | `log1p(num_vertices)` | ln(1 + 顶点数) = ln(49) ≈ 3.89 | 顶点数对数压缩 |
| 12 | `ratio[0] > 8 ? 1 : 0` | 是否极度细长（最长边 > 8× 次长边）| 1（是） |
| 13 | `ratio[1] > 4 ? 1 : 0` | 是否扁平（次长边 > 4× 最短边）| 0（否） |

> **归纳**：[0-5]是尺度信息，[6-8]是形状比例，[9]是体积，[10-11]是复杂度，[12-13]是形状分类的简化二值特征。

### 3.2 `svg_feature_mean` [17 维] — SVG 2D 空间特征

**来源**：简化 SVG 实例 (`simplified_svg/*/step_N/overlay.svg` 对应的 `simplified_instances.json`)
**函数**：`scripts/train/train_grounding_cnn.py → svg_feature()`

**画布基准**：793.701 × 1122.52（A4 竖版），对角线 ≈ 1375.0

```
输入: inst = {
    "bbox": [x1, y1, x2, y2],        # 零件在说明书图中的包围盒（像素坐标）
    "center": [cx, cy],               # 中心点
    "axis_length": float,             # PCA 主轴长度
    "axis_width": float,              # PCA 垂直宽度
    "elongation": float,              # 伸长率 = λ₁ / λ₂
    "simplified_polygon": [[x,y],..], # RDP 简化多边形顶点
    "convex_hull": [[x,y],..],        # 凸包顶点
    "shape_type": str,                # 形状类型
}
```

| 索引 | 计算方式 | 含义 | 示例(part 0) |
|:---:|------|------|:---:|
| 0 | `center_x / 793.701` | 归一化中心 X（0~1）| 0.482 |
| 1 | `center_y / 1122.52` | 归一化中心 Y（0~1）| 0.252 |
| 2 | `bbox_w / 793.701` | 归一化 bbox 宽度 | 0.289 |
| 3 | `bbox_h / 1122.52` | 归一化 bbox 高度 | 0.137 |
| 4 | `bbox_area / canvas_area` | bbox 面积占画布比例 | 0.040 |
| 5 | `polygon_area / canvas_area` | 简化多边形面积占比 | 0.003 |
| 6 | `hull_area / canvas_area` | 凸包面积占比 | 0.006 |
| 7 | `axis_length / diag` | 归一化主轴长度 | 0.199 |
| 8 | `axis_width / diag` | 归一化主轴宽度 | 0.019 |
| 9 | `log1p(elongation)` | 伸长率（对数压缩）| 6.13 |
| 10 | `log1p(axis_length/axis_width)` | 长宽比（对数压缩）| 2.45 |
| 11 | `len(simplified_polygon) / 12` | 简化多边形顶点数/12 | 0.25 |
| 12 | `len(convex_hull) / 32` | 凸包顶点数/32 | 0.47 |
| 13 | one-hot[0] | 形状 = elongated_bar（长条形）| 1.0 |
| 14 | one-hot[1] | 形状 = plate_like（板状）| 0.0 |
| 15 | one-hot[2] | 形状 = irregular（不规则）| 0.0 |
| 16 | one-hot[3] | 形状 = point_or_line（点/线）| 0.0 |

> **归纳**：[0-3]是空间位置和大小，[4-6]是面积特征，[7-10]是主轴/形状特征，[11-12]是复杂度，[13-16]是形状类型 one-hot。

**四种形状类型的判定**（来自 `simplify_svg_instances.py` 的 PCA 分析）：

| 类型 | 条件 | 典型零件 |
|------|------|------|
| `elongated_bar` | 伸长率 > 3.0 | 座板、横梁、桌腿 |
| `plate_like` | 伸长率 ≤ 3.0 且 area > 阈值 | 侧板、桌面 |
| `irregular` | 多边形复杂度高 | 不规则连接件 |
| `point_or_line` | 顶点数极少或面积极小 | 螺丝、小连接件 |

### 3.3 `svg_feature_std` [17 维]

同一个零件在**多个步骤中出现**时，其 SVG 特征的跨步骤标准差。大部分零件只在一个步骤出现 → 全为 0。

### 3.4 `svg_feature_count`

该零件在几个 manual step 中出现。通常为 1；若零件跨多步可见则 > 1。

### 3.5 `shape_distribution` [4 维]

该零件在所有出现步骤中的形状类型统计（归一化为概率分布）：

```json
[1.0, 0.0, 0.0, 0.0]  // 100% elongated_bar
```

### 3.6 `shape_histogram`

形状类型的原始计数：
```json
{"elongated_bar": 1}
```

### 3.7 `svg_examples`

零件到说明书 SVG 颜色实例的映射：

```json
[{
    "step_id": 0,                                       // 在步骤 0 出现
    "svg_instance_id": "red",                           // 在该步骤中的颜色 ID
    "source_svg": "line_seg/Bench/applaro/step_0.svg"   // 来源 SVG 文件
}]
```

> ⚠️ **颜色只在单个 step 内有意义**。同一颜色在不同 step 中不代表同一零件。例如 applaro：step 0 的绿色(#00ff4a)=零件 2，step 1 的绿色(#00ff4a)=零件 3。

### 3.8 辅助字段（不参与模型训练）

| 字段 | 来源 | 说明 |
|------|------|------|
| `part_id` | 索引 | 零件编号（字符串），从 0 开始 |
| `obj_path` | part_index | 3D OBJ 模型路径 |
| `image_paths` | part_index | 4 个视角的渲染 PNG |
| `num_vertices` | part_index | 3D 模型顶点数 |
| `num_faces` | part_index | 3D 模型三角面数 |
| `bbox_min` / `bbox_max` | part_index | 3D 包围盒角点 |
| `extent` | part_index | 包围盒尺寸 [x, y, z] |
| `center` | part_index | 3D 中心坐标 |

---

## 四、`composite_tokens` — Manual 子装配体特征

**来源**：从 `grounding_samples.jsonl` 中筛选 `is_composite=True` 且包含 ≥2 个 primitive parts 的样本

当说明书的一个步骤中，多个零件已经被组装成一个整体出现在后续步骤里，这个"整体"就是一个 composite token。

```json
{
    "part_ids": ["0", "1", "2"],          // 包含哪些零件
    "part_token": "0,1,2",                // 名称
    "svg_feature_mean": [17 维],           // 作为整体的 SVG 特征
    "svg_feature_std": [17 维],
    "svg_feature_count": 1,
    "shape_distribution": [4 维],
    "svg_examples": [{                     // 在 step 1 图中作为整体出现
        "step_id": 1,
        "svg_instance_id": "red",
        "source_svg": "line_seg/Bench/applaro/step_1.svg"
    }]
}
```

> ⚠️ **Composite token 是 manual 信息泄露**。它直接告诉模型"0,1,2 是一个子装配体"。使用 `svg_geometry_composite` 特征模式时，这些信息参与训练/推理。在 `svg_geometry` 模式下不使用。Hard F1 差距：0.332 vs 0.108。

---

## 五、`assembly_tree` 与 `tree_actions_postorder` — GT 装配树与训练标签

### 5.1 `assembly_tree` — GT 装配树

```json
[[0, 1, 2], 3]
```

含义：
```
        根 [0,1,2,3]
        /          \
    [0,1,2]         3      ← 先在 step 0 合并座板(0)+侧板(1)+侧板(2)
    /  |  \                 再在 step 1 与横梁(3)合并
   0   1   2
```

### 5.2 `tree_actions_postorder` — 后序遍历拆解

函数 `postorder_actions()` 将树递归拆为训练动作序列：

```json
[
    {   // Action 1（后序最先执行）: 合并单零件 0,1,2
        "parent": [0, 1, 2],
        "children": [[0], [1], [2]]
    },
    {   // Action 2: 合并子装配体 {0,1,2} 和单零件 3
        "parent": [0, 1, 2, 3],
        "children": [[0, 1, 2], [3]]
    }
]
```

**BCE 训练怎么用**：
- Action 1 的正例 pair：(0,1), (0,2), (1,2)
- Action 1 的负例 pair：(0,3), (1,3), (2,3)（此时还不到合并它们的时候）
- Action 2 的正例 pair：({0,1,2}, 3)

**344 个 tree actions 覆盖了 90.6% 的 composite token**（366/404），即大多数 tree action 的 parent 集合在 composite token 中有对应的 SVG 特征。

---

## 六、`manual_step_groups` — 说明书步骤标注

来自 `main_data.json` 中 `steps` 字段的精选信息：

```json
[
    {
        "step_id": 0,
        "parts": ["0", "2", "1"],                    // 本步骤涉及的零件
        "connections": [["0", "2"], ["1", "2"]]       // 本步骤中的连接关系
    },
    {
        "step_id": 1,
        "parts": ["0,1,2", "3"],                     // 子装配体 + 零件 3
        "connections": [["0,1,2", "3"]]
    }
]
```

**用途**：GRPO 的 SVG_coherence reward 检查预测的树节点是否出现在 manual step 的分组里。

---

## 七、`connection_relation` — 零件连接图（GNN 输入）

```json
[[0, 2], [0, 3], [1, 2], [1, 3]]
```

无向边列表。在 GNN 中被转换为 `edge_index [2, E]` 作为 GraphSAGE 的消息传递图。

```
     0(座板)
    / \
   2   3(横梁)
  /
 1(侧板)
```

> 注：此图是对象级的完整连接关系，不包含"在哪个步骤连接"的时序信息。

---

## 八、`geometric_equivalence_relation` — 几何等价关系

```json
{"0": ["0"], "1": ["1"], "2": ["2"]}
```

applaro 的每个零件形状都不同，各自只等价于自己。

对比 hemnes（12 零件，4 条相同桌腿）：

```json
{
    "6": ["7", "6", "9", "8"],    // 4 条桌腿相互等价
    "7": ["7", "6", "9", "8"],
    "8": ["7", "6", "9", "8"],
    "9": ["7", "6", "9", "8"]
}
```

**来源**：`main_data.json` 人工标注。  
**用途**：原始论文的核心洞察——形状相似的零件通常在装配树底层先聚类。当前模型未使用此信息，是潜在的特征增强方向。

---

## 九、特征维度速查表

| 特征组 | 来源 | 维度 | 描述 |
|------|------|:---:|------|
| `geometry_feature` | OBJ 3D 模型 | 14 | 包围盒尺寸、比例、体积、复杂度 |
| `svg_feature_mean` | 说明书 SVG 图 | 17 | 2D 空间位置、面积、主轴、形状类型 |
| `svg_feature_std` | 说明书 SVG 图 | 17 | 跨步骤特征方差（大部分为 0） |
| `svg_feature_count` | 说明书 SVG 图 | 1 | 跨步骤出现次数 |
| `shape_distribution` | 说明书 SVG 图 | 4 | 形状类型概率分布 |
| **→ 模型输入（基本）** | | **35** | geometry(14) + svg_mean(17) + shape(4) |
| `composite_tokens` | 说明书子装配 SVG | 23 | ⚠️ Manual 泄露（可选） |

---

## 十、构建流程命令速查

```powershell
# 1. 解析 SVG 为结构化特征
python scripts/build/build_svg_features.py

# 2. 简化 SVG 几何（生成 simplified_svg/）
python scripts/build/simplify_svg_instances.py

# 3. 渲染零件 3D 模型为多视角图
python scripts/build/render_part_images.py --views 4

# 4. 构建 grounding 数据集
python scripts/build/build_grounding_dataset.py

# 5. 构建对象级树生成数据集（最终产物）
python scripts/build/build_tree_generation_dataset.py
```
