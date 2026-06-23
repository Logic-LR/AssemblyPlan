# IKEA-Manual Assembly Tree Generation

SVG-enhanced assembly tree generation from IKEA furniture instruction manuals.
Given N parts (3D geometry + SVG spatial features), predict the hierarchical
assembly plan — e.g. `[[0, 1, 2], 3]` means "merge {0,1,2} first, then add 3".

## Repository Structure

```
├── README.md                          ← you are here
├── EXPERIMENT_REPORT.md               # GNN + GRPO experiment report (assembly_plan)
├── IKEA-Manual Dataset 详细总结报告.md  # Full technical summary (Chinese)
├── main_data.json                     # Core annotations (steps, connections, trees)
│
├── code/                              # Original baseline implementations
│   ├── manual_generation/             # Manual plan generation (DGCNN + K-Means)
│   └── part_assembly/                 # NeurIPS 2020: Generative 3D Part Assembly
│
├── scripts/                           # SVG assembly experiment tooling
│   ├── build/                         # Data construction & feature extraction (7 scripts)
│   ├── train/                         # Model training (12 scripts)
│   ├── eval/                          # Evaluation & inference (5 scripts)
│   └── export/                        # Export & analysis (4 scripts)
│
├── assembly_plan/                     # GNN + GraphSAGE tree planner (current)
│   ├── model.py                       # GNNMergeModel (GraphSAGE + MLP scorer)
│   ├── decoder.py                     # Group-aware k-ary tree decoder
│   ├── train.py / run.py              # Training & inference entry points
│   └── EXPERIMENT_REPORT.md           # Detailed results
│
├── vlm_distill/                       # VLM distillation pipeline (proposed)
│   ├── DESIGN.md / DESIGN_CN.md       # Design documents
│   └── test_vlm_assembly.py           # GPT-4o calibration script
│
├── experiments/svg_assembly/          # Experiment hub
│   ├── METHODS_ANALYSIS.md            # 6 methods, root-cause analysis
│   ├── EXPERIMENT_REPORT_CN.md        # Presentation-ready summary (Chinese)
│   ├── IMPROVEMENT_REPORT.md          # Grounding CNN improvements
│   ├── datasets/                      # Built datasets + field documentation
│   │   └── DATASET_FIELDS.md          # Per-feature-dimension explanation
│   └── reports/                       # Experiment result summaries
│
├── data/                              # Raw XML data samples
├── assembly_trees/                    # GT assembly tree exports
└── line_seg/                          # SVG line segmentation (per-step)
```

## Dataset

- **102 IKEA furniture objects** (73 train / 29 test)
- 754 primitive parts, 404 tree actions, 393 manual steps
- Per-part features: **34-dim** (14 geometry + 17 SVG spatial + 4 shape type)
- See `experiments/svg_assembly/datasets/DATASET_FIELDS.md` for full field documentation.

## Key Results

| Model | Hard F1 | Notes |
|------|:---:|------|
| BCE Context MLP + composite | 0.332 | Best supervised, but composite tokens leak manual answers |
| BCE Context MLP (pure features) | 0.108 | No manual info |
| GNN + GRPO + Group decoder | **0.204** | Current best, produces real nested trees |
| GRPO SVG-only (0% GT labels) | ~0.346 | 3-seed mean, high variance |

66% of BCE predictions are flat trees (single-step merge of all parts).
See `experiments/svg_assembly/EXPERIMENT_REPORT_CN.md` for the full presentation.

## Quick Start

```powershell
# Environment
.venv\Scripts\python.exe -m pip install -r requirements-ml.txt

# Build SVG features
python scripts/build/build_svg_features.py

# Build tree generation dataset
python scripts/build/build_tree_generation_dataset.py

# Train GNN planner
python assembly_plan/train.py --feature-mode svg_geometry

# Run VLM calibration (requires OPENAI_API_KEY)
python vlm_distill/test_vlm_assembly.py --num_samples 5
```

## Documentation Index

| Document | Language | Content |
|------|------|------|
| `IKEA-Manual Dataset 详细总结报告.md` | CN | Complete project survey (11 chapters) |
| `experiments/svg_assembly/METHODS_ANALYSIS.md` | EN | 6 methods, problems, root causes |
| `experiments/svg_assembly/EXPERIMENT_REPORT_CN.md` | CN | Presentation-ready summary |
| `EXPERIMENT_REPORT.md` | EN | GNN + GRPO experiment report |
| `experiments/svg_assembly/datasets/DATASET_FIELDS.md` | CN | Feature dimension documentation |
| `vlm_distill/DESIGN_CN.md` | CN | VLM distillation design |
