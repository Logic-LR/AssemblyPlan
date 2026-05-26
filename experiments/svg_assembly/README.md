# SVG Assembly Experiments

This folder tracks experiments for SVG-aware assembly understanding on
IKEA-Manual.

## Current Pipeline

1. Parse `line_seg/**/*.svg` into structured vector features.
2. Build a step index and spatial sanity-check baselines.
3. Recover SVG color-instance to primitive-part alignment from released masks.
4. Train a small pairwise connection classifier.
5. Render OBJ primitives into multi-view part images.
6. Build part-to-SVG grounding samples and train small grounding baselines.
7. Build object-level tree-generation samples and train SVG-enhanced planner baselines.

## Environment

The current local environment is:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-ml.txt
```

Installed core packages include CPU PyTorch, torchvision, scikit-learn, numpy,
Pillow, tqdm, and matplotlib.

## Commands

Generate SVG features:

```powershell
python scripts\build\build_svg_features.py
```

Build the index:

```powershell
python scripts\build\build_svg_feature_index.py
```

Run spatial baselines:

```powershell
python scripts\eval\eval_spatial_connection_baseline.py
```

Recover mask-derived visual-instance alignment:

```powershell
python scripts\export\analyze_instance_mask_alignment.py
```

Train pairwise connection model:

```powershell
python scripts\train\train_pairwise_connection_model.py --align mask
```

Generate simplified SVG geometry overlays:

```powershell
python scripts\build\simplify_svg_instances.py
```

The simplification output is written under:

```text
experiments/svg_assembly/simplified_svg/
```

Each step folder contains `simplified_instances.json`, per-instance JSON files,
and `overlay.svg`. The overlay draws three representations for each visual
instance: convex hull, RDP-simplified polygon, and principal axis.

Render primitive part images from OBJ files:

```powershell
python scripts\build\render_part_images.py --views 4
```

This writes:

```text
part_images/<category>/<name>/<part_id>/view_*.png
part_images/part_index.jsonl
```

Build grounding samples that link rendered parts to simplified SVG instances:

```powershell
python scripts\build\build_grounding_dataset.py
```

This writes:

```text
experiments/svg_assembly/datasets/grounding_samples.jsonl
experiments/svg_assembly/datasets/grounding_summary.json
```

Build object-level tree-generation samples:

```powershell
python scripts\build\build_tree_generation_dataset.py
```

This writes:

```text
experiments/svg_assembly/datasets/tree_generation_dataset.json
experiments/svg_assembly/datasets/tree_generation_summary.json
```

Each object record contains primitive part tokens, aggregated primitive SVG
prototypes, manual-derived composite/subassembly SVG prototypes, the target
assembly tree, post-order tree actions, manual step groups, and
geometric-equivalence metadata.

Train grounding baselines:

```powershell
python scripts\train\train_grounding_model.py --primitive-only
python scripts\train\train_grounding_model.py
python scripts\train\train_grounding_image_model.py --primitive-only
python scripts\train\train_grounding_image_model.py
python scripts\train\train_grounding_cnn.py --epochs 25 --output experiments\svg_assembly\reports\grounding_cnn_all_val_report.json --model-output experiments\svg_assembly\reports\grounding_cnn_all_val_model.pt
python scripts\train\train_grounding_cnn.py --augment --epochs 25 --output experiments\svg_assembly\reports\grounding_cnn_all_val_aug_report.json --model-output experiments\svg_assembly\reports\grounding_cnn_all_val_aug_model.pt
python scripts\train\train_grounding_cnn.py --equivalence-labels --epochs 25 --output experiments\svg_assembly\reports\grounding_cnn_all_equiv_val_report.json --model-output experiments\svg_assembly\reports\grounding_cnn_all_equiv_val_model.pt
python scripts\train\train_grounding_cnn.py --equivalence-labels --epochs 20 --max-images 8 --dropout 0.2 --output experiments\svg_assembly\reports\grounding_cnn_improved_val_report.json --model-output experiments\svg_assembly\reports\grounding_cnn_improved_val_model.pt
```

Train object-level tree planner baselines:

```powershell
python scripts\train\train_tree_planner_baseline.py --feature-mode geometry --output experiments\svg_assembly\reports\tree_planner_geometry_report.json --model-output experiments\svg_assembly\reports\tree_planner_geometry_model.npz
python scripts\train\train_tree_planner_baseline.py --feature-mode svg --output experiments\svg_assembly\reports\tree_planner_svg_report.json --model-output experiments\svg_assembly\reports\tree_planner_svg_model.npz
python scripts\train\train_tree_planner_baseline.py --feature-mode svg_geometry --output experiments\svg_assembly\reports\tree_planner_svg_geometry_report.json --model-output experiments\svg_assembly\reports\tree_planner_svg_geometry_model.npz
python scripts\train\train_tree_planner_baseline.py --feature-mode svg_geometry_composite --output experiments\svg_assembly\reports\tree_planner_svg_geometry_composite_report.json --model-output experiments\svg_assembly\reports\tree_planner_svg_geometry_composite_model.npz
```

Train the first neural merge-scorer planner baselines:

```powershell
python scripts\train\train_tree_planner_nn.py --feature-mode geometry --epochs 120 --output experiments\svg_assembly\reports\tree_planner_nn_geometry_report.json --model-output experiments\svg_assembly\reports\tree_planner_nn_geometry_model.pt --pred-output-dir experiments\svg_assembly\tree_planner_predictions_nn_geometry_test
python scripts\train\train_tree_planner_nn.py --feature-mode svg --epochs 120 --output experiments\svg_assembly\reports\tree_planner_nn_svg_report.json --model-output experiments\svg_assembly\reports\tree_planner_nn_svg_model.pt --pred-output-dir experiments\svg_assembly\tree_planner_predictions_nn_svg_test
python scripts\train\train_tree_planner_nn.py --feature-mode svg_geometry --epochs 120 --output experiments\svg_assembly\reports\tree_planner_nn_svg_geometry_report.json --model-output experiments\svg_assembly\reports\tree_planner_nn_svg_geometry_model.pt --pred-output-dir experiments\svg_assembly\tree_planner_predictions_nn_svg_geometry_test
python scripts\train\train_tree_planner_nn.py --feature-mode svg_composite --epochs 120 --output experiments\svg_assembly\reports\tree_planner_nn_svg_composite_report.json --model-output experiments\svg_assembly\reports\tree_planner_nn_svg_composite_model.pt --pred-output-dir experiments\svg_assembly\tree_planner_predictions_nn_svg_composite_test
python scripts\train\train_tree_planner_nn.py --feature-mode svg_geometry_composite --epochs 120 --output experiments\svg_assembly\reports\tree_planner_nn_svg_geometry_composite_report.json --model-output experiments\svg_assembly\reports\tree_planner_nn_svg_geometry_composite_model.pt --pred-output-dir experiments\svg_assembly\tree_planner_predictions_nn_svg_geometry_composite_test
python scripts\eval\evaluate_composite_context_decoder.py
python scripts\train\train_subassembly_candidate_model.py --feature-mode svg_geometry --epochs 40 --output experiments\svg_assembly\reports\subassembly_candidate_svg_geometry_report.json --model-output experiments\svg_assembly\reports\subassembly_candidate_svg_geometry_model.pt --pred-output-dir experiments\svg_assembly\tree_planner_predictions_subassembly_candidates_svg_geometry_test
python scripts\train\train_subassembly_candidate_model.py --feature-mode svg_geometry --epochs 60 --train-negatives-per-positive 20 --selection-penalty 0.001 --output experiments\svg_assembly\reports\subassembly_candidate_svg_geometry_neg20_report.json --model-output experiments\svg_assembly\reports\subassembly_candidate_svg_geometry_neg20_model.pt --pred-output-dir experiments\svg_assembly\tree_planner_predictions_subassembly_candidates_svg_geometry_neg20_test
```

The neural planner uses an MLP merge scorer, but it still decodes by thresholded
connected-component merges. Treat it as a stronger scorer baseline, not yet as a
full learned set/graph tree decoder.
The `composite` modes additionally consume manual-derived subassembly SVG
prototypes, so they are manual/RAG-context baselines rather than pure
observed-part inference. `evaluate_composite_context_decoder.py` is an oracle
sanity check: it shows that manual composite tokens can directly encode the
tree, so they must be treated as supervision or retrieved manual context.
`train_subassembly_candidate_model.py` is the first no-leakage attempt to
predict subassembly candidates from primitive part tokens only. It supports
negative sampling and a decoder selection penalty so recall-heavy and
precision-oriented candidate generators can be compared.

Export tree-planner predictions:

```powershell
python scripts\export\export_tree_planner_predictions.py --model experiments\svg_assembly\reports\tree_planner_svg_model.npz --split test --output-dir experiments\svg_assembly\tree_planner_predictions_svg_test --output-report experiments\svg_assembly\reports\tree_planner_svg_predictions_test_report.json
python scripts\export\export_tree_planner_predictions.py --model experiments\svg_assembly\reports\tree_planner_svg_geometry_model.npz --split test --output-dir experiments\svg_assembly\tree_planner_predictions_svg_geometry_test --output-report experiments\svg_assembly\reports\tree_planner_svg_geometry_predictions_test_report.json
python scripts\export\export_tree_planner_predictions.py --model experiments\svg_assembly\reports\tree_planner_svg_geometry_composite_model.npz --split test --output-dir experiments\svg_assembly\tree_planner_predictions_svg_geometry_composite_test --output-report experiments\svg_assembly\reports\tree_planner_svg_geometry_composite_predictions_test_report.json
```

Run planner inference from observed or grounded part tokens:

```powershell
python scripts\eval\run_tree_planner_inference.py --input observed_part_tokens.json --output predicted_assembly_tree.json --model experiments\svg_assembly\reports\tree_planner_nn_svg_geometry_model.pt
```

The input can be a list of part-token objects, or an object with
`observed_part_tokens` or `part_tokens`. Each token should contain the canonical
features used by the planner: `geometry_feature`, `svg_feature_mean`,
`shape_distribution`, and `svg_feature_count`. The output contains the predicted
tree plus an internal-to-input part-id mapping.

Build an HTML error viewer for planner failures:

```powershell
python scripts\build\build_tree_planner_error_viewer.py --prediction-dir experiments\svg_assembly\tree_planner_predictions_nn_svg_geometry_test --output-html experiments\svg_assembly\reports\tree_planner_error_viewer_nn_svg_geometry.html --output-json experiments\svg_assembly\reports\tree_planner_error_viewer_nn_svg_geometry.json
python scripts\build\build_tree_planner_error_viewer.py --prediction-dir experiments\svg_assembly\tree_planner_predictions_nn_svg_geometry_composite_test --output-html experiments\svg_assembly\reports\tree_planner_error_viewer_nn_svg_geometry_composite.html --output-json experiments\svg_assembly\reports\tree_planner_error_viewer_nn_svg_geometry_composite.json
python scripts\build\build_tree_planner_error_viewer.py --prediction-dir experiments\svg_assembly\tree_planner_predictions_subassembly_candidates_svg_geometry_test --output-html experiments\svg_assembly\reports\tree_planner_error_viewer_subassembly_candidates_svg_geometry.html --output-json experiments\svg_assembly\reports\tree_planner_error_viewer_subassembly_candidates_svg_geometry.json
python scripts\build\build_tree_planner_error_viewer.py --prediction-dir experiments\svg_assembly\tree_planner_predictions_subassembly_candidates_svg_geometry_neg20_test --output-html experiments\svg_assembly\reports\tree_planner_error_viewer_subassembly_candidates_svg_geometry_neg20.html --output-json experiments\svg_assembly\reports\tree_planner_error_viewer_subassembly_candidates_svg_geometry_neg20.json
```

Summarize all available reports:

```powershell
python scripts\export\summarize_svg_assembly_experiments.py
```

Run the current end-to-end diagnostic:

```powershell
python scripts\eval\run_end_to_end_diagnostic.py
python scripts\eval\run_end_to_end_diagnostic.py --split all --output experiments\svg_assembly\reports\end_to_end_diagnostic_all_report.json
python scripts\eval\run_end_to_end_diagnostic.py --grounding-model experiments\svg_assembly\reports\grounding_cnn_all_equiv_val_model.pt --split test --output experiments\svg_assembly\reports\end_to_end_diagnostic_equiv_test_report.json
python scripts\eval\run_end_to_end_diagnostic.py --grounding-model experiments\svg_assembly\reports\grounding_cnn_all_equiv_val_model.pt --split all --output experiments\svg_assembly\reports\end_to_end_diagnostic_equiv_all_report.json
python scripts\eval\run_end_to_end_diagnostic.py --grounding-model experiments\svg_assembly\reports\grounding_cnn_improved_val_model.pt --split test --output experiments\svg_assembly\reports\end_to_end_diagnostic_improved_report.json
python scripts\eval\evaluate_paper_tree_metrics.py
python scripts\export\export_tree_predictions_and_equivalence_report.py
python scripts\export\export_tree_predictions_and_equivalence_report.py --diagnostic-report experiments\svg_assembly\reports\end_to_end_diagnostic_report.json --output-report experiments\svg_assembly\reports\equivalence_test_report.json --output-dir experiments\svg_assembly\predicted_assembly_trees_test
python scripts\export\export_tree_predictions_and_equivalence_report.py --diagnostic-report experiments\svg_assembly\reports\end_to_end_diagnostic_improved_report.json --output-report experiments\svg_assembly\reports\equivalence_test_improved_report.json --output-dir experiments\svg_assembly\predicted_assembly_trees_improved_test
python scripts\export\export_tree_predictions_and_equivalence_report.py --diagnostic-report experiments\svg_assembly\reports\end_to_end_diagnostic_equiv_all_report.json --output-report experiments\svg_assembly\reports\equivalence_and_tree_export_equiv_model_report.json --output-dir experiments\svg_assembly\predicted_assembly_trees_equiv
```

This writes:

```text
experiments/svg_assembly/experiment_summary.json
experiments/svg_assembly/experiment_summary.md
experiments/svg_assembly/reports/end_to_end_diagnostic_report.json
experiments/svg_assembly/reports/end_to_end_diagnostic_all_report.json
experiments/svg_assembly/reports/end_to_end_diagnostic_improved_report.json
experiments/svg_assembly/reports/paper_tree_metric_report.json
experiments/svg_assembly/reports/equivalence_and_tree_export_report.json
experiments/svg_assembly/reports/equivalence_test_report.json
experiments/svg_assembly/reports/equivalence_test_improved_report.json
experiments/svg_assembly/reports/tree_planner_geometry_report.json
experiments/svg_assembly/reports/tree_planner_svg_report.json
experiments/svg_assembly/reports/tree_planner_svg_geometry_report.json
experiments/svg_assembly/reports/tree_planner_svg_geometry_composite_report.json
experiments/svg_assembly/reports/tree_planner_nn_geometry_report.json
experiments/svg_assembly/reports/tree_planner_nn_svg_report.json
experiments/svg_assembly/reports/tree_planner_nn_svg_geometry_report.json
experiments/svg_assembly/reports/tree_planner_nn_svg_composite_report.json
experiments/svg_assembly/reports/tree_planner_nn_svg_geometry_composite_report.json
experiments/svg_assembly/reports/tree_planner_composite_context_report.json
experiments/svg_assembly/reports/subassembly_candidate_svg_geometry_report.json
experiments/svg_assembly/reports/subassembly_candidate_svg_geometry_precision_report.json
experiments/svg_assembly/reports/subassembly_candidate_svg_geometry_neg20_report.json
experiments/svg_assembly/reports/tree_planner_inference_smoke.json
experiments/svg_assembly/reports/tree_planner_error_viewer_nn_svg_geometry.html
experiments/svg_assembly/reports/tree_planner_error_viewer_nn_svg_geometry.json
experiments/svg_assembly/reports/tree_planner_error_viewer_nn_svg_geometry_composite.html
experiments/svg_assembly/reports/tree_planner_error_viewer_nn_svg_geometry_composite.json
experiments/svg_assembly/reports/tree_planner_error_viewer_subassembly_candidates_svg_geometry.html
experiments/svg_assembly/reports/tree_planner_error_viewer_subassembly_candidates_svg_geometry.json
experiments/svg_assembly/reports/tree_planner_error_viewer_subassembly_candidates_svg_geometry_neg20.html
experiments/svg_assembly/reports/tree_planner_error_viewer_subassembly_candidates_svg_geometry_neg20.json
experiments/svg_assembly/predicted_assembly_trees/
experiments/svg_assembly/predicted_assembly_trees_improved_test/
experiments/svg_assembly/predicted_assembly_trees_equiv/
experiments/svg_assembly/tree_planner_predictions_svg_test/
experiments/svg_assembly/tree_planner_predictions_svg_geometry_test/
experiments/svg_assembly/tree_planner_predictions_svg_geometry_composite_test/
experiments/svg_assembly/tree_planner_predictions_nn_geometry_test/
experiments/svg_assembly/tree_planner_predictions_nn_svg_test/
experiments/svg_assembly/tree_planner_predictions_nn_svg_geometry_test/
experiments/svg_assembly/tree_planner_predictions_nn_svg_composite_test/
experiments/svg_assembly/tree_planner_predictions_nn_svg_geometry_composite_test/
experiments/svg_assembly/tree_planner_predictions_composite_context_test/
experiments/svg_assembly/tree_planner_predictions_subassembly_candidates_svg_geometry_test/
experiments/svg_assembly/tree_planner_predictions_subassembly_candidates_svg_geometry_precision_test/
experiments/svg_assembly/tree_planner_predictions_subassembly_candidates_svg_geometry_neg20_test/
```

## Key Findings

- `line_seg` color instances match `main_data.json` step parts in count for all
  393 steps.
- Pixel masks and RLE masks align perfectly for all 393 steps, so
  color-instance to primitive-part supervision can be recovered.
- Heuristic alignment is imperfect: `area_desc` exact color-to-part alignment is
  about 49.4%.
- With mask-derived alignment, the pairwise classifier reaches perfect test
  step top-k connection recovery on the released 40-step test split.
- Using only simplified SVG geometry also preserves most connection signal:
  simplified step top-k test F1 is 0.9875 versus 1.0000 for the original
  sampled SVG features.
- Part-side preprocessing currently renders 754 primitive OBJ parts into
  synthetic multi-view images. The grounding dataset contains 1056 SVG-instance
  samples: 754 primitive samples and 302 composite/subassembly samples.
- Grounding is the current bottleneck. With the improved residual CNN,
  all-sample strict test assignment instance accuracy is about 0.706 and
  equivalence-aware test assignment is about 0.849. The model improves
  downstream connection F1 more than raw grounding accuracy, which suggests it
  fixes several high-impact grounding cases.
- Light image augmentation slightly improves pairwise F1 but does not improve
  final assignment accuracy in the current setting. This suggests the main
  grounding gap is not just renderer overfitting; composite/subassembly
  representation and task formulation likely matter more.
- The current end-to-end diagnostic reaches test connection F1 0.8875 with
  improved grounding, versus 0.9875 with oracle grounding. The equivalence-aware
  connection F1 is 0.9375. This means many grounding ID swaps are harmless for
  the final connection tree, especially for symmetric or equivalent parts.
- Under the IKEA-Manual paper's assembly-tree metrics, the current
  connection-induced tree obtains Simple F1 about 0.997 and Hard F1 about
  0.994 on all 102 objects. This is not directly comparable to the paper's
  shape-only assembly-plan generation setting, because our diagnostic uses
  manual step SVGs and mostly tests whether predicted connections preserve the
  step-level part grouping.
- Equivalence-aware evaluation changes the interpretation of grounding. On the
  full export split with the equivalence-label model, strict grounding instance
  accuracy is about 0.757, equivalence-aware grounding accuracy is about 0.809,
  strict connection F1 is about 0.949, and equivalence-aware connection F1 is
  about 0.954.
- Full predicted assembly-tree JSON files are exported under
  `experiments/svg_assembly/predicted_assembly_trees_equiv/`, one folder per
  object.
- Object-level tree generation has an initial baseline that does not read
  per-step SVGs at inference. On the 29-object test split, geometry-only greedy
  planning gets Simple/Hard F1 about 0.396/0.034, while SVG-only gets about
  0.447/0.123 and SVG+geometry gets about 0.412/0.126. This is the first
  direct evidence that manual-derived SVG prototypes can improve tree planning,
  although the decoder is still only a greedy baseline.
- The first neural merge-scorer planner is implemented. On the same test split,
  geometry-only gets Simple/Hard F1 about 0.429/0.162, SVG-only gets about
  0.409/0.082, and SVG+geometry gets about 0.477/0.156. The current result
  improves Simple F1 with SVG+geometry but does not prove a clean SVG-only gain
  under the neural scorer, so the next step should improve the decoder and
  subassembly representation before making a stronger claim.
- Composite/subassembly SVG prototypes are now included as optional manual/RAG
  context. They cover about 90.6% of tree-action parent sets. With neural
  SVG+geometry+composite features, test Simple/Hard F1 is about 0.488/0.206.
  This is encouraging, but it should be reported separately from the pure
  observed-part setting because those composite prototypes come from manual
  middle steps.
- A context-only composite decoder reaches 1.000/1.000 Simple/Hard F1, which is
  the expected warning sign: manual composite part sets can reveal the assembly
  tree. Use them as supervision or retrieved manual context, not as ordinary
  no-manual inference input.
- The first no-leakage subassembly candidate model predicts candidates from
  primitive geometry+SVG tokens only. It reaches test candidate precision/recall
  about 0.009/0.667 and downstream Simple/Hard F1 about 0.287/0.068. This says
  the model can recover some true subassemblies, but the candidate set is far
  too noisy for the current decoder.
- A negative-sampling version of that candidate model improves the downstream
  tree score. With 20 negatives per positive and a small selection penalty, it
  selects about 6.03 candidate sets per test object, gets decoder-threshold
  candidate F1 about 0.022, and reaches Simple/Hard F1 about 0.395/0.113. This
  is better than the recall-heavy candidate row, but still not enough to replace
  manual/RAG composite context.
- `scripts/run_tree_planner_inference.py` is now the target application entry point
  for this module: after a grounding model turns robot observations into
  canonical part tokens, this script maps those tokens to a predicted assembly
  tree.

## Important Caveat

Mask-derived alignment uses released annotations, so it is a diagnostic and
supervised training signal. A real inference pipeline still needs a grounding
module that maps visual SVG instances to input primitive part identities.
Composite/subassembly samples do not correspond to one primitive OBJ; they are
represented by the set of primitive parts that make up the subassembly.

Train the first geometric grounding baseline:

```powershell
python scripts\train\train_grounding_model.py --primitive-only
python scripts\train\train_grounding_model.py
```

Current geometric-only grounding is intentionally modest: on the test split,
primitive-only assignment instance accuracy is about 0.471, and all-sample
assignment instance accuracy is about 0.529. This suggests simplified SVG
geometry alone is not enough for robust part-to-SVG grounding; the next model
should add rendered part-image features or stronger 3D shape features.

Rendered-image silhouette features were also tested:

```powershell
python scripts\train\train_grounding_image_model.py --primitive-only
python scripts\train\train_grounding_image_model.py
```

The current linear image-feature model does not improve all-sample grounding.
Primitive-only exact assignment improves slightly, but all-sample assignment
drops, likely because the rendered-image silhouette vector is high-dimensional
and the dataset is small. This points toward either a genuinely small CNN with
augmentation or stronger 3D descriptors, rather than a raw linear image
silhouette baseline.

The improved residual CNN + geometry baseline with equivalence-aware labels is
the strongest current grounding direction, while the object-level planner is
the right next modeling direction for the thesis task. The next useful
improvements are a stronger set/graph tree decoder, a much more precise
subassembly candidate generator, and a real-image or 3D-observation benchmark
for grounding.

For the intended robot application, report both grounding metrics and final
tree/connection metrics. Instance ID accuracy alone can underestimate useful
performance when equivalent legs, slats, or repeated connectors are swapped.
