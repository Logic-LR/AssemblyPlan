# SVG Feature Builder

`build_svg_features.py` converts IKEA-Manual `line_seg/**/*.svg` files into
structured vector features for SVG-aware assembly experiments. It does not infer
assembly trees; it attaches ground-truth step annotations from `main_data.json`.

## Usage

Build all step features:

```powershell
python scripts\build\build_svg_features.py
```

Build a single object:

```powershell
python scripts\build\build_svg_features.py --category Bench --name applaro
```

Useful options:

```powershell
python scripts\build\build_svg_features.py --samples-per-segment 8 --max-points-per-instance 256
python scripts\build\build_svg_features.py --include-raw-d
```

## Output

Files are written under:

```text
svg_features/<category>/<name>/step_<id>.json
```

Each JSON contains:

- `canvas`: SVG width, height, and viewBox when present.
- `instances`: one entry per stroke color, treated only as a step-level visual
  part instance. Stroke color is not a semantic part identity and must not be
  matched across steps.
- `sampled_points`: downsampled points sampled from the original SVG paths.
- `bbox`, `center`, `principal_axis`, `axis_length`, `axis_width`: geometric
  features for each visual instance.
- `spatial_relations`: pairwise bbox overlap, bbox distance, center delta, and
  sampled point distance between visual instances.
- `gt`: step-level supervision from `main_data.json`, including `parts` and
  `connections`.
- `object_gt`: object-level `assembly_tree`, final connection relation, and
  geometric equivalence relation.

## Current Approximation

The path parser supports common SVG commands (`M/L/H/V/C/S/Q/T/A/Z`) and samples
curves. Elliptical arcs (`A/a`) are approximated by interpolation between their
endpoints. This is sufficient for coarse geometry features but not exact SVG
rendering.

## Index and Baseline

Build a step-level index:

```powershell
python scripts\build\build_svg_feature_index.py
```

This writes:

```text
svg_features/index.jsonl
svg_features/index_summary.json
```

Run a simple spatial connection baseline:

```powershell
python scripts\eval\eval_spatial_connection_baseline.py
```

This writes:

```text
svg_features/spatial_baseline_report.json
```

The baseline is only a sanity check. It assumes visual instances can be aligned
to `gt.parts` by a deterministic order and predicts the top-k pairwise
connections from spatial features.

## Pairwise Connection Model

Train a small numpy logistic-regression classifier:

```powershell
python scripts\train\train_pairwise_connection_model.py
```

By default it uses `mask` alignment between visual instances and `gt.parts`.
Generate that alignment first:

```powershell
python scripts\export\analyze_instance_mask_alignment.py
```

The mask-derived alignment is appropriate for supervised training/evaluation
diagnostics because it comes from released IKEA-Manual annotations. It should
not be confused with an inference-time grounding model.

Try heuristic alignments with:

```powershell
python scripts\train\train_pairwise_connection_model.py --align feature_order
python scripts\train\train_pairwise_connection_model.py --align x_center
python scripts\train\train_pairwise_connection_model.py --align y_center
python scripts\train\train_pairwise_connection_model.py --align area_desc
```

Outputs:

```text
svg_features/pairwise_connection_report.json
svg_features/pairwise_connection_model.npz
```

## Simplified SVG Geometry

Generate inspectable simplified geometries:

```powershell
python scripts\build\simplify_svg_instances.py
```

For each step, this writes:

```text
experiments/svg_assembly/simplified_svg/<category>/<name>/step_<id>/overlay.svg
experiments/svg_assembly/simplified_svg/<category>/<name>/step_<id>/simplified_instances.json
```

Each instance keeps multiple simplified forms rather than a single lossy
representation: `convex_hull`, `simplified_polygon`, `oriented_bbox`, and
`principal_axis`.

Evaluate the simplified representation:

```powershell
python scripts\train\train_simplified_connection_model.py
```

This writes:

```text
experiments/svg_assembly/reports/simplified_connection_report.json
```

## Part Images and Grounding Samples

Render synthetic multi-view images from primitive OBJ parts:

```powershell
python scripts\build\render_part_images.py --views 4
```

Build samples for part-to-SVG-instance grounding:

```powershell
python scripts\build\build_grounding_dataset.py
```

Outputs:

```text
part_images/part_index.jsonl
experiments/svg_assembly/datasets/grounding_samples.jsonl
experiments/svg_assembly/datasets/grounding_summary.json
```

The grounding dataset includes primitive samples and composite/subassembly
samples. Composite samples reference multiple primitive parts because later
manual steps often show an already assembled subassembly.

## Object-Level Tree Generation

Build the planner dataset:

```powershell
python scripts\build\build_tree_generation_dataset.py
```

This converts the manual supervision into object-level records:

```text
experiments/svg_assembly/datasets/tree_generation_dataset.json
experiments/svg_assembly/datasets/tree_generation_summary.json
```

Each record contains primitive part tokens, aggregated primitive SVG prototypes,
manual-derived composite/subassembly SVG prototypes, the ground-truth assembly
tree, post-order tree actions, and manual step groups. The primitive-token
fields support the target setting where the planner receives all primitive part
tokens and predicts an assembly tree without reading per-step SVGs at inference
time. The composite-token fields are optional manual/RAG context because they
come from manual middle steps.

Train initial greedy tree-planner baselines:

```powershell
python scripts\train\train_tree_planner_baseline.py --feature-mode geometry --output experiments\svg_assembly\reports\tree_planner_geometry_report.json --model-output experiments\svg_assembly\reports\tree_planner_geometry_model.npz
python scripts\train\train_tree_planner_baseline.py --feature-mode svg --output experiments\svg_assembly\reports\tree_planner_svg_report.json --model-output experiments\svg_assembly\reports\tree_planner_svg_model.npz
python scripts\train\train_tree_planner_baseline.py --feature-mode svg_geometry --output experiments\svg_assembly\reports\tree_planner_svg_geometry_report.json --model-output experiments\svg_assembly\reports\tree_planner_svg_geometry_model.npz
python scripts\train\train_tree_planner_baseline.py --feature-mode svg_geometry_composite --output experiments\svg_assembly\reports\tree_planner_svg_geometry_composite_report.json --model-output experiments\svg_assembly\reports\tree_planner_svg_geometry_composite_model.npz
```

Train neural merge-scorer planner baselines:

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

Export predicted trees:

```powershell
python scripts\export\export_tree_planner_predictions.py --model experiments\svg_assembly\reports\tree_planner_svg_model.npz --split test --output-dir experiments\svg_assembly\tree_planner_predictions_svg_test --output-report experiments\svg_assembly\reports\tree_planner_svg_predictions_test_report.json
python scripts\export\export_tree_planner_predictions.py --model experiments\svg_assembly\reports\tree_planner_svg_geometry_model.npz --split test --output-dir experiments\svg_assembly\tree_planner_predictions_svg_geometry_test --output-report experiments\svg_assembly\reports\tree_planner_svg_geometry_predictions_test_report.json
python scripts\export\export_tree_planner_predictions.py --model experiments\svg_assembly\reports\tree_planner_svg_geometry_composite_model.npz --split test --output-dir experiments\svg_assembly\tree_planner_predictions_svg_geometry_composite_test --output-report experiments\svg_assembly\reports\tree_planner_svg_geometry_composite_predictions_test_report.json
```

Run the planner on observed or grounded part tokens:

```powershell
python scripts\eval\run_tree_planner_inference.py --input observed_part_tokens.json --output predicted_assembly_tree.json --model experiments\svg_assembly\reports\tree_planner_nn_svg_geometry_model.pt
```

The inference input can be a list of part-token objects, or an object containing
`observed_part_tokens` or `part_tokens`. It is the planned interface between the
robot grounding module and this assembly-tree planner.

Build an error viewer for predicted trees:

```powershell
python scripts\build\build_tree_planner_error_viewer.py --prediction-dir experiments\svg_assembly\tree_planner_predictions_nn_svg_geometry_test --output-html experiments\svg_assembly\reports\tree_planner_error_viewer_nn_svg_geometry.html --output-json experiments\svg_assembly\reports\tree_planner_error_viewer_nn_svg_geometry.json
python scripts\build\build_tree_planner_error_viewer.py --prediction-dir experiments\svg_assembly\tree_planner_predictions_nn_svg_geometry_composite_test --output-html experiments\svg_assembly\reports\tree_planner_error_viewer_nn_svg_geometry_composite.html --output-json experiments\svg_assembly\reports\tree_planner_error_viewer_nn_svg_geometry_composite.json
python scripts\build\build_tree_planner_error_viewer.py --prediction-dir experiments\svg_assembly\tree_planner_predictions_subassembly_candidates_svg_geometry_test --output-html experiments\svg_assembly\reports\tree_planner_error_viewer_subassembly_candidates_svg_geometry.html --output-json experiments\svg_assembly\reports\tree_planner_error_viewer_subassembly_candidates_svg_geometry.json
python scripts\build\build_tree_planner_error_viewer.py --prediction-dir experiments\svg_assembly\tree_planner_predictions_subassembly_candidates_svg_geometry_neg20_test --output-html experiments\svg_assembly\reports\tree_planner_error_viewer_subassembly_candidates_svg_geometry_neg20.html --output-json experiments\svg_assembly\reports\tree_planner_error_viewer_subassembly_candidates_svg_geometry_neg20.json
```

Current test Simple/Hard F1:

```text
geometry-only: 0.396 / 0.034
SVG-only:      0.447 / 0.123
SVG+geometry:  0.412 / 0.126
geometry NN:   0.429 / 0.162
SVG NN:        0.409 / 0.082
SVG+geom NN:   0.477 / 0.156
SVG+comp NN:   0.476 / 0.163
SVG+g+comp NN: 0.488 / 0.206
comp oracle:   1.000 / 1.000
pred subasm recall-heavy: 0.287 / 0.068
pred subasm conservative: 0.406 / 0.098
pred subasm neg20:        0.395 / 0.113
```

The SVG rows are the first object-level evidence for SVG-enhanced tree
generation in the greedy setting. The neural rows use an MLP merge scorer but
still rely on connected-component merge decoding; they are a baseline, not yet
a full set/graph tree decoder. The composite rows use manual-derived
subassembly SVG prototypes as optional manual/RAG context, so do not compare
them as pure observed-part inference. The composite oracle confirms the leakage
boundary: manual composite part sets can directly reconstruct the target tree.
The predicted-subassembly rows are no-leakage replacement attempts. Negative
sampling and conservative decoder selection reduce false positives and improve
Hard F1, but candidate precision is still too low to replace manual/RAG
composite context.

## Grounding Models and Experiment Summary

Install the local ML environment:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-ml.txt
```

Run the current validation-selected tiny CNN grounding baseline:

```powershell
python scripts\train\train_grounding_cnn.py --epochs 25 --output experiments\svg_assembly\reports\grounding_cnn_all_val_report.json --model-output experiments\svg_assembly\reports\grounding_cnn_all_val_model.pt
python scripts\train\train_grounding_cnn.py --augment --epochs 25 --output experiments\svg_assembly\reports\grounding_cnn_all_val_aug_report.json --model-output experiments\svg_assembly\reports\grounding_cnn_all_val_aug_model.pt
python scripts\train\train_grounding_cnn.py --equivalence-labels --epochs 25 --output experiments\svg_assembly\reports\grounding_cnn_all_equiv_val_report.json --model-output experiments\svg_assembly\reports\grounding_cnn_all_equiv_val_model.pt
python scripts\train\train_grounding_cnn.py --equivalence-labels --epochs 20 --max-images 8 --dropout 0.2 --output experiments\svg_assembly\reports\grounding_cnn_improved_val_report.json --model-output experiments\svg_assembly\reports\grounding_cnn_improved_val_model.pt
```

Collect the current reports into one table:

```powershell
python scripts\export\summarize_svg_assembly_experiments.py
```

Run the current grounding-to-connection diagnostic:

```powershell
python scripts\eval\run_end_to_end_diagnostic.py
python scripts\eval\run_end_to_end_diagnostic.py --split all --output experiments\svg_assembly\reports\end_to_end_diagnostic_all_report.json
python scripts\eval\run_end_to_end_diagnostic.py --grounding-model experiments\svg_assembly\reports\grounding_cnn_all_equiv_val_model.pt --split test --output experiments\svg_assembly\reports\end_to_end_diagnostic_equiv_test_report.json
python scripts\eval\run_end_to_end_diagnostic.py --grounding-model experiments\svg_assembly\reports\grounding_cnn_all_equiv_val_model.pt --split all --output experiments\svg_assembly\reports\end_to_end_diagnostic_equiv_all_report.json
python scripts\eval\run_end_to_end_diagnostic.py --grounding-model experiments\svg_assembly\reports\grounding_cnn_improved_val_model.pt --split test --output experiments\svg_assembly\reports\end_to_end_diagnostic_improved_report.json
python scripts\eval\evaluate_paper_tree_metrics.py
python scripts\export\export_tree_predictions_and_equivalence_report.py
python scripts\export\export_tree_predictions_and_equivalence_report.py --diagnostic-report experiments\svg_assembly\reports\end_to_end_diagnostic_improved_report.json --output-report experiments\svg_assembly\reports\equivalence_test_improved_report.json --output-dir experiments\svg_assembly\predicted_assembly_trees_improved_test
python scripts\export\export_tree_predictions_and_equivalence_report.py --diagnostic-report experiments\svg_assembly\reports\end_to_end_diagnostic_equiv_all_report.json --output-report experiments\svg_assembly\reports\equivalence_and_tree_export_equiv_model_report.json --output-dir experiments\svg_assembly\predicted_assembly_trees_equiv
```

The current takeaway is that simplified SVG geometry is strong enough for
connection prediction once visual instances are grounded, but grounding rendered
or real part images to SVG instances is still the limiting problem.
Equivalence-aware labels are useful because many repeated parts are
geometrically interchangeable. The current improved residual CNN reaches strict
test grounding accuracy 0.706, equivalence-aware test grounding accuracy 0.849,
strict test connection F1 0.8875, and equivalence-aware test connection F1
0.9375. The next modeling focus should be a stronger tree decoder, a more
precise subassembly candidate generator, and a real-image or 3D-observation
grounding benchmark, not only stronger 2D image jitter.

The paper's Simple/Hard assembly-tree metrics are implemented in
`evaluate_paper_tree_metrics.py`. They are useful for checking tree-node
recovery, but they do not penalize many within-step connection-edge errors.
`export_tree_predictions_and_equivalence_report.py` writes per-object predicted
tree JSON files and reports both strict and geometric-equivalence-aware
grounding/connection metrics.
