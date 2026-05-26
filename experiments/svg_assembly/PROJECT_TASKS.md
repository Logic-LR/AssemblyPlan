# SVG-Enhanced Assembly Tree Generation Tasks

Last updated: 2026-05-26

## Task Definition

The project goal is to use instruction-manual SVGs and assembly-step supervision
to improve assembly-tree generation.

The target inference setting is:

```text
observed real parts
-> recognize/ground each part to a simplified SVG or canonical part token
-> feed all part SVG/tokens to a trained planner
-> generate an assembly tree
```

Fine-grained physical manipulation details, such as screw angle, contact pose,
force, or exact robot trajectory, are outside this module and should be handled
by a later low-level assembly/action model.

Manuals are used in two ways:

1. Training supervision: manual steps provide part groupings, subassemblies,
   connection hints, order, and ground-truth assembly trees.
2. Optional inference support: if a relevant manual can be retrieved, it can
   guide planning; if not, the planner should still generate a plausible tree
   from observed part tokens and learned assembly patterns.

## Current Research Claim

Manual-derived SVG representations are useful intermediate supervision for
assembly planning. The first object-level tree-planner baseline already shows
that SVG prototypes improve tree generation over geometry-only features without
reading per-step SVGs at inference.

Current test Simple/Hard F1:

```text
geometry-only greedy planner: 0.3962 / 0.0345
SVG-only greedy planner:      0.4475 / 0.1229
SVG+geometry greedy planner:  0.4117 / 0.1259
geometry-only neural scorer:  0.4286 / 0.1622
SVG-only neural scorer:       0.4094 / 0.0818
SVG+geometry neural scorer:   0.4773 / 0.1557
SVG+composite neural scorer:  0.4758 / 0.1631
SVG+geom+composite neural:    0.4884 / 0.2057
manual composite oracle:      1.0000 / 1.0000
pred. subassembly recall-heavy: 0.2869 / 0.0675
pred. subassembly conservative: 0.4063 / 0.0976
pred. subassembly neg20:        0.3952 / 0.1128
```

These numbers are not yet the ceiling. The current neural rows use an MLP
merge scorer, but the decoder is still a simple connected-component merge
procedure rather than a full learned set/graph tree decoder. The composite rows
use manual-derived subassembly SVG prototypes as optional manual/RAG context,
so they should be reported separately from pure observed-part inference. The
manual composite oracle demonstrates that those tokens can directly encode the
assembly tree and should not be used as ordinary inference input. The greedy
rows support the SVG-enhancement claim; the neural rows show that feature/decoder
design still needs careful ablation. The predicted subassembly candidate row is
the first no-leakage attempt to replace manual composite tokens at inference.
Negative sampling and conservative decoder selection reduce false positives and
improve Hard F1, but candidate precision is still too low for this module to
replace manual/RAG composite context.

## Completed

- [x] Parse manual SVGs into structured vector features.
- [x] Simplify SVG instances into compact geometry tokens.
- [x] Recover supervised SVG-instance to part alignment from released masks.
- [x] Render primitive OBJ parts into synthetic multi-view part images.
- [x] Build part-to-SVG grounding samples.
- [x] Train and evaluate simplified-SVG connection classifier.
- [x] Train and evaluate grounding CNN baselines.
- [x] Remove invalid cross-step color tracking from the diagnostic path.
- [x] Build object-level tree-generation dataset.
- [x] Train initial geometry-only, SVG-only, and SVG+geometry greedy tree-planner baselines.
- [x] Train first geometry-only, SVG-only, and SVG+geometry neural merge-scorer baselines.
- [x] Add explicit subassembly/composite SVG prototypes to the tree-generation dataset.
- [x] Train first composite-context planner baselines and report them separately from pure observed-part inference.
- [x] Add a composite-context oracle decoder to quantify manual-step leakage/upper bound.
- [x] Train first no-leakage subassembly candidate predictor from primitive part tokens.
- [x] Add negative sampling and precision-oriented decoder selection to the no-leakage subassembly candidate predictor.
- [x] Export object-level planner predictions for test objects.
- [x] Add a realistic planner inference interface:
  `observed_part_tokens.json -> predicted_assembly_tree.json`.
- [x] Build an error viewer for planner failures, showing predicted tree, GT tree, part tokens, and SVG prototypes.
- [x] Add experiment summaries and README documentation for the current pipeline.

## Active Todo

- [ ] Replace the connected-component merge decoder with a stronger set/graph tree decoder.
- [ ] Further improve no-leakage subassembly/composite prediction; candidate precision is still too low.
- [ ] Remove the diagnostic dependency on known `k = len(gt_connections)` by learning or thresholding connection count.
- [ ] Add a real-image or synthetic-domain-gap grounding benchmark for the robot setting.
- [ ] Add RAG/manual-retrieval hooks as optional planner context.

## Next Immediate Step

Replace the current connected-component merge decoder with a stronger learned
tree decoder, and make the no-leakage subassembly candidate module less noisy.
The next decoder should keep the same paper Simple/Hard metrics so it can be
compared against:

```text
SingleStep
GeoCluster-style geometry-only baselines
current greedy geometry/SVG planners
current neural merge-scorer planners
```

## Rules For This Task List

- Every completed implementation or experiment should check off the matching
  item in this file.
- If a task changes direction, update the wording here instead of relying on
  chat history.
- Keep the manual-step diagnostic and object-level tree generation clearly
  separated: the former uses per-step SVGs, while the latter predicts a tree
  from object-level primitive part tokens.
