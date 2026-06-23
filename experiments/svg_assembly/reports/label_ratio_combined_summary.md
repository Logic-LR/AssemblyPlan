# Label-Ratio Tree Planner Summary

| label_ratio | method | feature_mode | labeled_fit_objects | gt_reward_objects | val_hard_f1 | test_simple_f1 | test_hard_f1 | all_hard_f1 | train_entropy | train_confident_0_95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1000 | BCE context MLP entropy=0 | svg_geometry_composite | 6 | - | 0.3789 | 0.4661 | 0.1819 | 0.2700 | 0.0154 | 1.0000 |
| 0.1000 | BCE context MLP entropy=0 | svg_geometry | 6 | - | 0.1273 | 0.3382 | 0.0601 | 0.1635 | 0.0562 | 0.9283 |
| 0.1000 | GRPO gt=0.50 svg=0.20 spatial=0.30 | svg_geometry_composite | - | 6 | 0.4164 | 0.4868 | 0.2373 | 0.3330 | - | - |
| 0.2500 | BCE context MLP entropy=0 | svg_geometry_composite | 16 | - | 0.3443 | 0.5353 | 0.2606 | 0.3862 | 0.0704 | 0.9354 |
| 0.2500 | BCE context MLP entropy=0 | svg_geometry | 16 | - | 0.2052 | 0.3655 | 0.0774 | 0.1641 | 0.3221 | 0.3802 |
| 0.2500 | GRPO gt=0.50 svg=0.20 spatial=0.30 | svg_geometry_composite | - | 16 | 0.3850 | 0.5522 | 0.2736 | 0.3765 | - | - |
| 0.5000 | BCE context MLP entropy=0 | svg_geometry_composite | 31 | - | 0.3723 | 0.4953 | 0.1988 | 0.4422 | 0.0379 | 0.9592 |
| 0.5000 | BCE context MLP entropy=0 | svg_geometry | 31 | - | 0.2186 | 0.4550 | 0.1242 | 0.3333 | 0.0698 | 0.8953 |
| 0.5000 | GRPO gt=0.50 svg=0.20 spatial=0.30 | svg_geometry_composite | - | 31 | 0.3723 | 0.4907 | 0.1954 | 0.4100 | - | - |
| 1.0000 | BCE context MLP entropy=0 | svg_geometry_composite | 62 | - | 0.4422 | 0.5704 | 0.3316 | 0.6729 | 0.1024 | 0.8414 |
| 1.0000 | BCE context MLP entropy=0 | svg_geometry | 62 | - | 0.2608 | 0.4071 | 0.1081 | 0.4232 | 0.1070 | 0.8181 |
| 1.0000 | GRPO gt=0.50 svg=0.20 spatial=0.30 | svg_geometry_composite | - | 62 | 0.4589 | 0.5157 | 0.2102 | 0.5463 | - | - |
