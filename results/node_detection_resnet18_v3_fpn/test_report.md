# Held-out test report

Frozen checkpoint: `results/node_detection_resnet18_v3_fpn/checkpoints/best_model.pt` (best epoch 7).

Frozen confidence threshold: **0.40**, selected on validation only. No test threshold sweep or tuning was performed.

## Test metrics

|   tolerance_fraction_diagonal |   threshold |   TP |   FP |   FN |   precision |   recall |       f1 |   mean_localization_error_pixels |   median_localization_error_pixels |   node_count_mae |   median_absolute_count_error |   exact_node_count_accuracy |
|------------------------------:|------------:|-----:|-----:|-----:|------------:|---------:|---------:|---------------------------------:|-----------------------------------:|-----------------:|------------------------------:|----------------------------:|
|                         0.005 |         0.4 |  121 |   81 |   32 |    0.59901  | 0.79085  | 0.68169  |                          15.1745 |                            13.6774 |            2.375 |                             1 |                    0.291667 |
|                         0.01  |         0.4 |  136 |   66 |   17 |    0.673267 | 0.888889 | 0.766197 |                          17.7049 |                            15.2673 |            2.375 |                             1 |                    0.291667 |
|                         0.02  |         0.4 |  138 |   64 |   15 |    0.683168 | 0.901961 | 0.777465 |                          18.8171 |                            15.3369 |            2.375 |                             1 |                    0.291667 |

Primary metric: F1 @ 1.0% of original-image diagonal.
