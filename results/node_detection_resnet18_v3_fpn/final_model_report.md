# Final frozen V3 FPN model report

## Frozen experiment

- Architecture: ResNet-18 + FPN-style multi-scale decoder
- Checkpoint: `checkpoints/best_model.pt` (best epoch 7)
- Input / heatmap: 768×576 / 192×144
- Post-processing: 3×3 local-max NMS and corrected Hungarian one-to-one matching
- Threshold: 0.40, selected on validation only

## Partition separation

**VALIDATION** was used for model development and threshold selection.

**TEST** was used once for final held-out generalization evaluation; it was not used for training or tuning.

| partition   | purpose                                           |   images |   gt_nodes |   threshold |   primary_f1_at_1pct_diagonal |   precision_at_1pct_diagonal |   recall_at_1pct_diagonal |
|:------------|:--------------------------------------------------|---------:|-----------:|------------:|------------------------------:|-----------------------------:|--------------------------:|
| validation  | model development and threshold selection         |       24 |        152 |         0.4 |                      0.892966 |                     0.834286 |                  0.960526 |
| test        | one-time final held-out generalization evaluation |       24 |        153 |         0.4 |                      0.766197 |                     0.673267 |                  0.888889 |

## Test metrics by tolerance

|   tolerance_fraction_diagonal |   threshold |   TP |   FP |   FN |   precision |   recall |       f1 |   mean_localization_error_pixels |   median_localization_error_pixels |   node_count_mae |   median_absolute_count_error |   exact_node_count_accuracy |
|------------------------------:|------------:|-----:|-----:|-----:|------------:|---------:|---------:|---------------------------------:|-----------------------------------:|-----------------:|------------------------------:|----------------------------:|
|                         0.005 |         0.4 |  121 |   81 |   32 |    0.59901  | 0.79085  | 0.68169  |                          15.1745 |                            13.6774 |            2.375 |                             1 |                    0.291667 |
|                         0.01  |         0.4 |  136 |   66 |   17 |    0.673267 | 0.888889 | 0.766197 |                          17.7049 |                            15.2673 |            2.375 |                             1 |                    0.291667 |
|                         0.02  |         0.4 |  138 |   64 |   15 |    0.683168 | 0.901961 | 0.777465 |                          18.8171 |                            15.3369 |            2.375 |                             1 |                    0.291667 |
