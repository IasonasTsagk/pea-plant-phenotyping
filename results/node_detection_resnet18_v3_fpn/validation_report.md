# Validation report

Model variant: `fpn`

Checkpoint: `results/node_detection_resnet18_v3_fpn/checkpoints/best_model.pt` (best checkpoint only)

Selected validation threshold: **0.40** (maximum F1 @ 1% diagonal; higher threshold wins ties).

## Metrics

|   tolerance_fraction_diagonal |   threshold |   TP |   FP |   FN |   precision |   recall |       f1 |   mean_localization_error_pixels |   median_localization_error_pixels |   node_count_mae |   median_absolute_count_error |   exact_node_count_accuracy |
|------------------------------:|------------:|-----:|-----:|-----:|------------:|---------:|---------:|---------------------------------:|-----------------------------------:|-----------------:|------------------------------:|----------------------------:|
|                         0.005 |         0.4 |  140 |   35 |   12 |    0.8      | 0.921053 | 0.856269 |                          14.1864 |                            14.0292 |            1.125 |                             0 |                    0.583333 |
|                         0.01  |         0.4 |  146 |   29 |    6 |    0.834286 | 0.960526 | 0.892966 |                          15.2183 |                            14.917  |            1.125 |                             0 |                    0.583333 |
|                         0.02  |         0.4 |  147 |   28 |    5 |    0.84     | 0.967105 | 0.899083 |                          15.6596 |                            14.9375 |            1.125 |                             0 |                    0.583333 |

## Scope

Only P04, P07 and P17 (24 images; 152 GT nodes) were evaluated. P01, P14 and P16 were explicitly excluded. TEST SET WAS NOT USED OR EVALUATED.
