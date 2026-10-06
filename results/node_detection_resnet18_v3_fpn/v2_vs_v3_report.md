# V2 versus V3 validation comparison

This is a validation-only controlled comparison: Dataset V1, plant split, input and heatmap resolution, augmentation, optimiser, learning rates, loss, heatmap prior and early stopping were held fixed. It tests whether V3 multi-scale FPN fusion improves localisation over the V2 coarse decoder. Do not claim improvement until the table below is reviewed.

| model   |   best_epoch |   best_validation_loss |   selected_validation_threshold |   precision_@0.5% |   recall_@0.5% |   f1_@0.5% |   precision_@1.0% |   recall_@1.0% |   f1_@1.0% |   precision_@2.0% |   recall_@2.0% |   f1_@2.0% |   node_count_mae |   exact_node_count_accuracy |
|:--------|-------------:|-----------------------:|--------------------------------:|------------------:|---------------:|-----------:|------------------:|---------------:|-----------:|------------------:|---------------:|-----------:|-----------------:|----------------------------:|
| V2      |           14 |               1.34702  |                             0.3 |             0.125 |      0.0592105 |  0.0803571 |          0.430556 |       0.203947 |   0.276786 |          0.847222 |       0.401316 |   0.544643 |          3.33333 |                    0        |
| V3 FPN  |            7 |               0.315837 |                             0.4 |             0.8   |      0.921053  |  0.856269  |          0.834286 |       0.960526 |   0.892966 |          0.84     |       0.967105 |   0.899083 |          1.125   |                    0.583333 |

TEST SET WAS NOT USED OR EVALUATED.
