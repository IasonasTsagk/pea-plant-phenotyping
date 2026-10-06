# Dataset V1 experimental split

Dataset V1 tests Plant_ID generalization within the September 2026 growth period. Its fixed split assigns 13 biological plants to training, three unseen plants to validation, and three unseen plants to test. Every image from a biological plant remains in one split, preventing leakage across repeated dates and views.

Use training plus validation for model development, architecture and heatmap choices, augmentation, thresholds, hyperparameters, and epoch selection. Do not use the test set for any of these choices. Evaluate the test set only after finalizing the configuration.

October–December acquisitions are reserved for a distinct temporal/growth-stage generalization experiment. Do not automatically add them to Dataset V1 training.
