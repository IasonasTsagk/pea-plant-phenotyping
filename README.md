# Pea Plant Phenotyping with Computer Vision

Research code and compact reproducibility artefacts for an MSc Data Science thesis on computer-vision-based monitoring of greenhouse pea-plant growth.

## Overview

The current frozen experiment addresses **main-stem pea-node localisation** as point/keypoint heatmap prediction. Earlier notebooks document image quality assessment, preprocessing, classical vegetation segmentation, and SAM2 promptable-segmentation pilots. They are exploratory steps rather than validated phenotyping measurements.

## Research motivation

Reliable node localisation can support image-based monitoring of plant development in greenhouse scenes. The setting is challenging: leaves overlap, stems are thin, support structures are visible, lighting varies, and neighbouring plants may enter the frame. The goal is therefore to localise manually defined main-stem nodes, not to claim full plant segmentation or biological trait accuracy from colour alone.

## Dataset

Dataset V1 contains 152 manually annotated images, 963 main-stem node annotations, 19 living plants, and four acquisition dates. Raw photographs and copied annotation images are deliberately excluded from this public repository. See `data/README.md` for the expected local layout.

The fixed plant-level split is:

| Partition | Plants | Images | Use |
| --- | ---: | ---: | --- |
| Training | 13 | 104 | Optimisation |
| Validation | 3 | 24 | Model development and threshold selection |
| Held-out test | 3 | 24 | One-time final generalisation evaluation |

## Node annotation

Each annotation represents a main-stem pea node as a point. Annotation design, quality control, and the fixed plant-level split are documented in `docs/`. Compact annotation and split summaries are retained under `results/`; source photographs are not included.

## Node-detection pipeline

Images are resized to 768×576 and annotated points are converted into Gaussian heatmap targets at 192×144. Local maxima are extracted with 3×3 NMS. Predicted and annotated points are paired with corrected one-to-one Hungarian matching, and detection is reported at 0.5%, 1.0%, and 2.0% of the original-image diagonal.

## Model architecture

The baseline uses a pretrained ResNet-18 with a simple heatmap decoder. The final improved model uses a pretrained ResNet-18 with an FPN-style multi-scale decoder. Both use Gaussian node targets and a numerically stable, logits-based modified CenterNet focal loss.

## Experimental setup

The frozen V3 FPN experiment uses Dataset V1, the fixed plant-level split, 768×576 inputs, 192×144 heatmaps, Gaussian sigma 2.0 heatmap pixels, AdamW with differential learning rates, and validation-only threshold selection. The selected V3 confidence threshold is **0.40**. Test data must never be used for threshold selection or other model-development decisions.

## Current results

The documented V3 validation result at 1% of image diagonal is approximately F1 0.893, precision 0.834, and recall 0.961. The one-time held-out test result on unseen plants P01, P14, and P16 is approximately F1 0.766, precision 0.673, and recall 0.889 at the same tolerance.

The principal remaining error mode is false-positive detection, particularly in cluttered greenhouse scenes and where neighbouring plants are visible. These values are localisation results for Dataset V1, not a claim of broader biological generalisation.

## Repository structure

```text
src/         Reusable data, heatmap, metric, and model modules
scripts/     Reproducible preparation, training, and evaluation entry points
notebooks/   Exploratory and methodological notebooks
docs/        Annotation, split, and project documentation
results/     Compact CSV/JSON/Markdown results and selected final figures
data/        Local-only raw and annotation-image layout (not versioned)
```

## Installation

Python 3.11 is recommended. Create an isolated environment, install the minimal project dependencies, and work from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For the optional SAM2 pilot, follow the official [SAM2 installation instructions](https://github.com/facebookresearch/sam2) separately. Its source checkout and checkpoint are intentionally not committed here.

## Running training

Training requires the local Dataset V1 images described in `data/README.md`. It does not evaluate the held-out test set.

```bash
python scripts/run_resnet18_node_baseline.py \
  --model-variant fpn \
  --output-dir results/node_detection_resnet18_v3_fpn
```

## Validation evaluation

Validation is used for model development and threshold selection only.

```bash
python scripts/evaluate_resnet18_node_validation.py \
  --model-variant fpn \
  --run-dir results/node_detection_resnet18_v3_fpn
```

## Held-out test evaluation

The test evaluator applies the frozen V3 best checkpoint and fixed threshold 0.40 to P01, P14, and P16. Do not use its output to tune thresholds or other model settings.

```bash
python scripts/evaluate_resnet18_node_test.py
```

## Future work

Priority next steps include error analysis for false positives, larger and more varied annotated datasets, independent greenhouse validation, and carefully designed temporal analyses. Additional segmentation or foundation-model work should be evaluated against suitable manual reference annotations.

## Thesis status

Dataset V1 construction, annotation QC, fixed plant-level splitting, ResNet-18 baseline development, FPN improvement, validation selection, and a final held-out test evaluation are complete. The repository preserves code and compact results needed to inspect those stages, while excluding raw images, checkpoints, virtual environments, and bulky generated visualisations.
