# Project status: node-localization pivot

## What Phases 1–4 tested

The earlier phases characterised the pilot images, explored RGB/HSV/LAB representations, tested conservative preprocessing, established classical colour-based vegetation masks, and evaluated prompted SAM2 including whole-plant and device-sanity checks. These remain exploratory evidence about image conditions and segmentation limitations.

## Why whole-plant segmentation is no longer primary

The biological supervisor clarified that the current cultivar has one non-branching vertical main stem. The useful phenotype representation is therefore an ordered base-to-apex sequence of main-stem nodes, rather than a generic whole-plant foreground mask. Earlier masks cannot reliably isolate the target plant from nearby vegetation and greenhouse structure, especially in dense, trellised, or blurred views.

## New pipeline

Read-only dataset audit → contact-sheet review → manual main-stem node keypoint pilot in CVAT → QA and plant-level split review → pretrained ResNet backbone with node/keypoint heatmap localization baseline.

No node model is trained in Phase 5. No automatic or pseudo node annotations are produced. The stated future split (about 13/3/3 plants for train/validation/test) remains a planning assumption until annotation coverage is reviewed; images from the same plant must not be randomly split across sets.
