# Node-detection dataset design

Dataset V1 contains September 2026 data: 152 manually annotated images from 19 living plants across four acquisition dates. It is the fixed dataset for the first node-localization baseline.

For the initial detector, every manual point has one semantic class: `node`. `node_index` is biological metadata, not a training class. Detection will first estimate node locations; ordering and longitudinal identity are separate later tasks.

Images acquired in October, November, and December must not automatically enter model training. They will support temporal/generalization testing, model-assisted annotation with human correction, and controlled expansion to Dataset V2/V3. This preserves an evaluation of whether a model trained on earlier growth stages generalizes to later ones.
