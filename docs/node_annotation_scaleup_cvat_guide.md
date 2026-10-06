# Node annotation scale-up CVAT guide

Create a **new** CVAT task named `Pea_Node_Annotation_Scaleup_v1` using only the 136 files in `data/node_annotation_scaleup/images/`. Do not upload or re-annotate the validated 16-image Pilot V1.

Create label `node` with shape **Point**. One CVAT object represents one biological main-stem node.

Attributes:

- `node_index`: integer biological node identity.
- `visibility`: `visible`, `partially_occluded`, or `uncertain`.

Annotate only the target plant's main stem, numbering from base to apex. Exclude the undeveloped apex and neighbouring plants. Do not place guessed coordinates for completely invisible nodes. Keep a reliable biological node identity across views; gaps are allowed (for example, N8 can remain N8 when N7 is fully invisible in that image).

After scale-up annotation and QC, the future workflow will merge the Pilot V1 CVAT export (16 images) with the scale-up export (136 images) programmatically. Do not merge exports or create empty/fake annotations now.
