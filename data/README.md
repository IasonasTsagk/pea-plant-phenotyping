# Local data layout

Raw photographs, copied annotation images, and other source data are excluded from version control. This protects the original research data and keeps the public repository lightweight.

To reproduce image-dependent steps locally, place the authorised Dataset V1 image copies in:

```text
data/node_annotation_pilot/images/
data/node_annotation_scaleup/images/
```

Do not edit, rename, overwrite, or commit the raw photographs. The compact split and point-coordinate tables retained in `results/` identify the expected filenames for authorised local data.
