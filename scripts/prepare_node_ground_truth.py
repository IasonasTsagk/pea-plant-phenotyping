"""Prepare verified Phase 6C ground-truth tables from read-only CVAT exports."""
from pathlib import Path
import argparse
import sys
from datetime import datetime

import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.node_ground_truth import (POINT_COLUMNS, coerce, image_manifest, parse_export,
                                   structural_errors, temporal_summary)

parser = argparse.ArgumentParser()
parser.add_argument("--pilot-zip", type=Path)
parser.add_argument("--scaleup-zip", type=Path)
parser.add_argument("--skip-overlays", action="store_true")
args = parser.parse_args()
raw = ROOT / "data" / "node_annotations" / "raw"
archives = sorted(raw.glob("*.zip"))
pilot_zip = args.pilot_zip or next((path for path in archives if "2630794" in path.name), None)
scaleup_zip = args.scaleup_zip or next((path for path in archives if "2630948" in path.name), None)
if not pilot_zip or not scaleup_zip: raise FileNotFoundError("Supply --pilot-zip and --scaleup-zip; CVAT sources were not renamed.")
pilot_points, pilot_images = parse_export(pilot_zip, "pilot_v1")
scaleup_points, scaleup_images = parse_export(scaleup_zip, "scaleup_v1")
points, images = coerce(pd.concat([pilot_points, scaleup_points], ignore_index=True), pd.concat([pilot_images, scaleup_images], ignore_index=True))
selection = pd.read_csv(ROOT / "results" / "node_annotation_scaleup" / "tables" / "node_annotation_selection_152.csv")
errors = structural_errors(points, images, set(selection.filename))
output = ROOT / "results" / "node_ground_truth"; tables = output / "tables"; tables.mkdir(parents=True, exist_ok=True)
qc = "# Final ground-truth structural QC\n\n" + f"- Pilot images/points: {len(pilot_images)} / {len(pilot_points)}\n- Scale-up images/points: {len(scaleup_images)} / {len(scaleup_points)}\n- Combined images/points: {len(images)} / {len(points)}\n\n" + "## Checks\n\n" + "\n".join(f"- {'FAIL' if name in errors else 'PASS'}: {name}" for name in ["expected image count", "expected point count", "duplicate filenames", "duplicate node indices", "coordinate bounds", "attribute validity", "Plant_ID validity", "date validity"]) + "\n\n" + ("## Errors\n\n" + "\n".join(f"- {item}" for item in errors) if errors else "## Result\n\nFINAL GROUND TRUTH STRUCTURAL QC: PASS") + "\n"
(output / "final_ground_truth_qc.md").write_text(qc, encoding="utf-8")
if errors: raise SystemExit("Structural QC failed; no derived ground-truth tables were written.")
points[POINT_COLUMNS].to_csv(tables / "node_points_152.csv", index=False)
image_rows = image_manifest(points, images); image_rows.to_csv(tables / "node_images_152.csv", index=False)
temporal_summary(image_rows).to_csv(tables / "node_temporal_summary.csv", index=False)
points[["filename", "plant_id", "acquisition_date", "x", "y", "x_normalized", "y_normalized", "visibility"]].to_csv(tables / "node_detection_points_152.csv", index=False)
plant = image_rows.groupby("plant_id", as_index=False).agg(number_of_images=("filename", "size"), number_of_node_annotations=("node_count", "sum"), dates_represented=("acquisition_date", "nunique"), mean_node_count_per_image=("node_count", "mean")); plant.to_csv(tables / "plant_dataset_summary.csv", index=False)
stats = "# Node ground-truth dataset statistics\n\n" + f"- Images: {len(image_rows)}\n- Node points: {len(points)}\n- Plants: {image_rows.plant_id.nunique()}\n- Acquisition dates: {sorted(image_rows.acquisition_date.unique())}\n- Node count per image (min/max/mean/median): {image_rows.node_count.min()} / {image_rows.node_count.max()} / {image_rows.node_count.mean():.2f} / {image_rows.node_count.median():.2f}\n- Visibility distribution: {points.visibility.value_counts().to_dict()}\n- Points per date: {points.groupby('acquisition_date').size().to_dict()}\n- Images per date: {image_rows.groupby('acquisition_date').size().to_dict()}\n- Images per plant: {image_rows.groupby('plant_id').size().to_dict()}\n- Pilot versus scale-up images: {image_rows.source_dataset.value_counts().to_dict()}\n- Pilot versus scale-up points: {points.source_dataset.value_counts().to_dict()}\n- Node-count distributions over time: {image_rows.groupby('acquisition_date').node_count.describe()[['min','max','mean','50%']].round(2).to_dict()}\n"
(output / "dataset_statistics.md").write_text(stats, encoding="utf-8")
if args.skip_overlays:
    print(f"pilot_images={len(pilot_images)} scaleup_images={len(scaleup_images)} images={len(images)} points={len(points)} overlays=skipped")
    raise SystemExit(0)

# Deterministic 24-image QC sample: low/high count within date/source, then visibility coverage.
source_paths = selection.set_index("filename")["source_path"].to_dict()
# The approved 136 scale-up CVAT-upload copies are local and byte-checked in
# Phase 6B. Prefer them for visual QC; Pilot V1 remains read-only at its source.
local_scaleup = ROOT / "data" / "node_annotation_scaleup" / "images"
for filename in selection.loc[selection.annotation_status.eq("pending"), "filename"]:
    if (local_scaleup / filename).is_file(): source_paths[filename] = str(local_scaleup / filename)
ranked = image_rows.sort_values(["acquisition_date", "source_dataset", "node_count", "filename"])
sample = []
for _, group in ranked.groupby(["acquisition_date", "source_dataset"]): sample += [group.iloc[0].filename, group.iloc[-1].filename]
for visibility in ("partially_occluded", "uncertain", "visible"):
    for name in ranked.loc[ranked[f"{visibility}_count"].gt(0), "filename"]:
        if name not in sample and len(sample) < 24: sample.append(name)
for name in ranked.filename:
    if name not in sample and len(sample) < 24: sample.append(name)
overlays = output / "qc_overlays"; overlays.mkdir(parents=True, exist_ok=True)
thumbs = []
colors = {"visible": "lime", "partially_occluded": "orange", "uncertain": "red"}
for filename in sample + [name for name in ranked.filename if name not in sample]:
    if len(thumbs) >= 24: break
    try:
      with Image.open(source_paths[filename]) as image:
        original_width, original_height = image.size
        image.draft("RGB", (1200, 900))
        image = image.convert("RGB"); draw = ImageDraw.Draw(image)
        scale_x, scale_y = image.width / original_width, image.height / original_height
        for point in points.loc[points.filename.eq(filename)].itertuples(index=False):
            x, y = float(point.x) * scale_x, float(point.y) * scale_y; draw.ellipse((x-8,y-8,x+8,y+8), outline=colors[point.visibility], width=3); draw.text((x+10,y-8), f"N{point.node_index} {point.visibility}", fill=colors[point.visibility], stroke_width=1, stroke_fill="black")
        draw.rectangle((0,0,image.width,50), fill="black"); draw.text((10,15), filename, fill="white")
        image.save(overlays / f"{Path(filename).stem}_overlay.jpg", quality=92)
        image.thumbnail((420, 320)); thumbs.append((filename, image.copy()))
    except (OSError, TimeoutError) as error:
        print(f"overlay_skipped={filename}; reason={type(error).__name__}")
sheet = Image.new("RGB", (4*420, ((len(thumbs)+3)//4)*350), "white"); draw = ImageDraw.Draw(sheet)
for i, (filename, thumbnail) in enumerate(thumbs):
    x, y = (i%4)*420, (i//4)*350; sheet.paste(thumbnail, (x+(420-thumbnail.width)//2,y)); draw.text((x+5,y+322),filename,fill="black")
sheet.save(output / "qc_contact_sheet.jpg", quality=92)
print(f"pilot_images={len(pilot_images)} scaleup_images={len(scaleup_images)} images={len(images)} points={len(points)} overlays={len(sample)}")
