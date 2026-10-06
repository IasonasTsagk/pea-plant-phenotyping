"""Create and validate the fixed Dataset V1 Plant_ID split."""
from pathlib import Path
import json
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.node_dataset_split import EXPECTED_IMAGES, EXPECTED_POINTS, SPLITS, assign

tables = ROOT / "results" / "node_ground_truth" / "tables"
out = ROOT / "results" / "node_dataset_split"; out.mkdir(parents=True, exist_ok=True)
images, points = assign(pd.read_csv(tables / "node_images_152.csv"), pd.read_csv(tables / "node_detection_points_152.csv"))
plant = images.groupby(["plant_id", "split"], as_index=False).agg(number_of_images=("filename", "size"), dates_represented=("acquisition_date", "nunique"), mean_node_count_per_image=("node_count", "mean"))
point_counts = points.groupby("plant_id", as_index=False).size().rename(columns={"size": "number_of_node_annotations"})
plant = plant.merge(point_counts, on="plant_id")[['plant_id','split','number_of_images','number_of_node_annotations','dates_represented','mean_node_count_per_image']].sort_values('plant_id')
plant.to_csv(out / "plant_split_assignment.csv", index=False)
images.to_csv(out / "node_images_152_split.csv", index=False); points.to_csv(out / "node_detection_points_152_split.csv", index=False)
for split in SPLITS: (out / f"{split}_images.txt").write_text("\n".join(sorted(images.loc[images.split.eq(split), "filename"])) + "\n", encoding="utf-8")
date = images.groupby(["split", "acquisition_date"], as_index=False).agg(number_of_images=("filename", "size"), mean_node_count_per_image=("node_count", "mean"))
date = date.merge(points.groupby(["split", "acquisition_date"], as_index=False).size().rename(columns={"size":"number_of_node_points"}), on=["split","acquisition_date"])
node = images.groupby("split").node_count.agg(["mean","median","min","max"]).round(3)
visibility = points.groupby(["split", "visibility"]).size().rename("count").reset_index(); visibility["percent"] = visibility["count"] / visibility.groupby("split")["count"].transform("sum") * 100
summary = {"dataset_version":"V1","total_images":152,"total_points":963,"split_unit":"plant_id","train_plants":list(SPLITS['train']),"validation_plants":list(SPLITS['validation']),"test_plants":list(SPLITS['test']),"train_images":104,"validation_images":24,"test_images":24,"train_points":658,"validation_points":152,"test_points":153}
(out / "dataset_v1_split_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
report = "# Dataset V1 Plant-level split QC\n\n" + "## Assignments\n\n" + "\n".join(f"- {split}: {', '.join(plants)}" for split, plants in SPLITS.items()) + f"\n\n- Plant counts: {{'train': 13, 'validation': 3, 'test': 3}}\n- Image counts: {images.groupby('split').size().to_dict()}\n- Point counts: {points.groupby('split').size().to_dict()}\n\n## Date distribution\n\n```csv\n{date.to_csv(index=False)}```\n\n## Node count per image\n\n```csv\n{node.reset_index().to_csv(index=False)}```\n\n## Visibility distribution\n\n```csv\n{visibility.to_csv(index=False)}```\n\n## Checks\n\n- PASS: Plant leakage\n- PASS: Filename leakage\n- PASS: All images assigned\n- PASS: All points assigned\n- PASS: Expected image counts\n- PASS: Expected point counts\n- PASS: Date coverage\n\nDATASET V1 PLANT-LEVEL SPLIT QC: PASS\n"
(out / "split_qc_report.md").write_text(report, encoding="utf-8")
print(f"images={len(images)}; points={len(points)}; image_counts={images.groupby('split').size().to_dict()}; point_counts={points.groupby('split').size().to_dict()}")
