"""Run the read-only Phase 5 node-dataset audit, optionally in resumable batches."""
from pathlib import Path
import argparse
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.node_dataset_audit import (build_manifest, configured_data_root, discover_images,
                                    discovery_diagnostics, finalize_manifest,
                                    find_unexpected_extensions, save_contact_sheets,
                                    summarise, write_report)

parser = argparse.ArgumentParser()
parser.add_argument("--data-root", required=True)
parser.add_argument("--batch-index", type=int)
parser.add_argument("--batch-count", type=int)
parser.add_argument("--finalize", action="store_true")
parser.add_argument("--contact-sheet-date")
parser.add_argument("--contact-sheet-plants", help="Comma-separated plant IDs for contact-sheet batching.")
args = parser.parse_args()
data_root = configured_data_root(args.data_root)
diagnostics = discovery_diagnostics(data_root)
print("Phase 5 discovery diagnostics:")
print(f"  DATA_ROOT exists: {diagnostics['data_root_exists']}")
print(f"  DATA_ROOT is directory: {diagnostics['data_root_is_directory']}")
print(f"  raw recursive supported-image count before filtering: {diagnostics['raw_recursive_supported_image_count']}")
print(f"  supported-image count after filtering: {diagnostics['supported_image_count_after_filtering']}")
output = ROOT / "results" / "node_dataset_audit"; tables = output / "tables"; work = output / ".audit_work"

if args.contact_sheet_date:
    manifest = pd.read_csv(tables / "node_dataset_manifest.csv")
    manifest["timestamp"] = pd.to_datetime(manifest["timestamp"])
    subset = manifest.loc[manifest["acquisition_date"].eq(args.contact_sheet_date)]
    if args.contact_sheet_plants:
        plants = {plant.strip().upper() for plant in args.contact_sheet_plants.split(",") if plant.strip()}
        subset = subset.loc[subset["plant_id"].isin(plants)]
    save_contact_sheets(subset, output / "contact_sheets")
    print(f"contact_sheet_date={args.contact_sheet_date}; plants={args.contact_sheet_plants or 'all'}; images={len(subset)}")
    raise SystemExit(0)

if args.batch_index is not None or args.batch_count is not None:
    if args.batch_index is None or args.batch_count is None or args.batch_count < 1 or not 0 <= args.batch_index < args.batch_count:
        parser.error("--batch-index and --batch-count must be supplied together, with 0 <= index < count.")
    paths = discover_images(data_root)[args.batch_index::args.batch_count]
    work.mkdir(parents=True, exist_ok=True)
    batch = build_manifest(data_root, paths=paths, calculate_duplicate_hashes=False)
    batch.to_csv(work / f"manifest_batch_{args.batch_index:02d}_of_{args.batch_count:02d}.csv", index=False)
    print(f"batch={args.batch_index + 1}/{args.batch_count}; images={len(batch)}")
    raise SystemExit(0)

if args.finalize:
    parts = sorted(work.glob("manifest_batch_*_of_*.csv"))
    if not parts:
        parser.error("No audit batch files found. Run --batch-index/--batch-count first.")
    manifest = finalize_manifest(pd.concat((pd.read_csv(part) for part in parts), ignore_index=True))
else:
    manifest = build_manifest(data_root)

summary, anomalies = summarise(manifest)
source_issues = []
for record in manifest.loc[~manifest.readable].itertuples():
    source_issues.append({"acquisition_date": record.acquisition_date, "plant_id": record.plant_id, "audit_status": "unreadable_image", "source_path": record.source_path})
for record in manifest.loc[manifest.duplicate_hash].itertuples():
    source_issues.append({"acquisition_date": record.acquisition_date, "plant_id": record.plant_id, "audit_status": "duplicate_file_hash", "source_path": record.source_path, "filename": record.filename})
duplicate_names = manifest.duplicated(["acquisition_date", "plant_id", "filename"], keep=False)
for record in manifest.loc[duplicate_names].itertuples():
    source_issues.append({"acquisition_date": record.acquisition_date, "plant_id": record.plant_id, "audit_status": "duplicate_filename_within_plant_date", "source_path": record.source_path, "filename": record.filename})
for record in manifest.loc[manifest.plant_id.isna() | manifest.acquisition_date.isna()].itertuples():
    source_issues.append({"acquisition_date": record.acquisition_date, "plant_id": record.plant_id, "audit_status": "missing_or_unexpected_path_metadata", "source_path": record.source_path})
for record in find_unexpected_extensions(data_root).itertuples():
    source_issues.append({"audit_status": "unexpected_extension", "source_path": record.source_path, "filename": record.filename, "extension": record.extension})
if source_issues:
    anomalies = pd.concat([anomalies, pd.DataFrame(source_issues)], ignore_index=True, sort=False)
tables.mkdir(parents=True, exist_ok=True)
manifest.to_csv(tables / "node_dataset_manifest.csv", index=False)
summary.to_csv(tables / "dataset_summary_by_date_plant.csv", index=False)
anomalies.to_csv(tables / "acquisition_anomalies.csv", index=False)
pilot = manifest.loc[manifest.plant_id.isin(["P07", "P12"]) & manifest.acquisition_date.isin(["2026-09-15", "2026-09-18", "2026-09-25", "2026-09-29"]), ["acquisition_date", "plant_id", "view_slot", "filename", "source_path", "width", "height", "mean_brightness", "mean_saturation", "sharpness_laplacian_variance"]].copy()
pilot["selected_for_cvat"] = ""; pilot["selection_reason"] = ""
pilot.to_csv(tables / "pilot_annotation_candidates.csv", index=False)
write_report(data_root, manifest, summary, anomalies, output / "dataset_audit_report.md")
print(f"data_root={data_root}; dates={manifest.acquisition_date.nunique()}; images={len(manifest)}")
