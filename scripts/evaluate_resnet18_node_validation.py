"""Validation-only evaluator for the ResNet-18 node-localisation experiments.

This script deliberately never creates a test dataset or dataloader. It only
accepts the locked Dataset V1 validation plants (P04, P07 and P17).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.node_detection_dataset import NodeDetectionDataset, image_lookup, node_detection_collate_fn  # noqa: E402
from src.node_detection_metrics import match, peaks  # noqa: E402
from src.train_node_detector import device  # noqa: E402

VALIDATION_PLANTS = {"P04", "P07", "P17"}
LOCKED_TEST_PLANTS = {"P01", "P14", "P16"}
THRESHOLDS = [round(value / 100, 2) for value in range(10, 91, 5)]
TOLERANCES = (0.005, 0.01, 0.02)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-variant", choices=("baseline", "fpn"), default="baseline")
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="Run directory relative to the project root or absolute.")
    return parser.parse_args()


def resolve_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir is not None:
        return args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    return ROOT / "results" / ("node_detection_resnet18_v3_fpn" if args.model_variant == "fpn" else "node_detection_resnet18_v2")


def make_model(variant: str) -> torch.nn.Module:
    if variant == "fpn":
        from src.models.resnet_node_heatmap_fpn import ResNet18NodeHeatmapFPN
        return ResNet18NodeHeatmapFPN(heatmap_prior_probability=0.10)
    from src.models.resnet_node_heatmap import ResNet18NodeHeatmap
    return ResNet18NodeHeatmap(heatmap_prior_probability=0.10)


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def metric_record(tp: int, fp: int, fn: int, errors: list[float], count_errors: list[int]) -> dict:
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    return {
        "TP": tp, "FP": fp, "FN": fn, "precision": precision, "recall": recall,
        "f1": safe_divide(2 * precision * recall, precision + recall),
        "mean_localization_error_pixels": float(np.mean(errors)) if errors else np.nan,
        "median_localization_error_pixels": float(np.median(errors)) if errors else np.nan,
        "node_count_mae": float(np.mean(count_errors)) if count_errors else np.nan,
        "median_absolute_count_error": float(np.median(count_errors)) if count_errors else np.nan,
        "exact_node_count_accuracy": float(np.mean(np.asarray(count_errors) == 0)) if count_errors else np.nan,
    }


def load_validation_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    images = pd.read_csv(ROOT / "results/node_dataset_split/node_images_152_split.csv")
    points = pd.read_csv(ROOT / "results/node_dataset_split/node_detection_points_152_split.csv")
    val_images = images.loc[images["split"].eq("validation")].copy()
    val_points = points.loc[points["split"].eq("validation")].copy()
    assert len(val_images) == 24, f"Expected 24 validation images, found {len(val_images)}"
    assert len(val_points) == 152, f"Expected 152 validation points, found {len(val_points)}"
    assert set(val_images["plant_id"]) == VALIDATION_PLANTS
    assert not set(val_images["plant_id"]) & LOCKED_TEST_PLANTS
    assert not set(val_points["plant_id"]) & LOCKED_TEST_PLANTS
    return val_images, val_points


def cache_logits(model: torch.nn.Module, images: pd.DataFrame, points: pd.DataFrame) -> list[dict]:
    # The lookup validates the complete 152-image Dataset V1 inventory.  Passing
    # only the validation rows would incorrectly look like 128 unexpected files.
    all_images = pd.read_csv(ROOT / "results/node_dataset_split/node_images_152_split.csv")
    dataset = NodeDetectionDataset(images, points, image_lookup(ROOT, all_images), training=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=node_detection_collate_fn)
    by_filename = {name: group for name, group in points.groupby("filename", sort=False)}
    cache = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            started = time.perf_counter()
            logits = model(batch["image"].to(device())).detach().cpu()
            elapsed = time.perf_counter() - started
            filename = batch["filename"][0]
            width, height = map(float, batch["original_size"][0].tolist())
            point_rows = by_filename[filename]
            cache.append({
                "filename": filename, "plant_id": point_rows["plant_id"].iloc[0],
                "acquisition_date": str(point_rows["acquisition_date"].iloc[0]),
                "width": width, "height": height,
                "gt": point_rows[["x", "y"]].to_numpy(dtype=float).tolist(),
                "visibility": point_rows["visibility"].astype(str).tolist(),
                "logits": logits, "inference_seconds": elapsed,
            })
    assert len(cache) == 24
    return cache


def score(cache: list[dict], threshold: float, tolerance_fraction: float, detailed: bool = False):
    totals = {"tp": 0, "fp": 0, "fn": 0}
    errors, count_errors, image_rows, prediction_rows, visibility_rows = [], [], [], [], []
    for item in cache:
        predicted = peaks(item["logits"], threshold, item["width"], item["height"])
        diagonal = float(np.hypot(item["width"], item["height"]))
        pairs, unmatched_predicted, unmatched_gt = match(predicted, item["gt"], tolerance_fraction * diagonal)
        totals["tp"] += len(pairs); totals["fp"] += len(unmatched_predicted); totals["fn"] += len(unmatched_gt)
        pair_map = {pred_index: (gt_index, distance) for pred_index, gt_index, distance in pairs}
        pair_errors = [distance for _, _, distance in pairs]
        errors.extend(pair_errors)
        count_error = abs(len(predicted) - len(item["gt"])); count_errors.append(count_error)
        if not detailed:
            continue
        per_image = metric_record(len(pairs), len(unmatched_predicted), len(unmatched_gt), pair_errors, [count_error])
        image_rows.append({"filename": item["filename"], "plant_id": item["plant_id"],
                           "acquisition_date": item["acquisition_date"], "gt_count": len(item["gt"]),
                           "predicted_count": len(predicted), "selected_threshold": threshold,
                           "inference_seconds": item["inference_seconds"],
                           "absolute_count_error": count_error, **per_image})
        for pred_index, point in enumerate(predicted):
            gt_index, distance = pair_map.get(pred_index, (None, None))
            prediction_rows.append({
                "filename": item["filename"], "plant_id": item["plant_id"],
                "acquisition_date": item["acquisition_date"], "pred_x_original": point[0],
                "pred_y_original": point[1], "confidence": point[2], "matched_gt": gt_index is not None,
                "matched_gt_x": item["gt"][gt_index][0] if gt_index is not None else np.nan,
                "matched_gt_y": item["gt"][gt_index][1] if gt_index is not None else np.nan,
                "gt_visibility": item["visibility"][gt_index] if gt_index is not None else None,
                "distance_pixels": distance,
                "distance_normalized": distance / diagonal if distance is not None else np.nan,
                "selected_threshold": threshold,
            })
        matched_gt = {gt_index for _, gt_index, _ in pairs}
        visibility_rows.extend({"filename": item["filename"], "plant_id": item["plant_id"],
                                "acquisition_date": item["acquisition_date"], "visibility": visibility,
                                "matched": index in matched_gt}
                               for index, visibility in enumerate(item["visibility"]))
    return metric_record(totals["tp"], totals["fp"], totals["fn"], errors, count_errors), image_rows, prediction_rows, visibility_rows


def aggregate_images(rows: pd.DataFrame, column: str) -> pd.DataFrame:
    records = []
    for value, group in rows.groupby(column, sort=True):
        record = metric_record(int(group.TP.sum()), int(group.FP.sum()), int(group.FN.sum()),
                               group.mean_localization_error_pixels.dropna().tolist(),
                               group.absolute_count_error.astype(int).tolist())
        records.append({column: value, "images": len(group), "gt_nodes": int(group.gt_count.sum()),
                        "predicted_nodes": int(group.predicted_count.sum()), **record})
    return pd.DataFrame(records)


def aggregate_visibility(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for visibility, group in rows.groupby("visibility", sort=True):
        matched, total = int(group.matched.sum()), len(group)
        records.append({"visibility": visibility, "gt_nodes": total, "matched_gt_nodes": matched,
                        "missed_gt_nodes": total - matched, "recall": safe_divide(matched, total),
                        "note": "FP cannot be assigned to one GT visibility subgroup."})
    return pd.DataFrame(records)


def plot_training_curves(run_dir: Path, best_epoch: int | None) -> None:
    history = pd.read_csv(run_dir / "training_history.csv")
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(history.epoch, history.train_loss, label="train")
    axis.plot(history.epoch, history.validation_loss, label="validation")
    if best_epoch is not None:
        axis.axvline(best_epoch, color="black", linestyle="--", linewidth=1, label=f"best epoch {best_epoch}")
    axis.set(xlabel="Epoch", ylabel="Focal loss"); axis.legend(); figure.tight_layout()
    figure.savefig(run_dir / "training_curves.png", dpi=180); plt.close(figure)


def make_overlays(run_dir: Path, image_rows: pd.DataFrame, predictions: pd.DataFrame) -> None:
    overlays = run_dir / "validation_overlays"; overlays.mkdir(exist_ok=True)
    ground_truth = pd.read_csv(ROOT / "results/node_ground_truth/tables/node_points_152.csv")
    locations = image_lookup(ROOT, pd.read_csv(ROOT / "results/node_dataset_split/node_images_152_split.csv"))
    thumbnails = []
    for row in image_rows.sort_values("filename").itertuples(index=False):
        with Image.open(locations[row.filename]) as raw:
            original_width, original_height = raw.size; image = raw.convert("RGB"); image.thumbnail((1200, 900))
        scale_x, scale_y = image.width / original_width, image.height / original_height
        draw = ImageDraw.Draw(image); current = predictions.loc[predictions.filename.eq(row.filename)]
        matched = {(round(x, 4), round(y, 4)) for x, y in current.loc[current.matched_gt, ["matched_gt_x", "matched_gt_y"]].dropna().to_numpy()}
        for point in ground_truth.loc[ground_truth.filename.eq(row.filename)].itertuples(index=False):
            x, y = point.x * scale_x, point.y * scale_y; color = "lime" if (round(point.x, 4), round(point.y, 4)) in matched else "red"
            draw.ellipse((x - 8, y - 8, x + 8, y + 8), outline=color, width=3)
        for point in current.itertuples(index=False):
            x, y = point.pred_x_original * scale_x, point.pred_y_original * scale_y; color = "cyan" if point.matched_gt else "yellow"
            draw.line((x - 7, y, x + 7, y), fill=color, width=3); draw.line((x, y - 7, x, y + 7), fill=color, width=3)
        title = f"{row.filename} | GT {row.gt_count} Pred {row.predicted_count} | TP {row.TP} FP {row.FP} FN {row.FN} | F1 {row.f1:.2f}"
        draw.rectangle((0, 0, image.width, 28), fill="black"); draw.text((5, 7), title, fill="white")
        image.save(overlays / f"{Path(row.filename).stem}_overlay.jpg", quality=90)
        thumbnail = image.copy(); thumbnail.thumbnail((400, 300)); thumbnails.append((row.filename, thumbnail))
    def montage(items: list[tuple[str, Image.Image]], destination: Path) -> None:
        sheet = Image.new("RGB", (1600, ((len(items) + 3) // 4) * 325), "white"); draw = ImageDraw.Draw(sheet)
        for index, (name, thumbnail) in enumerate(items):
            x, y = (index % 4) * 400, (index // 4) * 325; sheet.paste(thumbnail, (x + (400 - thumbnail.width) // 2, y)); draw.text((x + 4, y + 302), name, fill="black")
        sheet.save(destination, quality=90)
    montage(thumbnails, run_dir / "validation_contact_sheet.jpg")
    worst = set(image_rows.sort_values(["f1", "filename"]).head(8).filename)
    montage([item for item in thumbnails if item[0] in worst], run_dir / "validation_worst_cases.jpg")


def v2_v3_comparison(run_dir: Path, threshold: float, metrics: pd.DataFrame, config: dict) -> None:
    v2 = ROOT / "results/node_detection_resnet18_v2"; v2_config = json.loads((v2 / "config.json").read_text()); v2_metrics = pd.read_csv(v2 / "validation_metrics_by_tolerance.csv")
    v2_image_metrics = pd.read_csv(v2 / "validation_image_metrics.csv")
    rows = []
    for name, run_config, result in (("V2", v2_config, v2_metrics), ("V3 FPN", config, metrics)):
        values = {"model": name, "best_epoch": run_config.get("best_epoch"), "best_validation_loss": run_config.get("best_validation_loss"), "selected_validation_threshold": run_config.get("selected_validation_threshold", threshold)}
        for tolerance in TOLERANCES:
            record = result.loc[np.isclose(result.tolerance_fraction_diagonal, tolerance)].iloc[0]; label = f"@{tolerance * 100:.1f}%"
            values.update({f"precision_{label}": record.precision, f"recall_{label}": record.recall, f"f1_{label}": record.f1})
        if name == "V2":
            values.update({"node_count_mae": float(v2_image_metrics.absolute_count_error.mean()),
                           "exact_node_count_accuracy": float(v2_image_metrics.absolute_count_error.eq(0).mean())})
        else:
            primary = result.loc[np.isclose(result.tolerance_fraction_diagonal, .01)].iloc[0]
            values.update({"node_count_mae": primary.node_count_mae,
                           "exact_node_count_accuracy": primary.exact_node_count_accuracy})
        rows.append(values)
    table = pd.DataFrame(rows); table.to_csv(run_dir / "v2_vs_v3_validation.csv", index=False)
    (run_dir / "v2_vs_v3_report.md").write_text("# V2 versus V3 validation comparison\n\nThis is a validation-only controlled comparison: Dataset V1, plant split, input and heatmap resolution, augmentation, optimiser, learning rates, loss, heatmap prior and early stopping were held fixed. It tests whether V3 multi-scale FPN fusion improves localisation over the V2 coarse decoder. Do not claim improvement until the table below is reviewed.\n\n" + table.to_markdown(index=False) + "\n\nTEST SET WAS NOT USED OR EVALUATED.\n")


def main() -> None:
    args = arguments(); run_dir = resolve_run_dir(args); checkpoint = run_dir / "checkpoints/best_model.pt"
    if not checkpoint.is_file(): raise FileNotFoundError(f"Required best checkpoint not found: {checkpoint}")
    config_path = run_dir / "config.json"; config = json.loads(config_path.read_text())
    images, points = load_validation_data(); model = make_model(args.model_variant).to(device())
    state = torch.load(checkpoint, map_location=device(), weights_only=False); model.load_state_dict(state["model_state"], strict=True)
    cache = cache_logits(model, images, points)
    sweep = pd.DataFrame([{"threshold": value, **score(cache, value, .01)[0]} for value in THRESHOLDS]); sweep.to_csv(run_dir / "validation_threshold_sweep.csv", index=False)
    threshold = float(sweep.sort_values(["f1", "threshold"], ascending=[False, False]).iloc[0].threshold)
    metrics = pd.DataFrame([{"tolerance_fraction_diagonal": tolerance, "threshold": threshold, **score(cache, threshold, tolerance)[0]} for tolerance in TOLERANCES]); metrics.to_csv(run_dir / "validation_metrics_by_tolerance.csv", index=False)
    _, image_rows, prediction_rows, visibility_rows = score(cache, threshold, .01, detailed=True)
    image_metrics, predictions = pd.DataFrame(image_rows), pd.DataFrame(prediction_rows)
    image_metrics.to_csv(run_dir / "validation_image_metrics.csv", index=False); predictions.to_csv(run_dir / "validation_predictions.csv", index=False)
    aggregate_images(image_metrics, "plant_id").to_csv(run_dir / "validation_by_plant.csv", index=False); aggregate_images(image_metrics, "acquisition_date").to_csv(run_dir / "validation_by_date.csv", index=False); aggregate_visibility(pd.DataFrame(visibility_rows)).to_csv(run_dir / "validation_by_visibility.csv", index=False)
    figure, axis = plt.subplots(figsize=(6, 4)); axis.plot(sweep.threshold, sweep.f1, marker="o"); axis.axvline(threshold, color="black", linestyle="--", label=f"selected {threshold:.2f}"); axis.set(xlabel="Validation confidence threshold", ylabel="F1 @ 1% image diagonal"); axis.legend(); figure.tight_layout(); figure.savefig(run_dir / "validation_threshold_curve.png", dpi=180); plt.close(figure)
    config["selected_validation_threshold"] = threshold; config["test_evaluated"] = False; config_path.write_text(json.dumps(config, indent=2) + "\n")
    plot_training_curves(run_dir, config.get("best_epoch")); make_overlays(run_dir, image_metrics, predictions)
    report = f"# Validation report\n\nModel variant: `{args.model_variant}`\n\nCheckpoint: `{checkpoint.relative_to(ROOT)}` (best checkpoint only)\n\nSelected validation threshold: **{threshold:.2f}** (maximum F1 @ 1% diagonal; higher threshold wins ties).\n\n## Metrics\n\n" + metrics.to_markdown(index=False) + "\n\n## Scope\n\nOnly P04, P07 and P17 (24 images; 152 GT nodes) were evaluated. P01, P14 and P16 were explicitly excluded. TEST SET WAS NOT USED OR EVALUATED.\n"
    (run_dir / "validation_report.md").write_text(report)
    if args.model_variant == "fpn": v2_v3_comparison(run_dir, threshold, metrics, config)
    print(f"Validation evaluation complete: variant={args.model_variant}, threshold={threshold:.2f}"); print("TEST SET WAS NOT USED OR EVALUATED.")


if __name__ == "__main__":
    main()
