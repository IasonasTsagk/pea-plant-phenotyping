"""One-time held-out test evaluator for the frozen V3 FPN pea-node detector.

This program has no tuning path: the confidence threshold is deliberately
hard-coded to the validation-selected value of 0.40. It only creates a test
dataloader after asserting that it contains P01, P14 and P16 exclusively.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.node_detection_dataset import NodeDetectionDataset, image_lookup, node_detection_collate_fn  # noqa: E402
from src.node_detection_metrics import match, peaks  # noqa: E402
from src.models.resnet_node_heatmap_fpn import ResNet18NodeHeatmapFPN  # noqa: E402
from src.train_node_detector import device  # noqa: E402

RUN_DIR = ROOT / "results/node_detection_resnet18_v3_fpn"
CHECKPOINT = RUN_DIR / "checkpoints/best_model.pt"
THRESHOLD = 0.40  # Frozen from validation; never sweep or tune on test.
TEST_PLANTS = {"P01", "P14", "P16"}
NON_TEST_PLANTS = {"P04", "P07", "P17"}
TOLERANCES = (0.005, 0.01, 0.02)


def divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def metrics(tp: int, fp: int, fn: int, errors: list[float], count_errors: list[int]) -> dict:
    precision = divide(tp, tp + fp)
    recall = divide(tp, tp + fn)
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "f1": divide(2 * precision * recall, precision + recall),
        "mean_localization_error_pixels": float(np.mean(errors)) if errors else np.nan,
        "median_localization_error_pixels": float(np.median(errors)) if errors else np.nan,
        "node_count_mae": float(np.mean(count_errors)) if count_errors else np.nan,
        "median_absolute_count_error": float(np.median(count_errors)) if count_errors else np.nan,
        "exact_node_count_accuracy": float(np.mean(np.asarray(count_errors) == 0)) if count_errors else np.nan,
    }


def load_locked_test_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_images = pd.read_csv(ROOT / "results/node_dataset_split/node_images_152_split.csv")
    all_points = pd.read_csv(ROOT / "results/node_dataset_split/node_detection_points_152_split.csv")
    test_images = all_images.loc[all_images["split"].eq("test")].copy()
    test_points = all_points.loc[all_points["split"].eq("test")].copy()
    # Fail closed if a split, annotation count, or allowed Plant_ID changes.
    assert len(test_images) == 24, f"Expected 24 held-out test images; found {len(test_images)}"
    assert len(test_points) == 153, f"Expected 153 held-out test points; found {len(test_points)}"
    assert set(test_images.plant_id) == TEST_PLANTS
    assert set(test_points.plant_id) == TEST_PLANTS
    assert not set(test_images.plant_id) & NON_TEST_PLANTS
    assert not set(test_points.plant_id) & NON_TEST_PLANTS
    return all_images, test_images, test_points


def predict_test(model: torch.nn.Module, all_images: pd.DataFrame, test_images: pd.DataFrame,
                 test_points: pd.DataFrame) -> list[dict]:
    lookup = image_lookup(ROOT, all_images)
    dataset = NodeDetectionDataset(test_images, test_points, lookup, training=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=node_detection_collate_fn)
    grouped_points = {name: group for name, group in test_points.groupby("filename", sort=False)}
    cache = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            started = time.perf_counter()
            logits = model(batch["image"].to(device())).detach().cpu()
            elapsed = time.perf_counter() - started
            filename = batch["filename"][0]
            points = grouped_points[filename]
            width, height = map(float, batch["original_size"][0].tolist())
            cache.append({
                "filename": filename,
                "plant_id": points.plant_id.iloc[0],
                "acquisition_date": str(points.acquisition_date.iloc[0]),
                "width": width,
                "height": height,
                "gt": points[["x", "y"]].to_numpy(dtype=float).tolist(),
                "visibility": points.visibility.astype(str).tolist(),
                "logits": logits,
                "inference_seconds": elapsed,
            })
    assert len(cache) == 24
    assert {item["plant_id"] for item in cache} == TEST_PLANTS
    return cache


def score(cache: list[dict], tolerance_fraction: float, detailed: bool = False):
    totals = {"tp": 0, "fp": 0, "fn": 0}
    errors, count_errors, image_rows, prediction_rows, visibility_rows = [], [], [], [], []
    for item in cache:
        predicted = peaks(item["logits"], THRESHOLD, item["width"], item["height"])
        diagonal = float(np.hypot(item["width"], item["height"]))
        pairs, unmatched_predicted, unmatched_gt = match(predicted, item["gt"], tolerance_fraction * diagonal)
        totals["tp"] += len(pairs); totals["fp"] += len(unmatched_predicted); totals["fn"] += len(unmatched_gt)
        pair_map = {pred_index: (gt_index, distance) for pred_index, gt_index, distance in pairs}
        pair_errors = [distance for _, _, distance in pairs]
        errors.extend(pair_errors)
        count_error = abs(len(predicted) - len(item["gt"])); count_errors.append(count_error)
        if not detailed:
            continue
        per_image = metrics(len(pairs), len(unmatched_predicted), len(unmatched_gt), pair_errors, [count_error])
        image_rows.append({
            "filename": item["filename"], "plant_id": item["plant_id"],
            "acquisition_date": item["acquisition_date"], "gt_count": len(item["gt"]),
            "predicted_count": len(predicted), "threshold": THRESHOLD,
            "inference_seconds": item["inference_seconds"], "absolute_count_error": count_error,
            **per_image,
        })
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
                "threshold": THRESHOLD,
            })
        matched_gt = {gt_index for _, gt_index, _ in pairs}
        visibility_rows.extend({
            "filename": item["filename"], "plant_id": item["plant_id"],
            "acquisition_date": item["acquisition_date"], "visibility": visibility,
            "matched": index in matched_gt,
        } for index, visibility in enumerate(item["visibility"]))
    return metrics(totals["tp"], totals["fp"], totals["fn"], errors, count_errors), image_rows, prediction_rows, visibility_rows


def aggregate(cache: list[dict], column: str) -> pd.DataFrame:
    records = []
    for value in sorted({str(item[column]) for item in cache}):
        subset = [item for item in cache if str(item[column]) == value]
        record, _, _, _ = score(subset, 0.01)
        records.append({column: value, "images": len(subset), "gt_nodes": sum(len(item["gt"]) for item in subset),
                        "threshold": THRESHOLD, **record})
    return pd.DataFrame(records)


def visibility_summary(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for visibility, group in rows.groupby("visibility", sort=True):
        matched, total = int(group.matched.sum()), len(group)
        records.append({"visibility": visibility, "gt_nodes": total, "matched_gt_nodes": matched,
                        "missed_gt_nodes": total - matched, "recall": divide(matched, total),
                        "threshold": THRESHOLD,
                        "note": "FP cannot be assigned to an individual GT visibility subgroup."})
    return pd.DataFrame(records)


def create_overlays(all_images: pd.DataFrame, image_rows: pd.DataFrame, predictions: pd.DataFrame) -> None:
    overlay_dir = RUN_DIR / "test_overlays"; overlay_dir.mkdir(exist_ok=True)
    ground_truth = pd.read_csv(ROOT / "results/node_ground_truth/tables/node_points_152.csv")
    lookup = image_lookup(ROOT, all_images)
    thumbnails = []
    for row in image_rows.sort_values("filename").itertuples(index=False):
        with Image.open(lookup[row.filename]) as raw:
            original_width, original_height = raw.size
            image = raw.convert("RGB"); image.thumbnail((1200, 900))
        scale_x, scale_y = image.width / original_width, image.height / original_height
        draw = ImageDraw.Draw(image)
        current = predictions.loc[predictions.filename.eq(row.filename)]
        matched = {(round(x, 4), round(y, 4)) for x, y in current.loc[
            current.matched_gt, ["matched_gt_x", "matched_gt_y"]].dropna().to_numpy()}
        for point in ground_truth.loc[ground_truth.filename.eq(row.filename)].itertuples(index=False):
            x, y = point.x * scale_x, point.y * scale_y
            color = "lime" if (round(point.x, 4), round(point.y, 4)) in matched else "red"
            draw.ellipse((x - 8, y - 8, x + 8, y + 8), outline=color, width=3)
        for point in current.itertuples(index=False):
            x, y = point.pred_x_original * scale_x, point.pred_y_original * scale_y
            color = "cyan" if point.matched_gt else "yellow"
            draw.line((x - 7, y, x + 7, y), fill=color, width=3)
            draw.line((x, y - 7, x, y + 7), fill=color, width=3)
        title = f"{row.filename} | GT {row.gt_count} Pred {row.predicted_count} | TP {row.TP} FP {row.FP} FN {row.FN} | F1 {row.f1:.2f}"
        draw.rectangle((0, 0, image.width, 28), fill="black"); draw.text((5, 7), title, fill="white")
        image.save(overlay_dir / f"{Path(row.filename).stem}_overlay.jpg", quality=90)
        thumbnail = image.copy(); thumbnail.thumbnail((400, 300)); thumbnails.append((row.filename, thumbnail))

    def montage(items: list[tuple[str, Image.Image]], destination: Path) -> None:
        sheet = Image.new("RGB", (1600, ((len(items) + 3) // 4) * 325), "white")
        draw = ImageDraw.Draw(sheet)
        for index, (name, thumbnail) in enumerate(items):
            x, y = (index % 4) * 400, (index // 4) * 325
            sheet.paste(thumbnail, (x + (400 - thumbnail.width) // 2, y))
            draw.text((x + 4, y + 302), name, fill="black")
        sheet.save(destination, quality=90)

    montage(thumbnails, RUN_DIR / "test_contact_sheet.jpg")
    worst = set(image_rows.sort_values(["f1", "filename"]).head(8).filename)
    montage([item for item in thumbnails if item[0] in worst], RUN_DIR / "test_worst_cases.jpg")


def final_reports(test_metrics: pd.DataFrame) -> None:
    validation_metrics = pd.read_csv(RUN_DIR / "validation_metrics_by_tolerance.csv")
    config = json.loads((RUN_DIR / "config.json").read_text())
    primary_validation = validation_metrics.loc[np.isclose(validation_metrics.tolerance_fraction_diagonal, .01)].iloc[0]
    primary_test = test_metrics.loc[np.isclose(test_metrics.tolerance_fraction_diagonal, .01)].iloc[0]
    summary = pd.DataFrame([
        {"partition": "validation", "purpose": "model development and threshold selection", "images": 24,
         "gt_nodes": 152, "threshold": config.get("selected_validation_threshold", THRESHOLD),
         "primary_f1_at_1pct_diagonal": primary_validation.f1,
         "precision_at_1pct_diagonal": primary_validation.precision,
         "recall_at_1pct_diagonal": primary_validation.recall},
        {"partition": "test", "purpose": "one-time final held-out generalization evaluation", "images": 24,
         "gt_nodes": 153, "threshold": THRESHOLD,
         "primary_f1_at_1pct_diagonal": primary_test.f1,
         "precision_at_1pct_diagonal": primary_test.precision,
         "recall_at_1pct_diagonal": primary_test.recall},
    ])
    summary.to_csv(RUN_DIR / "final_validation_test_summary.csv", index=False)
    (RUN_DIR / "test_report.md").write_text(
        "# Held-out test report\n\n"
        f"Frozen checkpoint: `{CHECKPOINT.relative_to(ROOT)}` (best epoch 7).\n\n"
        f"Frozen confidence threshold: **{THRESHOLD:.2f}**, selected on validation only. No test threshold sweep or tuning was performed.\n\n"
        "## Test metrics\n\n" + test_metrics.to_markdown(index=False) +
        "\n\nPrimary metric: F1 @ 1.0% of original-image diagonal.\n"
    )
    (RUN_DIR / "final_model_report.md").write_text(
        "# Final frozen V3 FPN model report\n\n"
        "## Frozen experiment\n\n"
        "- Architecture: ResNet-18 + FPN-style multi-scale decoder\n"
        "- Checkpoint: `checkpoints/best_model.pt` (best epoch 7)\n"
        "- Input / heatmap: 768×576 / 192×144\n"
        "- Post-processing: 3×3 local-max NMS and corrected Hungarian one-to-one matching\n"
        "- Threshold: 0.40, selected on validation only\n\n"
        "## Partition separation\n\n"
        "**VALIDATION** was used for model development and threshold selection.\n\n"
        "**TEST** was used once for final held-out generalization evaluation; it was not used for training or tuning.\n\n"
        + summary.to_markdown(index=False) + "\n\n## Test metrics by tolerance\n\n" + test_metrics.to_markdown(index=False) + "\n"
    )


def main() -> None:
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Missing frozen best checkpoint: {CHECKPOINT}")
    all_images, test_images, test_points = load_locked_test_split()
    model = ResNet18NodeHeatmapFPN(heatmap_prior_probability=0.10).to(device())
    state = torch.load(CHECKPOINT, map_location=device(), weights_only=False)
    model.load_state_dict(state["model_state"], strict=True)
    cache = predict_test(model, all_images, test_images, test_points)
    metric_rows = []
    for tolerance in TOLERANCES:
        result, _, _, _ = score(cache, tolerance)
        metric_rows.append({"tolerance_fraction_diagonal": tolerance, "threshold": THRESHOLD, **result})
    test_metrics = pd.DataFrame(metric_rows)
    test_metrics.to_csv(RUN_DIR / "test_metrics_by_tolerance.csv", index=False)
    _, image_rows, prediction_rows, visibility_rows = score(cache, .01, detailed=True)
    image_metrics, predictions = pd.DataFrame(image_rows), pd.DataFrame(prediction_rows)
    image_metrics.to_csv(RUN_DIR / "test_image_metrics.csv", index=False)
    predictions.to_csv(RUN_DIR / "test_predictions.csv", index=False)
    aggregate(cache, "plant_id").to_csv(RUN_DIR / "test_by_plant.csv", index=False)
    aggregate(cache, "acquisition_date").to_csv(RUN_DIR / "test_by_date.csv", index=False)
    visibility_summary(pd.DataFrame(visibility_rows)).to_csv(RUN_DIR / "test_by_visibility.csv", index=False)
    create_overlays(all_images, image_metrics, predictions)
    final_reports(test_metrics)
    print(f"Held-out test evaluation complete at frozen threshold {THRESHOLD:.2f}.")


if __name__ == "__main__":
    main()
