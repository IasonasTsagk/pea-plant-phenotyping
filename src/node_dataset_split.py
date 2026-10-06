"""Fixed, reproducible Plant_ID split for pea-node Dataset V1."""
from __future__ import annotations

import pandas as pd

SPLITS = {
    "train": ("P02", "P03", "P05", "P08", "P09", "P10", "P11", "P12", "P13", "P15", "P18", "P19", "P20"),
    "validation": ("P04", "P07", "P17"),
    "test": ("P01", "P14", "P16"),
}
EXPECTED_IMAGES = {"train": 104, "validation": 24, "test": 24}
EXPECTED_POINTS = {"train": 658, "validation": 152, "test": 153}


def mapping() -> dict[str, str]:
    return {plant: split for split, plants in SPLITS.items() for plant in plants}


def assign(images: pd.DataFrame, points: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    plant_map = mapping()
    images = images.copy(); points = points.copy()
    images["split"] = images["plant_id"].map(plant_map)
    points["split"] = points["plant_id"].map(plant_map)
    validate(images, points)
    return images, points


def validate(images: pd.DataFrame, points: pd.DataFrame) -> None:
    errors: list[str] = []
    expected_plants = set(mapping())
    if set(images.plant_id) != expected_plants: errors.append("unexpected or missing Plant_ID")
    if images.split.isna().any() or points.split.isna().any(): errors.append("unassigned image or point")
    if images.groupby("plant_id").split.nunique().gt(1).any(): errors.append("Plant_ID leakage")
    if images.filename.duplicated().any(): errors.append("filename leakage")
    if points.merge(images[["filename", "split"]], on="filename", suffixes=("_point", "_image"), how="left").eval("split_point != split_image").any(): errors.append("point/image split mismatch")
    if len(images) != 152 or images.filename.nunique() != 152: errors.append("all images not assigned exactly once")
    if len(points) != 963: errors.append("all points not assigned exactly once")
    if images.groupby("split").size().to_dict() != EXPECTED_IMAGES: errors.append("expected image counts mismatch")
    if points.groupby("split").size().to_dict() != EXPECTED_POINTS: errors.append("expected point counts mismatch")
    date_counts = images.groupby(["split", "acquisition_date"]).size().to_dict()
    expected_date_counts = {(split, date): count for split, count in {"train": 26, "validation": 6, "test": 6}.items() for date in ("2026-09-15", "2026-09-18", "2026-09-25", "2026-09-29")}
    if date_counts != expected_date_counts: errors.append("date coverage mismatch")
    if errors: raise ValueError("Split QC failed: " + "; ".join(errors))
