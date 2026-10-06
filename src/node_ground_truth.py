"""Read-only CVAT XML parsing and structural QC for the pea-node ground truth."""
from __future__ import annotations

from pathlib import Path
import re
import zipfile
import xml.etree.ElementTree as ET

import pandas as pd

FILENAME = re.compile(r"^(?P<plant>P(?:0[1-9]|1[0-9]|20))_(?P<date>20\d{2}-\d{2}-\d{2})_.*\.(?:jpg|jpeg|png)$", re.I)
DATES = {"2026-09-15", "2026-09-18", "2026-09-25", "2026-09-29"}
PLANTS = {f"P{i:02d}" for i in range(1, 21) if i != 6}
VISIBILITY = {"visible", "partially_occluded", "uncertain"}
POINT_COLUMNS = ["filename", "plant_id", "acquisition_date", "source_dataset", "image_width", "image_height", "node_index", "x", "y", "x_normalized", "y_normalized", "visibility"]


def parse_export(path: Path, source_dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read CVAT for images 1.1 annotations.xml directly from a ZIP."""
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if Path(name).name == "annotations.xml"]
        if len(names) != 1:
            raise ValueError(f"{path.name}: expected exactly one annotations.xml")
        root = ET.fromstring(archive.read(names[0]))
    points, images = [], []
    for image in root.findall("image"):
        filename = image.get("name", "")
        match = FILENAME.fullmatch(filename)
        plant, date = (match.group("plant").upper(), match.group("date")) if match else (None, None)
        width, height = image.get("width"), image.get("height")
        images.append({"filename": filename, "plant_id": plant, "acquisition_date": date, "source_dataset": source_dataset, "image_width": width, "image_height": height})
        for element in image.findall("points"):
            attributes = {item.get("name"): (item.text or "").strip() for item in element.findall("attribute")}
            raw_coordinates = element.get("points", "").split(",")
            x, y = (raw_coordinates + [None, None])[:2]
            points.append({"filename": filename, "plant_id": plant, "acquisition_date": date, "source_dataset": source_dataset, "image_width": width, "image_height": height, "node_index": attributes.get("node_index"), "x": x, "y": y, "visibility": attributes.get("visibility"), "label": element.get("label")})
    return pd.DataFrame(points), pd.DataFrame(images)


def coerce(points: pd.DataFrame, images: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    for frame in (points, images):
        for column in ("image_width", "image_height"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in ("node_index", "x", "y"):
        points[column] = pd.to_numeric(points[column], errors="coerce")
    points["node_index"] = points["node_index"].astype("Int64")
    points["x_normalized"] = points["x"] / points["image_width"]
    points["y_normalized"] = points["y"] / points["image_height"]
    return points, images


def structural_errors(points: pd.DataFrame, images: pd.DataFrame, expected_filenames: set[str]) -> list[str]:
    errors: list[str] = []
    if len(images) != 152: errors.append(f"expected 152 images, found {len(images)}")
    if len(points) != 963: errors.append(f"expected 963 points, found {len(points)}")
    if images.filename.duplicated().any(): errors.append("duplicate filenames across exports")
    if set(images.filename) != expected_filenames: errors.append("CVAT image filenames do not exactly match approved 152-image selection")
    if images.plant_id.isna().any(): errors.append("unexpected filename format")
    if not set(images.plant_id.dropna()).issubset(PLANTS): errors.append("unexpected Plant_ID")
    if not set(images.acquisition_date.dropna()).issubset(DATES): errors.append("unexpected acquisition date")
    if (images.image_width.le(0) | images.image_height.le(0) | images.image_width.isna() | images.image_height.isna()).any(): errors.append("invalid image dimensions")
    if points.label.ne("node").any(): errors.append("non-node labels")
    if points.node_index.isna().any() or points.node_index.le(0).any(): errors.append("missing or invalid node_index")
    if points.visibility.isna().any() or not set(points.visibility.dropna()).issubset(VISIBILITY): errors.append("missing or invalid visibility")
    if points.x.isna().any() or points.y.isna().any(): errors.append("x/y parsing problems")
    if ((points.x < 0) | (points.y < 0) | (points.x >= points.image_width) | (points.y >= points.image_height)).any(): errors.append("points outside image bounds")
    if points.duplicated(["filename", "x", "y"]).any(): errors.append("duplicate node coordinates")
    if points.duplicated(["filename", "node_index"]).any(): errors.append("duplicate node_index within image")
    return errors


def image_manifest(points: pd.DataFrame, images: pd.DataFrame) -> pd.DataFrame:
    counts = points.pivot_table(index="filename", columns="visibility", values="x", aggfunc="size", fill_value=0)
    overall = points.groupby("filename").agg(node_count=("x", "size"), min_node_index=("node_index", "min"), max_node_index=("node_index", "max")).reset_index()
    result = images.merge(overall, on="filename", how="left").merge(counts, on="filename", how="left")
    for column in VISIBILITY:
        result[f"{column}_count"] = result.get(column, 0).fillna(0).astype(int)
    return result[["filename", "plant_id", "acquisition_date", "source_dataset", "image_width", "image_height", "node_count", "visible_count", "partially_occluded_count", "uncertain_count", "min_node_index", "max_node_index"]]


def temporal_summary(images: pd.DataFrame) -> pd.DataFrame:
    return images.groupby(["plant_id", "acquisition_date"], as_index=False).agg(number_of_selected_views=("filename", "size"), mean_node_count=("node_count", "mean"), min_node_count=("node_count", "min"), max_node_count=("node_count", "max"), visible_nodes=("visible_count", "sum"), partially_occluded_nodes=("partially_occluded_count", "sum"), uncertain_nodes=("uncertain_count", "sum"))
