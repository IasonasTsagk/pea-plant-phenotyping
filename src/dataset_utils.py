"""Safe image discovery, inventory, and visual-assessment utilities."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import hashlib
import re

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ExifTags

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
MANUAL_COLUMNS = ["View_Type", "Vegetation_Density", "Occlusion", "Trellis_Interference",
                  "Background_Complexity", "Plant_Visibility", "Manual_Image_Quality",
                  "Segmentation_Suitability", "Notes"]
TIMESTAMP_PATTERN = re.compile(r"(?P<date>20\d{6})_(?P<time>\d{6})")


def discover_images(image_directory: Path) -> list[Path]:
    """Return supported images in a deterministic filename order; never alter inputs."""
    return sorted((path for path in image_directory.iterdir()
                   if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
                  key=lambda path: path.name.lower())


def timestamp_from_filename(filename: str) -> datetime | None:
    """Parse common smartphone ``YYYYMMDD_HHMMSS`` timestamp text when present."""
    match = TIMESTAMP_PATTERN.search(Path(filename).stem)
    if not match:
        return None
    try:
        return datetime.strptime(match.group("date") + match.group("time"), "%Y%m%d%H%M%S")
    except ValueError:
        return None


def sha256(path: Path) -> str:
    """Return a content hash in chunks, without changing the file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _exif_datetime(path: Path) -> str | None:
    """Read capture time where EXIF is available, returning an ISO-like string."""
    try:
        with Image.open(path) as image:
            tags = {ExifTags.TAGS.get(key, key): value for key, value in image.getexif().items()}
        value = tags.get("DateTimeOriginal") or tags.get("DateTime")
        return str(value) if value else None
    except Exception:
        return None


def image_statistics(path: Path) -> dict:
    """Compute descriptive technical/quality statistics, not plant measurements."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return {"Is_Readable": False}
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    height, width = rgb.shape[:2]
    means, stds = rgb.reshape(-1, 3).mean(axis=0), rgb.reshape(-1, 3).std(axis=0)
    return {
        "Is_Readable": True, "Width_px": width, "Height_px": height, "Channels": 3,
        "Color_Mode": "RGB", "Aspect_Ratio": round(width / height, 5),
        "Total_Pixels": width * height,
        "Mean_R": round(float(means[0]), 3), "Mean_G": round(float(means[1]), 3),
        "Mean_B": round(float(means[2]), 3), "Std_R": round(float(stds[0]), 3),
        "Std_G": round(float(stds[1]), 3), "Std_B": round(float(stds[2]), 3),
        "Mean_Brightness": round(float(gray.mean()), 3), "Brightness_SD": round(float(gray.std()), 3),
        "Mean_HSV_Saturation": round(float(hsv[:, :, 1].mean()), 3),
        "Mean_HSV_Value": round(float(hsv[:, :, 2].mean()), 3),
        "Sharpness_Laplacian_Variance": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 3),
        "Contrast_Gray_SD": round(float(gray.std()), 3),
    }


def build_inventory(image_directory: Path) -> pd.DataFrame:
    """Create a complete automatic inventory plus intentionally blank human fields."""
    records = []
    for number, path in enumerate(discover_images(image_directory), start=1):
        timestamp = timestamp_from_filename(path.name)
        record = {"Image_ID": f"image_{number:02d}", "Filename": path.name,
                  "File_Extension": path.suffix.lower(), "Filename_Date": timestamp.date().isoformat() if timestamp else None,
                  "Filename_Time": timestamp.time().isoformat() if timestamp else None,
                  "EXIF_DateTime": _exif_datetime(path), "File_Size_Bytes": path.stat().st_size,
                  "SHA256": sha256(path)}
        record.update(image_statistics(path))
        record.update({column: "" for column in MANUAL_COLUMNS})
        records.append(record)
    return pd.DataFrame(records)


def duplicates_by_hash(inventory: pd.DataFrame) -> pd.DataFrame:
    """Return potential byte-identical duplicates; an empty table means none found."""
    return inventory[inventory.duplicated("SHA256", keep=False)].sort_values("SHA256")


def save_csv(frame: pd.DataFrame, destination: Path) -> None:
    """Save a derived table, creating only its output directory."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
