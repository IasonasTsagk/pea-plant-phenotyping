"""Read-only audit utilities for the future pea main-stem node dataset."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import os
from pathlib import Path
import re

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
EXPECTED_DATES = ("2026-09-15", "2026-09-18", "2026-09-25", "2026-09-29")
PILOT_PLANTS = ("P07", "P12")
PLANT_PATTERN = re.compile(r"^P(?:0[1-9]|1[0-9]|20)$", re.IGNORECASE)
TIMESTAMP_PATTERNS = (re.compile(r"(?P<date>20\d{6})[_-](?P<time>\d{6})"),
                      re.compile(r"(?P<date>20\d{2}[-_]\d{2}[-_]\d{2})[T_ -]+(?P<time>\d{2}[:_-]\d{2}[:_-]\d{2})"))


def configured_data_root(value: str | Path | None = None) -> Path:
    """Resolve an explicit root or ``NODE_DATA_ROOT``; never guess a Drive path."""
    raw_value = str(value) if value is not None else os.environ.get("NODE_DATA_ROOT", "")
    if not raw_value:
        raise FileNotFoundError("DATA_ROOT is unavailable. Set NODE_DATA_ROOT to the locally synced, read-only acquisition folder.")
    candidate = Path(raw_value).expanduser()
    if not candidate.is_dir():
        raise FileNotFoundError("DATA_ROOT is unavailable. Set NODE_DATA_ROOT to the locally synced, read-only acquisition folder.")
    return candidate.resolve()


def is_supported_image(path: Path, data_root: Path) -> bool:
    """Return whether a source file is supported after *relative* dataset filtering.

    A synced cloud-storage path may include a hidden ancestor. That ancestor
    is outside ``data_root`` and
    must not cause every legitimate image below the configured root to be
    rejected.
    """
    try:
        relative_parts = path.relative_to(data_root).parts
    except ValueError:
        return False
    relative_lower = tuple(part.lower() for part in relative_parts)
    return (
        path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and not any(part.startswith(".") for part in relative_parts)
        and ".ipynb_checkpoints" not in relative_lower
        and "checkpoint" not in path.name.lower()
        and "temp" not in path.name.lower()
    )


def discover_images(data_root: Path) -> list[Path]:
    """Recursively discover supported images in deterministic path order without writing to source."""
    return sorted(
        (path for path in data_root.rglob("*") if is_supported_image(path, data_root)),
        key=lambda path: str(path).lower(),
    )


def discovery_diagnostics(data_root: Path) -> dict[str, int | bool]:
    """Return auditable counts before and after relative-path filtering."""
    raw_supported_image_count = sum(
        path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        for path in data_root.rglob("*")
    )
    filtered_supported_image_count = len(discover_images(data_root))
    return {
        "data_root_exists": data_root.exists(),
        "data_root_is_directory": data_root.is_dir(),
        "raw_recursive_supported_image_count": raw_supported_image_count,
        "supported_image_count_after_filtering": filtered_supported_image_count,
    }


def parse_timestamp(filename: str) -> datetime | None:
    """Parse common acquisition timestamps in names when reliable, otherwise return None."""
    stem = Path(filename).stem
    for pattern in TIMESTAMP_PATTERNS:
        match = pattern.search(stem)
        if not match:
            continue
        date, clock = match.group("date").replace("-", "").replace("_", ""), match.group("time").replace(":", "").replace("_", "").replace("-", "")
        try:
            return datetime.strptime(date + clock, "%Y%m%d%H%M%S")
        except ValueError:
            pass
    return None


def sha256(path: Path) -> str:
    """Hash a source image in chunks for duplicate diagnosis, without altering it."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def path_metadata(path: Path, data_root: Path) -> tuple[str | None, str | None]:
    """Extract date and plant ID from directory components, not from image content."""
    parts = path.relative_to(data_root).parts[:-1]
    date = next((part for part in parts if part in EXPECTED_DATES or re.fullmatch(r"20\d{2}-\d{2}-\d{2}", part)), None)
    plant = next((part.upper() for part in parts if PLANT_PATTERN.fullmatch(part)), None)
    return date, plant


def inspect_image(path: Path, data_root: Path) -> dict:
    """Return deterministic technical metadata without changing a source image.

    Image-quality metrics are intentionally left blank: they are optional for
    candidate review, while opening every high-resolution source into OpenCV
    adds avoidable cost to a read-only inventory. SHA-256 is calculated later
    only for same-size candidate files, because unequal sizes cannot be exact
    binary duplicates.
    """
    date, plant = path_metadata(path, data_root)
    row = {"acquisition_date": date, "plant_id": plant, "timestamp": parse_timestamp(path.name), "filename": path.name,
           "source_path": str(path), "file_size_bytes": path.stat().st_size, "width": np.nan, "height": np.nan,
           "channels": np.nan, "image_mode": None, "readable": False, "sha256": None, "duplicate_hash": False,
           "plant_status": "dead" if plant == "P06" else "active", "view_slot": None, "view_label": "unknown",
           "annotation_status": "not_annotated", "notes": "", "mean_brightness": np.nan,
           "mean_saturation": np.nan, "sharpness_laplacian_variance": np.nan}
    try:
        with Image.open(path) as image:
            row.update({"width": image.width, "height": image.height, "channels": len(image.getbands()), "image_mode": image.mode})
            image.verify()
        row.update({"readable": True})
    except Exception as error:  # Preserve auditable failure rather than discarding a source file.
        row["notes"] = f"inspection_error: {type(error).__name__}"
    return row


MANIFEST_COLUMNS = ["acquisition_date", "plant_id", "timestamp", "filename", "source_path", "width", "height", "file_size_bytes", "channels", "image_mode", "view_slot", "view_label", "plant_status", "readable", "sha256", "duplicate_hash", "annotation_status", "notes", "mean_brightness", "mean_saturation", "sharpness_laplacian_variance"]


def finalize_manifest(manifest: pd.DataFrame, calculate_duplicate_hashes: bool = True) -> pd.DataFrame:
    """Sort records, assign neutral view slots, and optionally confirm duplicates."""
    if manifest.empty:
        return pd.DataFrame(columns=MANIFEST_COLUMNS)
    manifest = manifest.copy()
    manifest["timestamp"] = pd.to_datetime(manifest["timestamp"])
    manifest = manifest.sort_values(["acquisition_date", "plant_id", "timestamp", "filename"], na_position="last").reset_index(drop=True)
    manifest["view_slot"] = manifest.groupby(["acquisition_date", "plant_id"], dropna=False).cumcount().add(1).map(lambda value: f"view_{value:02d}")
    manifest["sha256"] = manifest.get("sha256", pd.Series(index=manifest.index, dtype=object))
    manifest["duplicate_hash"] = False
    if not calculate_duplicate_hashes:
        return manifest[MANIFEST_COLUMNS]
    # Metadata/header verification and content hashing are independent per file.
    # Threads reduce wall time for the cloud-synchronised but read-only source.
    # A byte-identical pair necessarily has the same size. Hashing only those
    # candidate groups preserves an exact duplicate check while avoiding a
    # needless full read of every large, cloud-synchronised image.
    same_size_candidates = manifest["file_size_bytes"].duplicated(keep=False)
    candidate_paths = manifest.loc[same_size_candidates, "source_path"].tolist()
    if candidate_paths:
        with ThreadPoolExecutor(max_workers=8) as executor:
            candidate_hashes = list(executor.map(lambda value: sha256(Path(value)), candidate_paths))
        manifest.loc[same_size_candidates, "sha256"] = candidate_hashes
    manifest["duplicate_hash"] = manifest["sha256"].notna() & manifest.duplicated("sha256", keep=False)
    return manifest[MANIFEST_COLUMNS]


def build_manifest(data_root: Path, paths: list[Path] | None = None, calculate_duplicate_hashes: bool = True) -> pd.DataFrame:
    """Build a complete or resumable read-only manifest from supplied paths."""
    selected_paths = discover_images(data_root) if paths is None else paths
    with ThreadPoolExecutor(max_workers=8) as executor:
        records = list(executor.map(lambda path: inspect_image(path, data_root), selected_paths))
    return finalize_manifest(pd.DataFrame(records), calculate_duplicate_hashes=calculate_duplicate_hashes)


def expected_count(date: str, plant_id: str) -> int | None:
    """Return supplied expected audit count; unknown/new dates deliberately return None."""
    if plant_id == "P06":
        return 1 if date in EXPECTED_DATES else None
    if date in {"2026-09-15", "2026-09-25"}: return 6
    if date == "2026-09-18": return 7
    sept29 = {"P01": 6, "P02": 9, "P03": 6, "P04": 4, "P05": 6, "P07": 6, "P08": 6, "P09": 6, "P10": 6, "P11": 6, "P12": 6, "P13": 6, "P14": 7, "P15": 6, "P16": 7, "P17": 6, "P18": 6, "P19": 7, "P20": 6}
    return sept29.get(plant_id) if date == "2026-09-29" else None


def summarise(manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create per-plant/date audit summary and explicit anomalies without treating P06 as missing."""
    observed = manifest.groupby(["acquisition_date", "plant_id"], dropna=False).agg(observed_image_count=("filename", "size"), unreadable_count=("readable", lambda values: int((~values.astype(bool)).sum())), duplicate_hash_count=("duplicate_hash", "sum"), duplicate_filename_count=("filename", lambda values: int(values.duplicated(keep=False).sum()))).reset_index()
    expected_grid = pd.DataFrame(
        [{"acquisition_date": date, "plant_id": plant, "expected_image_count": expected_count(date, plant)}
         for date in EXPECTED_DATES for plant in (f"P{number:02d}" for number in range(1, 21))]
    )
    # Expected rows with no source files must remain visible as missing counts.
    summary = expected_grid.merge(observed, on=["acquisition_date", "plant_id"], how="left")
    for column in ("observed_image_count", "unreadable_count", "duplicate_hash_count", "duplicate_filename_count"):
        summary[column] = summary[column].fillna(0).astype(int)
    extra_groups = observed.merge(expected_grid[["acquisition_date", "plant_id"]], on=["acquisition_date", "plant_id"], how="left", indicator=True)
    extra_groups = extra_groups.loc[extra_groups["_merge"].eq("left_only")].drop(columns="_merge")
    if not extra_groups.empty:
        extra_groups["expected_image_count"] = np.nan
        summary = pd.concat([summary, extra_groups], ignore_index=True, sort=False)
    summary["count_difference"] = summary.observed_image_count - summary.expected_image_count
    summary["plant_status"] = np.where(summary.plant_id.eq("P06"), "dead", "active")
    def status(row):
        if row.plant_id == "P06" and row.observed_image_count == 1: return "matches_expected_dead_plant"
        if pd.isna(row.expected_image_count): return "new_or_unexpected_date_or_plant"
        if row.acquisition_date == "2026-09-29" and row.count_difference != 0: return "acquisition_count_difference_manual_review"
        return "matches_expected" if row.count_difference == 0 else "count_difference_manual_review"
    summary["audit_status"] = summary.apply(status, axis=1)
    accepted_statuses = {"matches_expected", "matches_expected_dead_plant"}
    anomalies = summary.loc[~summary.audit_status.isin(accepted_statuses), ["acquisition_date", "plant_id", "plant_status", "observed_image_count", "expected_image_count", "count_difference", "audit_status"]].copy()
    return summary.sort_values(["acquisition_date", "plant_id"], na_position="last"), anomalies


def find_unexpected_extensions(data_root: Path) -> pd.DataFrame:
    """List visible non-image files within the source root for audit review only."""
    rows: list[dict[str, str]] = []
    for path in data_root.rglob("*"):
        relative_parts = path.relative_to(data_root).parts
        if not path.is_file() or any(part.startswith(".") for part in relative_parts):
            continue
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            rows.append({"source_path": str(path.resolve()), "filename": path.name, "extension": path.suffix.lower()})
    return pd.DataFrame(rows, columns=["source_path", "filename", "extension"])


def save_contact_sheets(manifest: pd.DataFrame, destination: Path) -> None:
    """Create thumbnail-only Pillow contact sheets grouped by date and plant.

    This never changes source images and avoids loading them at their full
    resolution into a plotting backend.
    """
    for (date, plant), group in manifest.groupby(["acquisition_date", "plant_id"], dropna=False):
        if pd.isna(date) or pd.isna(plant):
            continue
        paths = list(group.sort_values("view_slot").itertuples())
        columns, thumb_width, thumb_height, label_height = 3, 360, 270, 48
        rows = int(np.ceil(len(paths) / columns))
        sheet = Image.new("RGB", (columns * thumb_width, rows * (thumb_height + label_height)), "white")
        draw = ImageDraw.Draw(sheet)
        for index, record in enumerate(paths):
            column, row = index % columns, index // columns
            x, y = column * thumb_width, row * (thumb_height + label_height)
            try:
                with Image.open(record.source_path) as image:
                    # JPEG draft mode asks the decoder for a reduced-resolution
                    # image before conversion; this is sufficient for a visual
                    # contact sheet and never changes the source file.
                    image.draft("RGB", (thumb_width - 8, thumb_height - 8))
                    thumbnail = image.convert("RGB")
                    thumbnail.thumbnail((thumb_width - 8, thumb_height - 8), Image.Resampling.LANCZOS)
                    offset_x = x + (thumb_width - thumbnail.width) // 2
                    offset_y = y + (thumb_height - thumbnail.height) // 2
                    sheet.paste(thumbnail, (offset_x, offset_y))
            except Exception:
                draw.text((x + 4, y + 4), "Unreadable image", fill="red")
            timestamp = record.timestamp.strftime("%H:%M:%S") if pd.notna(record.timestamp) else "time unavailable"
            draw.text((x + 4, y + thumb_height + 3), f"{record.view_slot} | {record.filename}", fill="black")
            draw.text((x + 4, y + thumb_height + 20), timestamp, fill="black")
        target = destination / str(date); target.mkdir(parents=True, exist_ok=True)
        sheet.save(target / f"{plant}_contact_sheet.png")


def write_report(data_root: Path, manifest: pd.DataFrame, summary: pd.DataFrame, anomalies: pd.DataFrame, destination: Path) -> None:
    """Write a concise Markdown audit report from observed values only."""
    date_totals = manifest.groupby("acquisition_date").size().to_dict()
    resolutions = manifest[["width", "height", "channels", "image_mode"]].drop_duplicates().to_dict("records")
    anomaly_text = anomalies.to_csv(index=False) if not anomalies.empty else "None."
    destination.write_text("# Node dataset audit report\n\n" + f"- DATA_ROOT (read-only): `{data_root}`\n- Images observed: {len(manifest)}\n- Acquisition dates observed: {len(date_totals)}\n- Per-date totals: {date_totals}\n- Resolutions observed: {resolutions}\n- Unreadable images: {int((~manifest.readable).sum())}\n- Duplicate-hash records: {int(manifest.duplicate_hash.sum())}\n- P06 is retained as `dead`; one image per expected session is acceptable.\n\n" + "## Expected Drive inventory comparison\n\nExpected total for the four specified sessions is 483 images. Observed results above are generated from the configured local DATA_ROOT; new dates are reported rather than ignored.\n\n" + "## Anomalies requiring review\n\n```csv\n" + anomaly_text + "```\n", encoding="utf-8")
