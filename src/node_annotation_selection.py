"""Deterministic preparation of the manual pea-node annotation scale-up set."""
from __future__ import annotations

from pathlib import Path
import filecmp
import shutil

import pandas as pd
from PIL import Image, ImageDraw

DATES = ("2026-09-15", "2026-09-18", "2026-09-25", "2026-09-29")
LIVING_PLANTS = tuple(f"P{value:02d}" for value in range(1, 21) if value != 6)
PILOT = {
    ("2026-09-15", "P07"): ("P07_2026-09-15_11-17-31.jpg", "P07_2026-09-15_11-16-35.jpg"),
    ("2026-09-15", "P12"): ("P12_2026-09-15_11-22-48.jpg", "P12_2026-09-15_11-23-32.jpg"),
    ("2026-09-18", "P07"): ("P07_2026-09-18_12-32-26.jpg", "P07_2026-09-18_12-32-03.jpg"),
    ("2026-09-18", "P12"): ("P12_2026-09-18_12-37-00.jpg", "P12_2026-09-18_12-37-30.jpg"),
    ("2026-09-25", "P07"): ("P07_2026-09-25_15-10-49.jpg", "P07_2026-09-25_15-11-19.jpg"),
    ("2026-09-25", "P12"): ("P12_2026-09-25_15-16-05.jpg", "P12_2026-09-25_15-15-37.jpg"),
    ("2026-09-29", "P07"): ("P07_2026-09-29_07-51-07.jpg", "P07_2026-09-29_07-50-32.jpg"),
    ("2026-09-29", "P12"): ("P12_2026-09-29_07-56-01.jpg", "P12_2026-09-29_07-56-31.jpg"),
}
SELECTION_COLUMNS = ["acquisition_date", "plant_id", "filename", "source_path", "view_slot", "readable", "selection_group", "selection_method", "selection_reason", "pilot_status", "annotation_status"]


def _as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().eq("true")


def _valid(group: pd.DataFrame) -> pd.DataFrame:
    duplicate = _as_bool(group["duplicate_hash"]) if "duplicate_hash" in group else False
    return group.loc[_as_bool(group["readable"]) & ~duplicate & group["source_path"].map(lambda value: Path(value).is_file())].copy()


def _choose(group: pd.DataFrame, slots: tuple[str, ...], group_name: str) -> pd.Series:
    by_slot = group.set_index("view_slot", drop=False)
    for index, slot in enumerate(slots):
        if slot in by_slot.index:
            row = by_slot.loc[slot]
            if isinstance(row, pd.DataFrame):
                raise ValueError(f"Multiple valid rows use {slot} in {group_name}.")
            row = row.copy()
            row["selection_group"] = group_name
            row["selection_method"] = "preferred_slot" if index == 0 else "fallback_slot"
            row["selection_reason"] = f"{slot} {'preferred' if index == 0 else 'fallback'} acquisition slot"
            return row
    if group_name == "B":
        extras = group.loc[group["view_slot"].str.extract(r"(\d+)")[0].astype(int).ge(7)].sort_values("view_slot")
        if not extras.empty:
            row = extras.iloc[0].copy()
            row["selection_group"] = "B"; row["selection_method"] = "fallback_slot"
            row["selection_reason"] = "later acquisition slot used after view_04–view_06 were unavailable"
            return row
    raise ValueError(f"No valid {group_name} selection is available.")


def build_selection(manifest: pd.DataFrame) -> pd.DataFrame:
    """Select exactly two valid images per living plant/date, preserving Pilot V1."""
    rows: list[pd.Series] = []
    for date in DATES:
        for plant in LIVING_PLANTS:
            subset = manifest.loc[(manifest["acquisition_date"].eq(date)) & (manifest["plant_id"].eq(plant))]
            valid = _valid(subset)
            pilot_names = PILOT.get((date, plant))
            if pilot_names:
                pilot_rows = valid.loc[valid["filename"].isin(pilot_names)].copy()
                if len(pilot_rows) != 2 or set(pilot_rows["filename"]) != set(pilot_names):
                    raise ValueError(f"Validated Pilot V1 filename is absent or invalid: {date}/{plant}.")
                for row in pilot_rows.itertuples(index=False):
                    record = pd.Series(row._asdict())
                    record["selection_group"] = "A" if record["view_slot"] == "view_02" else "B"
                    record["selection_method"] = "validated_pilot"
                    record["selection_reason"] = "Validated Pilot V1 filename preserved; generic slot rule overridden"
                    record["pilot_status"] = "validated_pilot"; record["annotation_status"] = "validated"
                    rows.append(record)
            else:
                first = _choose(valid, ("view_02", "view_01", "view_03"), "A")
                second = _choose(valid.loc[valid["filename"].ne(first["filename"])], ("view_05", "view_04", "view_06"), "B")
                for record in (first, second):
                    record["pilot_status"] = "not_pilot"; record["annotation_status"] = "pending"
                    rows.append(record)
    selection = pd.DataFrame(rows)[SELECTION_COLUMNS].sort_values(["acquisition_date", "plant_id", "selection_group"]).reset_index(drop=True)
    validate_selection(selection)
    return selection


def validate_selection(selection: pd.DataFrame) -> None:
    checks = {
        "selected rows": len(selection) == 152,
        "living plants": selection["plant_id"].nunique() == 19,
        "dates": selection["acquisition_date"].nunique() == 4,
        "two per plant/date": selection.groupby(["acquisition_date", "plant_id"]).size().eq(2).all(),
        "P06 excluded": not selection["plant_id"].eq("P06").any(),
        "no unreadable selection": _as_bool(selection["readable"]).all(),
        "unique filenames": selection["filename"].is_unique,
        "pilot count": selection["pilot_status"].eq("validated_pilot").sum() == 16,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("Selection validation failed: " + ", ".join(failed))


def copy_pending(selection: pd.DataFrame, destination: Path, batch_index: int = 0, batch_count: int = 1) -> int:
    """Copy a deterministic pending-image batch; fail if destination names drift."""
    pending = selection.loc[selection["annotation_status"].eq("pending")].reset_index(drop=True)
    expected = set(pending["filename"])
    destination.mkdir(parents=True, exist_ok=True)
    existing = {path.name for path in destination.iterdir() if path.is_file()}
    unexpected = existing - expected
    if unexpected:
        raise ValueError(f"CVAT upload directory contains unexpected file(s): {sorted(unexpected)[:5]}")
    copied = 0
    for record in pending.iloc[batch_index::batch_count].itertuples(index=False):
        source, target = Path(record.source_path), destination / record.filename
        if target.exists():
            if target.stat().st_size != source.stat().st_size or not filecmp.cmp(source, target, shallow=False):
                raise ValueError(f"Existing copied image differs from source: {target}")
            continue
        shutil.copy2(source, target)
        copied += 1
    return copied


def review_contact_sheets(selection: pd.DataFrame, destination: Path, date: str | None = None) -> int:
    """Write compact two-image review sheets without biological annotations."""
    subset = selection if date is None else selection.loc[selection["acquisition_date"].eq(date)]
    count = 0
    for (day, plant), group in subset.groupby(["acquisition_date", "plant_id"]):
        group = group.sort_values("selection_group")
        sheet = Image.new("RGB", (900, 390), "white"); draw = ImageDraw.Draw(sheet)
        for index, record in enumerate(group.itertuples(index=False)):
            x = index * 450
            with Image.open(record.source_path) as image:
                image.draft("RGB", (430, 320)); thumbnail = image.convert("RGB")
                thumbnail.thumbnail((430, 320), Image.Resampling.LANCZOS)
                sheet.paste(thumbnail, (x + (450 - thumbnail.width) // 2, 2))
            draw.text((x + 8, 326), record.filename, fill="black")
            draw.text((x + 8, 344), f"{record.view_slot} | {day} | {plant}", fill="black")
            draw.text((x + 8, 362), f"{record.pilot_status} | {record.selection_method}", fill="black")
        target = destination / day; target.mkdir(parents=True, exist_ok=True)
        sheet.save(target / f"{plant}_selection_review.png")
        count += 1
    return count
