"""Exploratory visualisations for the greenhouse image pilot."""
from __future__ import annotations

from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np


def load_rgb(path: Path) -> np.ndarray:
    """Load an image as RGB for display; raises a useful error if it cannot be read."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unreadable image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_contact_sheet(paths: list[Path], destination: Path, columns: int = 4) -> None:
    """Save labelled thumbnail contact sheet without altering source photographs."""
    rows = int(np.ceil(len(paths) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(16, 4 * rows))
    for axis, path in zip(axes.flat, paths):
        thumbnail = load_rgb(path)
        scale = min(1.0, 600 / thumbnail.shape[1])
        if scale < 1:
            thumbnail = cv2.resize(thumbnail, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        axis.imshow(thumbnail); axis.set_title(path.name, fontsize=8); axis.axis("off")
    for axis in axes.flat[len(paths):]: axis.axis("off")
    figure.tight_layout(); destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, bbox_inches="tight"); plt.close(figure)


def save_colour_space_panel(path: Path, destination: Path) -> None:
    """Visualise RGB, HSV and LAB channels for one representative image."""
    rgb = load_rgb(path); hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV); lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    panels = [(rgb[:, :, i], label, cmap) for i, (label, cmap) in enumerate((("R", "Reds"), ("G", "Greens"), ("B", "Blues")))]
    panels += [(hsv[:, :, i], label, cmap) for i, (label, cmap) in enumerate((("H", "hsv"), ("S", "gray"), ("V", "gray")))]
    panels += [(lab[:, :, i], label, cmap) for i, (label, cmap) in enumerate((("L*", "gray"), ("a*", "coolwarm"), ("b*", "coolwarm")))]
    figure, axes = plt.subplots(3, 3, figsize=(12, 10))
    for axis, (array, label, cmap) in zip(axes.flat, panels):
        axis.imshow(array, cmap=cmap); axis.set_title(label); axis.axis("off")
    figure.suptitle(path.name); figure.tight_layout(); destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, bbox_inches="tight"); plt.close(figure)


def save_quality_plots(inventory, destination: Path) -> None:
    """Create brightness, saturation, sharpness, and contrast diagnostic plots."""
    metrics = [("Mean_Brightness", "Mean brightness (0–255)"), ("Mean_HSV_Saturation", "Mean HSV saturation (0–255)"),
               ("Sharpness_Laplacian_Variance", "Laplacian variance"), ("Contrast_Gray_SD", "Grayscale contrast (SD)")]
    figure, axes = plt.subplots(2, 2, figsize=(13, 8))
    for axis, (column, title) in zip(axes.flat, metrics):
        axis.bar(inventory["Image_ID"], inventory[column], color="#3f7f5f")
        axis.set_title(title); axis.tick_params(axis="x", rotation=60, labelsize=7); axis.set_ylabel(column)
    figure.tight_layout(); destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, bbox_inches="tight"); plt.close(figure)


QUALITY_METRICS = {
    "Mean_Brightness": "Mean brightness (0–255)",
    "Mean_HSV_Saturation": "Mean HSV saturation (0–255)",
    "Sharpness_Laplacian_Variance": "Sharpness: Laplacian variance",
    "Contrast_Gray_SD": "Contrast: grayscale SD",
}


def save_metric_comparisons(inventory, destination: Path) -> None:
    """Save one readable comparison panel for the four Phase 2 quality metrics."""
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    for axis, (column, label) in zip(axes.flat, QUALITY_METRICS.items()):
        axis.bar(inventory["Image_ID"], inventory[column], color="#3f7f5f")
        axis.axhline(inventory[column].median(), color="#c04b37", linestyle="--", label="Median")
        axis.set_title(label); axis.set_ylabel(label); axis.tick_params(axis="x", rotation=55, labelsize=8)
        axis.legend(frameon=False)
    figure.tight_layout(); destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, bbox_inches="tight"); plt.close(figure)


def save_quality_extremes(inventory, paths_by_name: dict[str, Path], destination: Path) -> None:
    """Show high/low examples for each metric to support visual metric interpretation."""
    selections = []
    for metric, label in QUALITY_METRICS.items():
        selections.extend([(metric, label, "Low", inventory.loc[inventory[metric].idxmin()]),
                           (metric, label, "High", inventory.loc[inventory[metric].idxmax()])])
    figure, axes = plt.subplots(4, 2, figsize=(10, 20))
    for axis, (metric, label, direction, row) in zip(axes.flat, selections):
        image = load_rgb(paths_by_name[row["Filename"]])
        image = cv2.resize(image, None, fx=min(1, 700 / image.shape[1]), fy=min(1, 700 / image.shape[1]), interpolation=cv2.INTER_AREA)
        axis.imshow(image); axis.axis("off")
        axis.set_title(f"{direction} {label}\n{row['Image_ID']} | {row['Filename']}\n{row[metric]:.2f}", fontsize=9)
    figure.tight_layout(); destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, bbox_inches="tight"); plt.close(figure)


def save_selection_contact_sheet(inventory, paths_by_name: dict[str, Path], destination: Path, columns: int = 3) -> None:
    """Save a metric-labelled contact sheet for human representative-image selection."""
    rows = int(np.ceil(len(inventory) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(15, 5 * rows))
    for axis, (_, record) in zip(axes.flat, inventory.iterrows()):
        image = load_rgb(paths_by_name[record["Filename"]])
        scale = min(1, 520 / image.shape[1])
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        axis.imshow(image); axis.axis("off")
        axis.set_title(f"{record['Image_ID']} | {record['Filename']}\n"
                       f"brightness {record['Mean_Brightness']:.1f} | saturation {record['Mean_HSV_Saturation']:.1f}\n"
                       f"sharpness {record['Sharpness_Laplacian_Variance']:.1f} | contrast {record['Contrast_Gray_SD']:.1f}", fontsize=8)
    for axis in axes.flat[len(inventory):]: axis.axis("off")
    figure.tight_layout(); destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, bbox_inches="tight"); plt.close(figure)
