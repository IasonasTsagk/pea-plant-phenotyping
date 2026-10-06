"""Mild, reversible preprocessing experiments for Phase 2."""
from __future__ import annotations

import cv2
import numpy as np


def colour_spaces(rgb: np.ndarray) -> dict[str, np.ndarray]:
    """Return RGB, grayscale, HSV, and LAB representations without writing files."""
    return {"RGB": rgb, "Grayscale": cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY),
            "HSV": cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV), "LAB": cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)}


def mild_filters(rgb: np.ndarray) -> dict[str, np.ndarray]:
    """Return conservative 3×3 Gaussian and median versions for visual comparison."""
    return {"Original": rgb, "Gaussian blur (3×3)": cv2.GaussianBlur(rgb, (3, 3), 0),
            "Median blur (3×3)": cv2.medianBlur(rgb, 3)}


def clahe_luminance(rgb: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Optionally enhance LAB lightness only; use for comparison, never overwrite inputs."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
