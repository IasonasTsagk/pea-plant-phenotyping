"""Conservative, unvalidated vegetation-mask baselines for Phase 3."""
from __future__ import annotations

import cv2
import numpy as np


MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
PARAMETERS = {
    "exg_threshold": 20,  # ExG is computed on 8-bit RGB values, range approximately -510 to 510.
    "hue_low": 35,        # OpenCV hue scale: 0–179.
    "hue_high": 95,
    "saturation_low": 40,
    "lab_a_high": 125,   # OpenCV LAB a* is offset around 128; greener pixels tend to be lower.
}


def exg_index(rgb: np.ndarray) -> np.ndarray:
    """Return Excess Green = 2G − R − B on the original 8-bit channel scale."""
    red, green, blue = (rgb[:, :, index].astype(np.int16) for index in range(3))
    return 2 * green - red - blue


def otsu_mask(scalar: np.ndarray) -> np.ndarray:
    """Return an Otsu mask after safe uint8 normalisation of a scalar image."""
    normalised = cv2.normalize(scalar, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(normalised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def clean_mask(mask: np.ndarray) -> np.ndarray:
    """Apply only a 3×3 opening and closing; no erosion/dilation is used directly."""
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL, iterations=1)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, MORPH_KERNEL, iterations=1)


def exg_masks(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return manual ExG, Otsu ExG, and minimally cleaned manual ExG masks."""
    index = exg_index(rgb)
    manual = np.where(index >= PARAMETERS["exg_threshold"], 255, 0).astype(np.uint8)
    return manual, otsu_mask(index), clean_mask(manual)


def hsv_masks(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return hue/saturation manual mask, saturation Otsu comparison, and cleaned mask."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    manual = cv2.inRange(hsv, (PARAMETERS["hue_low"], PARAMETERS["saturation_low"], 0),
                          (PARAMETERS["hue_high"], 255, 255))
    return manual, otsu_mask(hsv[:, :, 1]), clean_mask(manual)


def lab_masks(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return low-a* manual mask, inverse-a* Otsu comparison, and cleaned mask."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    manual = np.where(lab[:, :, 1] <= PARAMETERS["lab_a_high"], 255, 0).astype(np.uint8)
    inverted_a = 255 - lab[:, :, 1]
    return manual, otsu_mask(inverted_a), clean_mask(manual)


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int] = (255, 0, 255), alpha: float = 0.45) -> np.ndarray:
    """Overlay a mask for inspection; the returned display image never alters raw input."""
    overlay = rgb.copy()
    foreground = mask > 0
    overlay[foreground] = (1 - alpha) * overlay[foreground] + alpha * np.asarray(color)
    return overlay.astype(np.uint8)
