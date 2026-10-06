"""Prompted, pretrained SAM2 image-pilot helpers; no training or ground truth."""
from __future__ import annotations

from pathlib import Path
import time
import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
MODEL_NAME = "sam2.1_hiera_tiny"


def choose_device() -> str:
    """Prefer Apple Metal Performance Shaders, otherwise use CPU; never assume CUDA."""
    return "mps" if torch.backends.mps.is_available() else "cpu"


def build_predictor(checkpoint: Path, device: str | None = None) -> tuple[SAM2ImagePredictor, str]:
    """Build a pretrained Tiny SAM2 image predictor on MPS or CPU."""
    selected = device or choose_device()
    try:
        model = build_sam2(MODEL_CONFIG, str(checkpoint), device=selected)
        return SAM2ImagePredictor(model), selected
    except RuntimeError:
        if selected != "mps":
            raise
        model = build_sam2(MODEL_CONFIG, str(checkpoint), device="cpu")
        return SAM2ImagePredictor(model), "cpu"


def load_rgb(path: Path) -> np.ndarray:
    """Read a raw image into memory without modifying it."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Unreadable image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def prompted_mask(predictor: SAM2ImagePredictor, rgb: np.ndarray, *, points: np.ndarray | None = None,
                  labels: np.ndarray | None = None, box: np.ndarray | None = None) -> tuple[np.ndarray, float, float]:
    """Return one best SAM2 mask, predicted score, and elapsed inference seconds."""
    started = time.perf_counter()
    predictor.set_image(rgb)
    masks, scores, _ = predictor.predict(point_coords=points, point_labels=labels, box=box, multimask_output=False)
    elapsed = time.perf_counter() - started
    return masks[0].astype(bool), float(scores[0]), elapsed


def mask_overlay(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Create a magenta inspection overlay in memory, preserving the original array."""
    output = rgb.copy(); color = np.array([255, 0, 255], dtype=np.float32)
    output[mask] = ((1 - alpha) * output[mask] + alpha * color).astype(np.uint8)
    return output
