"""Non-destructive diagnostics for verifying SAM2 mask fidelity across devices."""
from __future__ import annotations

import time
import cv2
import numpy as np


def predict_raw(predictor, rgb: np.ndarray, *, points=None, labels=None, box=None) -> dict:
    """Capture SAM2 image-predictor outputs before local conversion or visualization.

    Phase 4B requests ``multimask_output=False``. SAM2 consequently returns exactly
    one candidate mask; index 0 is selected and its matching score is index 0.
    """
    started = time.perf_counter(); predictor.set_image(rgb)
    masks, scores, logits = predictor.predict(point_coords=points, point_labels=labels, box=box, multimask_output=False)
    elapsed = time.perf_counter() - started
    selected = np.asarray(masks[0]).astype(bool)
    return {"masks": masks, "scores": scores, "logits": logits, "selected": selected, "seconds": elapsed,
            "selected_index": 0}


def component_statistics(mask: np.ndarray) -> tuple[int, int, float]:
    """Measure components only; do not remove, smooth, or otherwise modify them."""
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    foreground = int(mask.sum())
    largest = int(areas.max()) if len(areas) else 0
    return int(count - 1), largest, (largest / foreground if foreground else 0.0)


def agreement(mps: np.ndarray, cpu: np.ndarray) -> dict:
    """Return device-output agreement, explicitly not accuracy against ground truth."""
    intersection = int(np.logical_and(mps, cpu).sum()); union = int(np.logical_or(mps, cpu).sum())
    differing = int(np.logical_xor(mps, cpu).sum()); total = int(mps.size)
    return {"Intersection_Pixels": intersection, "Union_Pixels": union,
            "MPS_vs_CPU_IoU": intersection / union if union else 1.0,
            "Differing_Pixel_Count": differing, "Differing_Pixel_Fraction": differing / total,
            "MPS_Foreground_Fraction": float(mps.mean()), "CPU_Foreground_Fraction": float(cpu.mean())}
