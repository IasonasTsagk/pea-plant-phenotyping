"""Documented whole-plant SAM2 prompt definitions for Phase 4B."""
from __future__ import annotations

import json
import numpy as np


def parse_points(value: str) -> np.ndarray:
    """Parse JSON ``[[x, y], ...]`` prompt coordinates into SAM2 float32 input."""
    return np.asarray(json.loads(value), dtype=np.float32)


def parse_box(value: str) -> np.ndarray:
    """Parse JSON ``[x0, y0, x1, y1]`` into SAM2 float32 box input."""
    return np.asarray(json.loads(value), dtype=np.float32)


def strategy_inputs(record) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Return SAM2 points, labels, and optional box for a documented strategy row."""
    positive = parse_points(record["Positive_Points"])
    negative = parse_points(record["Negative_Points"])
    strategy = record["Prompt_Strategy"]
    box = parse_box(record["Bounding_Box"]) if "box" in strategy else None
    if strategy == "wholeplant_box":
        return None, None, box
    if strategy == "wholeplant_multi_positive":
        return positive, np.ones(len(positive), dtype=np.int32), None
    if strategy == "wholeplant_pos_neg":
        points = np.vstack([positive, negative])
        return points, np.r_[np.ones(len(positive), dtype=np.int32), np.zeros(len(negative), dtype=np.int32)], None
    if strategy == "wholeplant_box_positive":
        return positive, np.ones(len(positive), dtype=np.int32), box
    if strategy == "wholeplant_box_pos_neg":
        points = np.vstack([positive, negative])
        return points, np.r_[np.ones(len(positive), dtype=np.int32), np.zeros(len(negative), dtype=np.int32)], box
    raise ValueError(f"Unknown strategy: {strategy}")
