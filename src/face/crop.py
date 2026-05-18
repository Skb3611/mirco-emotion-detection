"""Face region cropping from bounding boxes and landmarks."""

from __future__ import annotations

import numpy as np


def expand_box(
    box: tuple[int, int, int, int],
    width: int,
    height: int,
    margin_px: int = 0,
    margin_ratio: float = 0.0,
) -> tuple[int, int, int, int]:
    """Expand (x1, y1, x2, y2) with pixel and/or proportional margin."""
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    pad_x = margin_px + int(bw * margin_ratio)
    pad_y = margin_px + int(bh * margin_ratio)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(width, x2 + pad_x)
    y2 = min(height, y2 + pad_y)
    return x1, y1, x2, y2


def bbox_from_landmarks_pixel(
    landmarks_px: np.ndarray,
    width: int,
    height: int,
    margin_ratio: float = 0.12,
) -> tuple[int, int, int, int]:
    """Build a tight axis-aligned box around 2D landmark points."""
    xs = landmarks_px[:, 0]
    ys = landmarks_px[:, 1]
    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()), int(ys.max())
    return expand_box((x1, y1, x2, y2), width, height, margin_ratio=margin_ratio)


def bbox_from_relative(
    rel_box,
    width: int,
    height: int,
    margin_ratio: float = 0.12,
) -> tuple[int, int, int, int]:
    """Convert MediaPipe relative bounding box to pixel coords."""
    x = int(rel_box.xmin * width)
    y = int(rel_box.ymin * height)
    w = int(rel_box.width * width)
    h = int(rel_box.height * height)
    return expand_box((x, y, x + w, y + h), width, height, margin_ratio=margin_ratio)


def crop_rgb(
    rgb: np.ndarray,
    box: tuple[int, int, int, int],
) -> np.ndarray | None:
    """Return face crop or None if the box is invalid."""
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return None
    return rgb[y1:y2, x1:x2].copy()


def box_area(box: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)
