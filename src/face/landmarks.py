"""Landmark normalization and AU-like geometric features."""

from __future__ import annotations

import numpy as np

# MediaPipe Face Mesh indices (468-point model)
IDX_NOSE_TIP = 1
IDX_CHIN = 152
IDX_LEFT_EYE_OUTER = 33
IDX_RIGHT_EYE_OUTER = 263
IDX_LEFT_EYEBROW = 70
IDX_RIGHT_EYEBROW = 300
IDX_MOUTH_LEFT = 61
IDX_MOUTH_RIGHT = 291
IDX_UPPER_LIP = 13
IDX_LOWER_LIP = 14
IDX_LEFT_CHEEK = 234
IDX_RIGHT_CHEEK = 454


def landmarks_to_pixel(
    landmarks_normalized: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Shape (N, 2) pixel coordinates from normalized [0, 1] landmarks."""
    px = np.zeros((len(landmarks_normalized), 2), dtype=np.float32)
    px[:, 0] = landmarks_normalized[:, 0] * width
    px[:, 1] = landmarks_normalized[:, 1] * height
    return px


def normalize_landmarks_face_relative(
    landmarks_normalized: np.ndarray,
    reference_indices: tuple[int, int] = (IDX_LEFT_EYE_OUTER, IDX_RIGHT_EYE_OUTER),
) -> np.ndarray:
    """
    Translate to nose tip and scale by inter-ocular distance.
    Output is scale/translation invariant — useful as ML features.
    """
    pts = landmarks_normalized[:, :2].astype(np.float64)
    origin = pts[IDX_NOSE_TIP].copy()
    pts -= origin

    left, right = reference_indices
    iod = np.linalg.norm(pts[right] - pts[left])
    if iod < 1e-6:
        iod = 1.0
    pts /= iod
    return pts.astype(np.float32)


def normalize_landmarks_image(
    landmarks_normalized: np.ndarray,
) -> np.ndarray:
    """Keep MediaPipe's native normalized coords (x, y in [0, 1])."""
    return landmarks_normalized[:, :2].astype(np.float32)


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def extract_au_like_features(
    landmarks_normalized: np.ndarray,
) -> dict[str, float]:
    """
    Geometry-based proxies for Action Units (no FACS coder required).
    All distances are normalized by inter-ocular distance.
    """
    rel = normalize_landmarks_face_relative(landmarks_normalized)
    iod = 1.0  # already scaled

    le, re = rel[IDX_LEFT_EYE_OUTER], rel[IDX_RIGHT_EYE_OUTER]
    lb, rb = rel[IDX_LEFT_EYEBROW], rel[IDX_RIGHT_EYEBROW]
    ml, mr = rel[IDX_MOUTH_LEFT], rel[IDX_MOUTH_RIGHT]
    upper, lower = rel[IDX_UPPER_LIP], rel[IDX_LOWER_LIP]
    chin, nose = rel[IDX_CHIN], rel[IDX_NOSE_TIP]

    face_width = _dist(rel[IDX_LEFT_CHEEK], rel[IDX_RIGHT_CHEEK])

    brow_raise_l = _dist(lb, le) / iod
    brow_raise_r = _dist(rb, re) / iod
    mouth_width = _dist(ml, mr) / max(face_width, 1e-6)
    lip_open = _dist(upper, lower) / iod
    jaw_drop = _dist(chin, nose) / iod
    smile_asym = (mr[1] - ml[1])  # corner height diff in face-relative space

    return {
        "brow_raise_left": brow_raise_l,
        "brow_raise_right": brow_raise_r,
        "brow_raise_mean": (brow_raise_l + brow_raise_r) / 2.0,
        "mouth_width_ratio": mouth_width,
        "lip_opening": lip_open,
        "jaw_drop": jaw_drop,
        "smile_asymmetry": smile_asym,
    }


def au_features_to_vector(features: dict[str, float]) -> np.ndarray:
    """Fixed-order vector for sklearn / torch fusion."""
    keys = [
        "brow_raise_left",
        "brow_raise_right",
        "brow_raise_mean",
        "mouth_width_ratio",
        "lip_opening",
        "jaw_drop",
        "smile_asymmetry",
    ]
    return np.array([features[k] for k in keys], dtype=np.float32)
