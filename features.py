"""Normalize MediaPipe hand landmarks → 42 int16 features (0–1000).

Usage:
    from features import normalize_and_encode
    feats = normalize_and_encode(landmarks)  # list[NormalisedLandmark] with .x, .y

The 42 values are: wrist-normalized, scale-normalized x,y for 21 landmarks,
clamped and encoded as unsigned int16 in range 0–1000.
"""

import numpy as np

# MediaPipe landmark indices
WRIST = 0


def normalize_landmarks(landmarks, mirror_left: bool = False) -> np.ndarray:
    """Return (21, 2) normalized x,y array, wrist-centered and scale-normalized."""
    pts = np.array([[lm.x, lm.y] for lm in landmarks], dtype=np.float64)
    if mirror_left:
        pts[:, 0] = 1.0 - pts[:, 0]
    wrist = pts[WRIST]
    pts -= wrist
    span = max(pts.max() - pts.min(), 1e-6)
    pts /= span
    pts = (pts - pts.min()) / (pts.max() - pts.min() + 1e-6)
    return pts


def encode_int16(pts: np.ndarray) -> np.ndarray:
    """Map normalized (21, 2) → 42 int16 in 0–1000."""
    flat = pts.flatten()  # 42
    return np.clip(np.round(flat * 1000), 0, 1000).astype(np.int16)


def normalize_and_encode(landmarks, mirror_left: bool = False) -> np.ndarray:
    """End-to-end: raw landmarks → 42 int16 features."""
    pts = normalize_landmarks(landmarks, mirror_left)
    return encode_int16(pts)
