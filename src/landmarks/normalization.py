"""Wrist-relative, scale-invariant landmark normalization.

Raw MediaPipe landmarks are in normalized image coordinates, making them
dependent on hand distance and camera framing. This module re-centers
landmarks relative to the wrist and scales them to a unit bounding box,
producing coordinates that generalize across distances and framings.
"""

from __future__ import annotations

import numpy as np

# MediaPipe hand landmark indices
WRIST = 0
MIDDLE_FINGER_MCP = 9


def normalize_landmarks(
    landmarks: np.ndarray,
    method: str = "wrist_relative",
    scale_invariant: bool = True,
) -> np.ndarray:
    """Normalize a (21, 3) landmark array.

    Args:
        landmarks: Raw landmark coordinates, shape (21, 3).
        method: Normalization origin — "wrist_relative" or "palm_center".
        scale_invariant: If True, scale to unit distance from origin to MCP.

    Returns:
        Normalized landmarks, shape (21, 3).
    """
    lm = landmarks.copy()

    # Re-center
    if method == "wrist_relative":
        origin = lm[WRIST]
    elif method == "palm_center":
        origin = lm[[WRIST, MIDDLE_FINGER_MCP]].mean(axis=0)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    lm -= origin

    # Scale
    if scale_invariant:
        ref_dist = np.linalg.norm(lm[MIDDLE_FINGER_MCP])
        if ref_dist > 1e-6:
            lm /= ref_dist

    return lm
