"""Optical flow computation — Farneback (baseline) and RAFT (optional).

Computes dense motion vectors between consecutive video frames. In Sanctum,
optical flow is specifically used for cosmetic visual effects — driving a
pixel-smearing / temporal distortion effect (a stylized "reality unraveling" look)
during time-reversal frame rewind.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


class FarnebackOpticalFlow:
    """Dense Farneback optical flow computer."""

    def __init__(
        self,
        pyr_scale: float = 0.5,
        levels: int = 3,
        winsize: int = 15,
        iterations: int = 3,
        poly_n: int = 5,
        poly_sigma: float = 1.2,
    ) -> None:
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self.iterations = iterations
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma
        self._prev_gray: Optional[np.ndarray] = None

    def compute(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Compute optical flow vectors (dx, dy) between previous and current frame.

        Args:
            frame_bgr: Current BGR image array, shape (H, W, 3).

        Returns:
            Flow vector field array of shape (H, W, 2) containing (dx, dy) per pixel.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if self._prev_gray is None:
            self._prev_gray = gray
            return np.zeros((gray.shape[0], gray.shape[1], 2), dtype=np.float32)

        flow = cv2.calcOpticalFlowFarneback(
            prev=self._prev_gray,
            next=gray,
            flow=None,
            pyr_scale=self.pyr_scale,
            levels=self.levels,
            winsize=self.winsize,
            iterations=self.iterations,
            poly_n=self.poly_n,
            poly_sigma=self.poly_sigma,
            flags=0,
        )

        self._prev_gray = gray
        return flow

    def compute_distortion_map(self, flow: np.ndarray) -> np.ndarray:
        """Convert flow field into a 2D distortion magnitude map for shader effects.

        Returns:
            Normalized distortion intensity map, shape (H, W), values in [0, 1].
        """
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        max_val = np.max(magnitude)
        if max_val > 1e-5:
            magnitude = magnitude / max_val
        return magnitude.astype(np.float32)

    def reset(self) -> None:
        """Reset internal frame memory."""
        self._prev_gray = None
