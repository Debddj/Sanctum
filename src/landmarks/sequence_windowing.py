"""Sliding window over landmark sequences for the gesture classifier.

Maintains a fixed-length window of normalized landmark frames, suitable
for feeding into the 1D-CNN or LSTM classifier.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np


class SequenceWindow:
    """Fixed-length sliding window over landmark frames."""

    def __init__(self, window_size: int = 30, feature_dim: int = 63) -> None:
        """
        Args:
            window_size: Number of frames in the window.
            feature_dim: Flattened feature dimension per frame (21 landmarks × 3 coords = 63).
        """
        self.window_size = window_size
        self.feature_dim = feature_dim
        self._buffer: deque[np.ndarray] = deque(maxlen=window_size)

    @property
    def is_ready(self) -> bool:
        """Whether the window has accumulated enough frames for inference."""
        return len(self._buffer) == self.window_size

    def push(self, landmarks: np.ndarray) -> None:
        """Add a frame's landmarks to the window.

        Args:
            landmarks: Normalized landmarks, shape (21, 3). Will be flattened to (63,).
        """
        flat = landmarks.flatten().astype(np.float32)
        assert flat.shape == (self.feature_dim,), (
            f"Expected feature_dim={self.feature_dim}, got {flat.shape[0]}"
        )
        self._buffer.append(flat)

    def get_sequence(self) -> Optional[np.ndarray]:
        """Return the current window as a numpy array.

        Returns:
            Array of shape (window_size, feature_dim), or None if not ready.
        """
        if not self.is_ready:
            return None
        return np.stack(list(self._buffer), axis=0)

    def clear(self) -> None:
        """Reset the window."""
        self._buffer.clear()
