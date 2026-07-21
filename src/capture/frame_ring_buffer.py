"""Fixed-length ring buffer for frame storage, backing time-reversal playback.

Stores the last N seconds of frames in a pre-allocated numpy array to avoid
per-frame allocation. Supports forward iteration and reverse playback.
"""

from __future__ import annotations

from typing import Iterator, Optional

import numpy as np


class FrameRingBuffer:
    """Circular buffer for video frames."""

    def __init__(self, max_seconds: float = 5.0, fps: int = 30) -> None:
        self.capacity = int(max_seconds * fps)
        self._buffer: list[Optional[np.ndarray]] = [None] * self.capacity
        self._write_idx = 0
        self._count = 0

    @property
    def is_full(self) -> bool:
        return self._count >= self.capacity

    def __len__(self) -> int:
        return self._count

    def push(self, frame: np.ndarray) -> None:
        """Add a frame to the buffer, overwriting the oldest if full."""
        self._buffer[self._write_idx] = frame.copy()
        self._write_idx = (self._write_idx + 1) % self.capacity
        self._count = min(self._count + 1, self.capacity)

    def get_reversed(self) -> Iterator[np.ndarray]:
        """Yield frames in reverse chronological order (newest first)."""
        if self._count == 0:
            return

        idx = (self._write_idx - 1) % self.capacity
        for _ in range(self._count):
            frame = self._buffer[idx]
            if frame is not None:
                yield frame
            idx = (idx - 1) % self.capacity

    def get_forward(self) -> Iterator[np.ndarray]:
        """Yield frames in chronological order (oldest first)."""
        if self._count == 0:
            return

        start = (self._write_idx - self._count) % self.capacity
        for i in range(self._count):
            idx = (start + i) % self.capacity
            frame = self._buffer[idx]
            if frame is not None:
                yield frame

    def clear(self) -> None:
        """Reset the buffer."""
        self._buffer = [None] * self.capacity
        self._write_idx = 0
        self._count = 0
