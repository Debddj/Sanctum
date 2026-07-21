"""Time-reversal effect controller.

On gesture trigger (pinch_pull), switches pipeline output from live webcam feed
to ring buffer reverse playback, computing optical flow distortion maps per frame
to generate a stylized "reality unraveling" cosmetic effect.
"""

from __future__ import annotations

from typing import Iterator, Optional, Tuple

import numpy as np

from src.capture.frame_ring_buffer import FrameRingBuffer
from src.vision.optical_flow import FarnebackOpticalFlow


class TimeReversalEffect:
    """Manages rewind state, reverse frame playback, and optical flow distortion."""

    def __init__(self, ring_buffer: FrameRingBuffer) -> None:
        self.ring_buffer = ring_buffer
        self.flow_computer = FarnebackOpticalFlow()
        self.is_active = False
        self._reverse_gen: Optional[Iterator[np.ndarray]] = None
        self._frames_yielded = 0

    def activate(self) -> bool:
        """Activate the time-reversal rewind effect."""
        if len(self.ring_buffer) == 0:
            return False

        self.is_active = True
        self.flow_computer.reset()
        self._reverse_gen = self.ring_buffer.get_reversed()
        self._frames_yielded = 0
        return True

    def get_next_frame(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Retrieve the next reversed frame and its optical flow distortion map.

        Returns:
            Tuple of (frame_bgr, distortion_map) or None if rewind finished.
        """
        if not self.is_active or self._reverse_gen is None:
            return None

        try:
            frame = next(self._reverse_gen)
            self._frames_yielded += 1

            # Compute cosmetic optical flow vector field
            flow = self.flow_computer.compute(frame)
            distortion_map = self.flow_computer.compute_distortion_map(flow)

            return frame, distortion_map
        except StopIteration:
            self.deactivate()
            return None

    def deactivate(self) -> None:
        """Deactivate time-reversal and return to live feed mode."""
        self.is_active = False
        self._reverse_gen = None
        self.flow_computer.reset()
