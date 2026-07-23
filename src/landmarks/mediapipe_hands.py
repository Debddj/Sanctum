"""Dual-hand landmark extraction via MediaPipe Hands.

Wraps mediapipe.solutions.hands for dual-hand mode, returning structured
landmark arrays suitable for downstream normalization and classification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class HandLandmarks:
    """Container for a single hand's 21 landmarks."""

    landmarks: np.ndarray  # shape: (21, 3) — x, y, z in normalized coords
    handedness: str         # "Left" or "Right"
    score: float            # detection confidence


class MediaPipeHands:
    """MediaPipe Hands wrapper for dual-hand landmark extraction."""

    def __init__(
        self,
        max_num_hands: int = 2,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        try:
            import mediapipe as mp
            self._mp = mp
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=max_num_hands,
                model_complexity=model_complexity,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize MediaPipe Hands: {e}")

    def extract(self, frame_bgr: np.ndarray) -> list[HandLandmarks]:
        """Extract hand landmarks from a BGR/RGB frame.

        Args:
            frame_bgr: Input frame, shape (H, W, 3).

        Returns:
            List of HandLandmarks for each detected hand.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        # Ensure C-contiguous RGB array required by MediaPipe C++ engine
        if len(frame_bgr.shape) == 3 and frame_bgr.shape[2] == 3:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = np.ascontiguousarray(frame_bgr)

        results = self._hands.process(frame_rgb)

        if not results.multi_hand_landmarks:
            return []

        hands: list[HandLandmarks] = []
        for hand_lms, hand_info in zip(
            results.multi_hand_landmarks,
            results.multi_handedness,
        ):
            coords = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_lms.landmark],
                dtype=np.float32,
            )
            classification = hand_info.classification[0]
            hands.append(
                HandLandmarks(
                    landmarks=coords,
                    handedness=classification.label,
                    score=classification.score,
                )
            )

        return hands

    def close(self) -> None:
        """Release MediaPipe resources."""
        if hasattr(self, "_hands") and self._hands is not None:
            self._hands.close()

    def __enter__(self) -> "MediaPipeHands":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
