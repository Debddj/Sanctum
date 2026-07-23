"""Dual-hand landmark extraction via MediaPipe Hands.

Wraps MediaPipe for dual-hand mode, supporting both legacy solutions API and
modern Tasks API (HandLandmarker) with automatic fallback.
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
    """MediaPipe Hands wrapper supporting both legacy solutions and modern Tasks API."""

    def __init__(
        self,
        max_num_hands: int = 2,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        import mediapipe as mp

        self._mp = mp
        self._use_tasks = False
        self._hands = None
        self._detector = None

        # 1. Try legacy mp.solutions API
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            try:
                self._hands = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=max_num_hands,
                    model_complexity=model_complexity,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence,
                )
                print("[MediaPipeHands] Initialized using legacy mp.solutions.hands API.")
                return
            except Exception:
                pass

        # 2. Fallback to modern Tasks API (mediapipe 0.10.20+)
        try:
            import urllib.request
            from pathlib import Path
            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision

            model_path = Path("models/hand_landmarker.task")
            model_path.parent.mkdir(parents=True, exist_ok=True)
            if not model_path.exists():
                url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
                print(f"[MediaPipeHands] Downloading hand_landmarker.task model to {model_path}...")
                urllib.request.urlretrieve(url, model_path)

            base_options = mp_tasks.BaseOptions(model_asset_path=str(model_path))
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=max_num_hands,
                min_hand_detection_confidence=min_detection_confidence,
                min_hand_presence_confidence=min_tracking_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self._detector = vision.HandLandmarker.create_from_options(options)
            self._use_tasks = True
            print("[MediaPipeHands] Initialized using modern MediaPipe Tasks API (HandLandmarker).")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize MediaPipe Hands: {e}")

    def extract(self, frame_bgr: np.ndarray) -> list[HandLandmarks]:
        """Extract hand landmarks from a BGR/RGB frame."""
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        if len(frame_bgr.shape) == 3 and frame_bgr.shape[2] == 3:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = np.ascontiguousarray(frame_bgr)

        # Legacy API
        if not self._use_tasks and self._hands is not None:
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

        # Modern Tasks API
        if self._use_tasks and self._detector is not None:
            mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame_rgb)
            results = self._detector.detect(mp_image)

            if not results.hand_landmarks:
                return []

            hands = []
            for hand_lms, hand_info in zip(results.hand_landmarks, results.handedness):
                coords = np.array(
                    [[lm.x, lm.y, lm.z] for lm in hand_lms],
                    dtype=np.float32,
                )
                classification = hand_info[0]
                hands.append(
                    HandLandmarks(
                        landmarks=coords,
                        handedness=classification.category_name,
                        score=classification.score,
                    )
                )
            return hands

        return []

    def close(self) -> None:
        """Release MediaPipe resources."""
        if hasattr(self, "_hands") and self._hands is not None:
            self._hands.close()
        if hasattr(self, "_detector") and self._detector is not None:
            self._detector.close()

    def __enter__(self) -> "MediaPipeHands":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
