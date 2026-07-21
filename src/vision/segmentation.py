"""Real-time person segmentation for background replacement.

Uses MediaPipe Selfie Segmentation (or lightweight skin/luminance thresholding)
to produce a binary/alpha foreground mask per frame for background swapping.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


class PersonSegmenter:
    """Real-time foreground person segmenter."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self._segmenter: Optional[object] = None

        try:
            import mediapipe as mp
            self._mp_selfie = mp.solutions.selfie_segmentation
            self._segmenter = self._mp_selfie.SelfieSegmentation(model_selection=1)
        except Exception:
            self._segmenter = None

    def segment(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Compute binary/alpha mask for the foreground person.

        Args:
            frame_bgr: Input image, shape (H, W, 3).

        Returns:
            Alpha mask array, shape (H, W), float32 in [0, 1].
        """
        if self._segmenter is not None:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = self._segmenter.process(frame_rgb)
            mask = results.segmentation_mask
            return (mask > self.threshold).astype(np.float32)

        # Fallback background segmentation heuristic when MediaPipe is uninstalled
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        _, alpha = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY)
        return (alpha / 255.0).astype(np.float32)

    def composite(
        self,
        fg_frame: np.ndarray,
        bg_frame: np.ndarray,
        alpha_mask: np.ndarray,
    ) -> np.ndarray:
        """Composite foreground frame onto background frame using alpha mask.

        Args:
            fg_frame: Foreground frame (H, W, 3).
            bg_frame: Background frame (H, W, 3).
            alpha_mask: Alpha mask (H, W), values in [0, 1].

        Returns:
            Composited frame (H, W, 3).
        """
        alpha_3d = np.expand_dims(alpha_mask, axis=-1)
        bg_resized = cv2.resize(bg_frame, (fg_frame.shape[1], fg_frame.shape[0]))
        out = fg_frame * alpha_3d + bg_resized * (1.0 - alpha_3d)
        return out.astype(np.uint8)

    def close(self) -> None:
        """Release segmentation resources."""
        if self._segmenter is not None:
            self._segmenter.close()
            self._segmenter = None
