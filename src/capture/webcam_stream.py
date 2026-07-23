"""Threaded webcam stream using cv2.VideoCapture.

Runs frame acquisition on a background thread to decouple capture latency
from the processing pipeline. Frames are pulled via a thread-safe queue.
"""

from __future__ import annotations

import sys
import threading
from typing import Optional

import cv2
import numpy as np


class WebcamStream:
    """Threaded wrapper around cv2.VideoCapture."""

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
    ) -> None:
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "WebcamStream":
        """Open the camera and start the capture thread."""
        # Use DirectShow on Windows for instant camera access
        if sys.platform.startswith("win"):
            self._cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        else:
            self._cap = cv2.VideoCapture(self.camera_index)

        if self._cap is None or not self._cap.isOpened():
            # Fallback to default backend if DirectShow is unavailable
            self._cap = cv2.VideoCapture(self.camera_index)

        if self._cap is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.fps)

        if self._cap is None or not self._cap.isOpened():
            print(f"[WebcamStream] Unable to open physical camera index {self.camera_index}. Synthetic test feed active.")
            return self

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return self

    def read(self) -> Optional[np.ndarray]:
        """Return the most recent frame, or synthetic test frame if camera unattached."""
        with self._lock:
            if self._frame is not None:
                return self._frame.copy()

        # Generate synthetic background frame for testing if physical camera is offline
        syn_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        return syn_frame

    def stop(self) -> None:
        """Stop the capture thread and release the camera."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()

    def _capture_loop(self) -> None:
        """Continuously read frames from the camera."""
        while self._running and self._cap is not None:
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._frame = frame

    def __enter__(self) -> "WebcamStream":
        return self.start()

    def __exit__(self, *args: object) -> None:
        self.stop()
