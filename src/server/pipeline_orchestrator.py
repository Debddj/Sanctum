"""Pipeline orchestrator — stage sequencing and effect dispatch.

Coordinates the end-to-end processing pipeline:
capture → landmarks → normalization → sequence window → gesture classification → effect dispatch → WebSocket broadcast.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Optional, Set

import numpy as np
import yaml
from fastapi import WebSocket

from src.capture.webcam_stream import WebcamStream
from src.landmarks.mediapipe_hands import MediaPipeHands
from src.landmarks.normalization import normalize_landmarks
from src.landmarks.sequence_windowing import SequenceWindow
from src.profiling.latency_tracer import LatencyTracer
from src.server.ws_protocol import LandmarkMessage, GestureMessage, EffectStateMessage, MessageType


class PipelineOrchestrator:
    """Manages stage sequencing, frame processing, and WebSocket client broadcasts."""

    def __init__(
        self,
        config_path: str | Path = "configs/pipeline.yaml",
        gestures_path: str | Path = "configs/gestures.yaml",
    ) -> None:
        self.config_path = Path(config_path)
        self.gestures_path = Path(gestures_path)

        self.pipeline_config: dict[str, Any] = {}
        self.gestures_config: dict[str, Any] = {}

        self.active_websockets: Set[WebSocket] = set()
        self.is_running = False
        self._loop_task: Optional[asyncio.Task] = None

        self.tracer = LatencyTracer()

        self.webcam_stream: Optional[WebcamStream] = None
        self.hands_extractor: Optional[MediaPipeHands] = None
        self.sequence_window: Optional[SequenceWindow] = None

        self._load_configs()

    def _load_configs(self) -> None:
        """Load YAML configuration files if available."""
        if self.config_path.exists():
            with open(self.config_path) as f:
                self.pipeline_config = yaml.safe_load(f) or {}

        if self.gestures_path.exists():
            with open(self.gestures_path) as f:
                self.gestures_config = yaml.safe_load(f) or {}

    async def register_websocket(self, websocket: WebSocket) -> None:
        """Register a connected WebSocket client."""
        self.active_websockets.add(websocket)

    async def unregister_websocket(self, websocket: WebSocket) -> None:
        """Unregister a disconnected WebSocket client."""
        self.active_websockets.discard(websocket)

    async def broadcast_json(self, data: dict[str, Any]) -> None:
        """Broadcast a JSON message to all connected WebSocket clients."""
        if not self.active_websockets:
            return

        disconnected = set()
        for ws in self.active_websockets:
            try:
                await ws.send_json(data)
            except Exception:
                disconnected.add(ws)

        for ws in disconnected:
            self.active_websockets.discard(ws)

    def start(self) -> None:
        """Initialize pipeline components and start the background processing loop."""
        if self.is_running:
            return

        self.is_running = True
        self._init_stages()
        self._loop_task = asyncio.create_task(self._processing_loop())

    def _init_stages(self) -> None:
        """Initialize enabled stages based on configuration."""
        stages = self.pipeline_config.get("stages", {})

        # Capture stage
        if stages.get("capture", {}).get("enabled", True):
            try:
                self.webcam_stream = WebcamStream(camera_index=0).start()
            except Exception as e:
                print(f"[Orchestrator] Webcam initialization skipped/failed: {e}")
                self.webcam_stream = None

        # Landmarks stage
        if stages.get("landmarks", {}).get("enabled", True):
            try:
                self.hands_extractor = MediaPipeHands()
            except Exception as e:
                print(f"[Orchestrator] MediaPipe initialization skipped/failed: {e}")
                self.hands_extractor = None

        # Sequence window
        if stages.get("classifier", {}).get("enabled", True):
            window_size = stages.get("classifier", {}).get("window_size", 30)
            self.sequence_window = SequenceWindow(window_size=window_size)

    async def _processing_loop(self) -> None:
        """Main processing loop executing per-frame pipeline stages."""
        frame_id = 0
        target_fps = self.pipeline_config.get("pipeline", {}).get("target_fps", 30)
        frame_delay = 1.0 / target_fps

        while self.is_running:
            start_time = asyncio.get_event_loop().time()
            frame_id += 1
            self.tracer.next_frame()

            # 1. Capture stage
            frame = None
            if self.webcam_stream is not None:
                with self.tracer.trace("capture"):
                    frame = self.webcam_stream.read()

            # 2. Landmark & Normalization stages
            detected_hands: list[dict[str, Any]] = []
            if frame is not None and self.hands_extractor is not None:
                with self.tracer.trace("landmarks"):
                    # Convert BGR (OpenCV default) to RGB for MediaPipe
                    frame_rgb = frame[:, :, ::-1] if len(frame.shape) == 3 else frame
                    raw_hands = self.hands_extractor.extract(frame_rgb)

                with self.tracer.trace("normalization"):
                    for hand in raw_hands:
                        norm_lms = normalize_landmarks(hand.landmarks, method="wrist_relative")
                        detected_hands.append({
                            "handedness": hand.handedness,
                            "score": float(hand.score),
                            "landmarks": norm_lms.tolist(),
                        })

                        if self.sequence_window is not None:
                            self.sequence_window.push(norm_lms)

            # Broadcast landmark frame message
            msg = LandmarkMessage(
                frame_id=frame_id,
                timestamp_ms=start_time * 1000.0,
                hands=detected_hands,
            )
            await self.broadcast_json(msg.model_dump())

            # 3. Rate limiting for target FPS
            elapsed = asyncio.get_event_loop().time() - start_time
            sleep_time = max(0.0, frame_delay - elapsed)
            await asyncio.sleep(sleep_time)

    def stop(self) -> None:
        """Stop the background processing loop and release resources."""
        self.is_running = False
        if self._loop_task is not None:
            self._loop_task.cancel()
            self._loop_task = None

        if self.webcam_stream is not None:
            self.webcam_stream.stop()
            self.webcam_stream = None

        if self.hands_extractor is not None:
            self.hands_extractor.close()
            self.hands_extractor = None
