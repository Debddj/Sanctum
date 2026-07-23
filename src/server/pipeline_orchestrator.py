"""Pipeline orchestrator — stage sequencing and effect dispatch.

Coordinates the end-to-end processing pipeline:
capture → landmarks → normalization → sequence window → gesture classification → effect dispatch → WebSocket broadcast.
"""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path
from typing import Any, Optional, Set

import cv2
import numpy as np
import yaml
from fastapi import WebSocket

from src.capture.webcam_stream import WebcamStream
from src.gesture.inference_backend import BaseInferenceBackend, get_inference_backend
from src.landmarks.mediapipe_hands import MediaPipeHands
from src.landmarks.normalization import normalize_landmarks
from src.landmarks.sequence_windowing import SequenceWindow
from src.profiling.latency_tracer import LatencyTracer
from src.server.ws_protocol import LandmarkMessage, GestureMessage, EffectStateMessage


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
        self.classifier_backend: Optional[BaseInferenceBackend] = None

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

        # 1. Capture stage
        if stages.get("capture", {}).get("enabled", True):
            try:
                self.webcam_stream = WebcamStream(camera_index=0).start()
                print("[Orchestrator] Webcam opened successfully.")
            except Exception as e:
                print(f"[Orchestrator] Webcam initialization skipped/failed: {e}")
                self.webcam_stream = None

        # 2. Landmarks stage
        if stages.get("landmarks", {}).get("enabled", True):
            try:
                self.hands_extractor = MediaPipeHands()
                print("[Orchestrator] MediaPipe Hands initialized.")
            except Exception as e:
                print(f"[Orchestrator] MediaPipe initialization skipped/failed: {e}")
                self.hands_extractor = None

        # 3. Classifier & Sequence Window stage
        if stages.get("classifier", {}).get("enabled", True):
            window_size = stages.get("classifier", {}).get("window_size", 30)
            backend_type = stages.get("classifier", {}).get("backend", "pytorch")
            self.sequence_window = SequenceWindow(window_size=window_size)
            self.classifier_backend = get_inference_backend(backend_type=backend_type)

    async def _processing_loop(self) -> None:
        """Main processing loop executing per-frame pipeline stages."""
        frame_id = 0
        target_fps = self.pipeline_config.get("pipeline", {}).get("target_fps", 30)
        frame_delay = 1.0 / target_fps
        log_interval = 90  # Log hand detection info every N frames

        while self.is_running:
            start_time = asyncio.get_event_loop().time()
            frame_id += 1
            self.tracer.next_frame()

            # 1. Capture stage
            frame = None
            image_b64 = None
            if self.webcam_stream is not None:
                with self.tracer.trace("capture"):
                    frame = self.webcam_stream.read()

                if frame is not None and np.any(frame > 0):
                    # Encode camera frame as JPEG base64 for frontend streaming
                    h, w = frame.shape[:2]
                    display_w = min(640, w)
                    display_h = int(display_w * h / w)
                    display_frame = cv2.resize(frame, (display_w, display_h))
                    # Flip horizontally for selfie-mirror view
                    display_frame = cv2.flip(display_frame, 1)
                    ret, jpeg_buf = cv2.imencode(".jpg", display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                    if ret:
                        b64_str = base64.b64encode(jpeg_buf).decode("ascii")
                        image_b64 = f"data:image/jpeg;base64,{b64_str}"

            # 2. Landmark & Normalization stages
            detected_hands: list[dict[str, Any]] = []
            raw_hands = []
            if frame is not None and self.hands_extractor is not None:
                with self.tracer.trace("landmarks"):
                    frame_rgb = frame[:, :, ::-1] if len(frame.shape) == 3 else frame
                    raw_hands = self.hands_extractor.extract(frame_rgb)

                if frame_id % log_interval == 1:
                    print(f"[Frame {frame_id}] Hands detected: {len(raw_hands)}")

                with self.tracer.trace("normalization"):
                    for hand in raw_hands:
                        # Wrist-relative normalization for classifier input
                        norm_lms = normalize_landmarks(hand.landmarks, method="wrist_relative")

                        # Send RAW MediaPipe landmarks (0-1 range) to frontend for skeleton display
                        # Send normalized landmarks separately for classifier consumption
                        detected_hands.append({
                            "handedness": hand.handedness,
                            "score": float(hand.score),
                            "landmarks": hand.landmarks.tolist(),  # RAW (0-1) for display
                        })

                        if self.sequence_window is not None:
                            self.sequence_window.push(norm_lms)

            # Broadcast landmark frame message with camera video payload
            msg = LandmarkMessage(
                frame_id=frame_id,
                timestamp_ms=start_time * 1000.0,
                hands=detected_hands,
                image_b64=image_b64,
            )
            await self.broadcast_json(msg.model_dump())

            # 3. Live Gesture Classification & Effect Dispatch
            if self.sequence_window is not None and self.sequence_window.is_ready:
                with self.tracer.trace("classifier"):
                    window = self.sequence_window.get_sequence()
                    if window is not None and self.classifier_backend is not None:
                        pred_class, confidence = self.classifier_backend.predict(window)

                        gesture_cfg = self.gestures_config.get("gestures", {}).get(pred_class, {})
                        min_conf = gesture_cfg.get("min_confidence", 0.8)

                        if confidence >= min_conf:
                            dispatch_action = gesture_cfg.get("dispatch_action", "")

                            # Compute hand centroid for VFX positioning (from raw landmarks)
                            centroid = [0.5, 0.5]
                            if raw_hands:
                                wrist = raw_hands[0].landmarks[0]
                                middle_mcp = raw_hands[0].landmarks[9]
                                centroid = [
                                    float((wrist[0] + middle_mcp[0]) / 2.0),
                                    float((wrist[1] + middle_mcp[1]) / 2.0),
                                ]

                            # Broadcast gesture detection message
                            with self.tracer.trace("dispatch"):
                                g_msg = GestureMessage(
                                    frame_id=frame_id,
                                    gesture_class=pred_class,
                                    confidence=confidence,
                                    action=dispatch_action,
                                )
                                await self.broadcast_json(g_msg.model_dump())

                                eff_msg = EffectStateMessage(
                                    frame_id=frame_id,
                                    portal={
                                        "active": (dispatch_action == "spawn_portal"),
                                        "center": centroid,
                                        "radius": 0.35,
                                        "intensity": float(confidence),
                                    },
                                    active_effects=[{"action": dispatch_action}] if dispatch_action else [],
                                )
                                await self.broadcast_json(eff_msg.model_dump())

            # Rate limiting for target FPS
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
