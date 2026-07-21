"""End-to-end pipeline test.

Runs against recorded/synthetic landmark sequences so it's reproducible in CI
without requiring a connected physical camera.
"""

from __future__ import annotations

import asyncio
import numpy as np

from src.landmarks.normalization import normalize_landmarks
from src.landmarks.sequence_windowing import SequenceWindow
from src.server.pipeline_orchestrator import PipelineOrchestrator
from src.server.ws_protocol import LandmarkMessage, GestureMessage


class TestPipelineE2E:
    """End-to-end pipeline integration tests."""

    def test_landmark_windowing_and_normalization_pipeline(self) -> None:
        """Verify frame landmarks pass through normalization and windowing correctly."""
        window = SequenceWindow(window_size=30, feature_dim=63)
        assert not window.is_ready

        for i in range(30):
            raw_lms = np.random.rand(21, 3).astype(np.float32)
            norm_lms = normalize_landmarks(raw_lms, method="wrist_relative")
            window.push(norm_lms)

        assert window.is_ready
        seq = window.get_sequence()
        assert seq is not None
        assert seq.shape == (30, 63)

    def test_orchestrator_lifecycle(self) -> None:
        """Verify PipelineOrchestrator starts and stops cleanly."""
        async def _test() -> None:
            orchestrator = PipelineOrchestrator()
            assert not orchestrator.is_running

            orchestrator.start()
            assert orchestrator.is_running
            await asyncio.sleep(0.1)

            orchestrator.stop()
            assert not orchestrator.is_running

        asyncio.run(_test())

    def test_message_serialization(self) -> None:
        """Verify LandmarkMessage and GestureMessage serialize to valid JSON dicts."""
        lm_msg = LandmarkMessage(
            frame_id=1,
            timestamp_ms=1000.0,
            hands=[{"handedness": "Right", "score": 0.98, "landmarks": [[0, 0, 0]]}],
        )
        lm_dict = lm_msg.model_dump()
        assert lm_dict["type"] == "landmarks"
        assert lm_dict["frame_id"] == 1

        g_msg = GestureMessage(
            frame_id=1,
            gesture_class="sling_ring",
            confidence=0.92,
            action="spawn_portal",
        )
        g_dict = g_msg.model_dump()
        assert g_dict["type"] == "gesture"
        assert g_dict["action"] == "spawn_portal"
