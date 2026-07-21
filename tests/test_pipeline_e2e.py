"""End-to-end pipeline test.

Runs against a recorded session rather than a live camera so it's
reproducible in CI. Covers: webcam → landmarks → classifier → dispatch → render.
"""

from __future__ import annotations

import pytest


class TestPipelineE2E:
    """End-to-end pipeline integration tests."""

    @pytest.mark.skip(reason="Not yet implemented — see Phase 8")
    def test_recorded_session_roundtrip(self) -> None:
        """Pipeline should process a recorded session end-to-end."""
        pass

    @pytest.mark.skip(reason="Not yet implemented — see Phase 8")
    def test_gesture_dispatch(self) -> None:
        """Recognized gesture should trigger correct VFX dispatch."""
        pass

    @pytest.mark.skip(reason="Not yet implemented — see Phase 8")
    def test_websocket_output(self) -> None:
        """WebSocket should emit landmark + effect state messages."""
        pass
