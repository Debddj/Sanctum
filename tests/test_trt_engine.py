"""Tests for TensorRT engine build and inference.

These tests require a CUDA-capable GPU and TensorRT installed.
Mark as skipped if TensorRT is not available.
"""

from __future__ import annotations

import pytest

# Skip entire module if TensorRT is not installed
pytest.importorskip("tensorrt", reason="TensorRT not installed")


class TestTRTEngine:
    """Test TensorRT engine build and inference."""

    @pytest.mark.skip(reason="Not yet implemented — see Phase 7")
    def test_engine_build(self) -> None:
        """TensorRT engine should build from ONNX model."""
        pass

    @pytest.mark.skip(reason="Not yet implemented — see Phase 7")
    def test_inference_matches_pytorch(self) -> None:
        """TRT inference output should match PyTorch within tolerance."""
        pass

    @pytest.mark.skip(reason="Not yet implemented — see Phase 7")
    def test_fp16_precision(self) -> None:
        """FP16 engine should produce acceptable accuracy."""
        pass
