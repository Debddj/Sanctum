"""Tests for landmark extraction and normalization."""

from __future__ import annotations

import numpy as np
import pytest

from src.landmarks.normalization import normalize_landmarks


class TestNormalization:
    """Test landmark normalization."""

    def test_wrist_relative_centers_on_wrist(self) -> None:
        """Normalized landmarks should have wrist at origin."""
        landmarks = np.random.rand(21, 3).astype(np.float32)
        normalized = normalize_landmarks(landmarks, method="wrist_relative")
        np.testing.assert_allclose(normalized[0], [0, 0, 0], atol=1e-6)

    def test_scale_invariance(self) -> None:
        """Scaling input landmarks should not change normalized output."""
        landmarks = np.random.rand(21, 3).astype(np.float32)
        scaled = landmarks * 2.5

        norm1 = normalize_landmarks(landmarks, scale_invariant=True)
        norm2 = normalize_landmarks(scaled, scale_invariant=True)

        np.testing.assert_allclose(norm1, norm2, atol=1e-5)

    def test_palm_center_method(self) -> None:
        """Palm center normalization should center between wrist and MCP."""
        landmarks = np.random.rand(21, 3).astype(np.float32)
        normalized = normalize_landmarks(landmarks, method="palm_center")
        assert normalized.shape == (21, 3)

    def test_invalid_method_raises(self) -> None:
        """Unknown normalization method should raise ValueError."""
        landmarks = np.random.rand(21, 3).astype(np.float32)
        with pytest.raises(ValueError, match="Unknown normalization method"):
            normalize_landmarks(landmarks, method="invalid")
