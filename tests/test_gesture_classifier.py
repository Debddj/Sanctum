"""Tests for gesture classifier models."""

from __future__ import annotations

import pytest
import torch

from src.gesture.model import GestureCNN1D, GestureLSTM


class TestGestureCNN1D:
    """Test 1D CNN gesture classifier."""

    def test_output_shape(self) -> None:
        """Output shape should be (batch, num_classes)."""
        model = GestureCNN1D(input_features=63, num_classes=4)
        x = torch.randn(8, 30, 63)
        out = model(x)
        assert out.shape == (8, 4)

    def test_single_sample(self) -> None:
        """Should handle batch size of 1."""
        model = GestureCNN1D()
        x = torch.randn(1, 30, 63)
        out = model(x)
        assert out.shape == (1, 4)


class TestGestureLSTM:
    """Test LSTM gesture classifier."""

    def test_output_shape(self) -> None:
        """Output shape should be (batch, num_classes)."""
        model = GestureLSTM(input_features=63, num_classes=4)
        x = torch.randn(8, 30, 63)
        out = model(x)
        assert out.shape == (8, 4)

    def test_bidirectional_vs_unidirectional(self) -> None:
        """Both modes should produce correct output shapes."""
        for bidir in [True, False]:
            model = GestureLSTM(bidirectional=bidir)
            x = torch.randn(4, 30, 63)
            out = model(x)
            assert out.shape == (4, 4)
