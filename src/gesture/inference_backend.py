"""Inference backend abstraction layer for gesture classification.

Provides a unified interface (BaseInferenceBackend) implemented by PyTorchBackend (CPU/GPU PyTorch)
and TensorRTBackend (NVIDIA TensorRT GPU execution engine).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple

import numpy as np

GESTURE_CLASSES = ["sling_ring", "mudra_hold", "pinch_pull", "open_palm"]


class BaseInferenceBackend(ABC):
    """Abstract base class for gesture inference backends."""

    @abstractmethod
    def predict(self, window: np.ndarray) -> Tuple[str, float]:
        """Predict gesture class and confidence from landmark sequence window.

        Args:
            window: Input array of shape (30, 63) or (1, 30, 63).

        Returns:
            Tuple of (predicted_gesture_class, confidence_score).
        """
        pass


class PyTorchBackend(BaseInferenceBackend):
    """PyTorch inference backend (CPU/CUDA)."""

    def __init__(self, checkpoint_path: str | Path = "models/checkpoints/gesture_classifier.pt") -> None:
        import torch
        from src.gesture.model import GestureCNN1D, GestureLSTM

        self.cp_path = Path(checkpoint_path)
        self.device = torch.device("cpu")

        if not self.cp_path.exists():
            # Fallback to model architecture without weights if checkpoint missing
            print(f"[PyTorchBackend] Checkpoint missing at {self.cp_path}, using randomly initialized model.")
            self.model = GestureCNN1D(input_features=63, num_classes=4).to(self.device)
            self.model.eval()
            return

        checkpoint = torch.load(self.cp_path, map_location=self.device)
        arch = checkpoint.get("architecture", "cnn_1d")

        if arch == "lstm":
            self.model = GestureLSTM(input_features=63, num_classes=4).to(self.device)
        else:
            self.model = GestureCNN1D(input_features=63, num_classes=4).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.torch = torch

    def predict(self, window: np.ndarray) -> Tuple[str, float]:
        arr = np.ascontiguousarray(window, dtype=np.float32)
        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=0)  # (1, 30, 63)

        x_tensor = self.torch.from_numpy(arr).to(self.device)
        with self.torch.no_grad():
            logits = self.model(x_tensor)
            probs = self.torch.softmax(logits, dim=1).squeeze(0).numpy()

        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
        return GESTURE_CLASSES[idx], confidence


class TensorRTBackend(BaseInferenceBackend):
    """TensorRT GPU inference backend wrapper."""

    def __init__(self, engine_path: str | Path = "models/trt_engines/gesture_classifier.engine") -> None:
        self.engine_path = Path(engine_path)
        self.fallback_backend: BaseInferenceBackend | None = None
        self._trt_engine = None

        if not self.engine_path.exists():
            print(f"[TensorRTBackend] Engine file missing at {self.engine_path}. Falling back to PyTorchBackend.")
            self.fallback_backend = PyTorchBackend()
            return

        try:
            from src.optimization.trt_inference import TensorRTInference
            self._trt_engine = TensorRTInference(self.engine_path)
            if self._trt_engine.context is None:
                raise RuntimeError("TensorRT context creation returned None.")
        except Exception as e:
            print(f"[TensorRTBackend] Execution initialization skipped/failed: {e}. Falling back to PyTorchBackend.")
            self.fallback_backend = PyTorchBackend()

    def predict(self, window: np.ndarray) -> Tuple[str, float]:
        if self.fallback_backend is not None:
            return self.fallback_backend.predict(window)

        logits = self._trt_engine.predict(window)
        # Compute softmax over logits
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        idx = int(np.argmax(probs[0]))
        confidence = float(probs[0, idx])
        return GESTURE_CLASSES[idx], confidence


def get_inference_backend(
    backend_type: str = "pytorch",
    checkpoint_path: str | Path = "models/checkpoints/gesture_classifier.pt",
    engine_path: str | Path = "models/trt_engines/gesture_classifier.engine",
) -> BaseInferenceBackend:
    """Factory function for instantiating inference backends based on config."""
    b_type = backend_type.lower()
    if b_type == "tensorrt":
        return TensorRTBackend(engine_path=engine_path)
    return PyTorchBackend(checkpoint_path=checkpoint_path)
