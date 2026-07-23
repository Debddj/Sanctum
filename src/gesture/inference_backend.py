"""Inference backend abstraction layer for gesture classification.

Provides a unified interface (BaseInferenceBackend) implemented by PyTorchBackend (CPU/GPU PyTorch)
and TensorRTBackend (NVIDIA TensorRT GPU execution engine) with real-time 3D kinematic gesture analysis.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple

import numpy as np

GESTURE_CLASSES = ["sling_ring", "mudra_hold", "pinch_pull", "open_palm"]


def classify_hand_kinematics(window: np.ndarray) -> Tuple[str, float]:
    """Analyze real 3D hand geometry and motion trajectory over the 30-frame sequence."""
    arr = np.ascontiguousarray(window, dtype=np.float32)
    if arr.ndim == 3:
        seq = arr[0]  # (30, 63)
    else:
        seq = arr  # (30, 63)

    # 1. Inspect last frame 21 3D landmarks
    last_frame = seq[-1].reshape(21, 3)
    wrist = last_frame[0]
    thumb_tip = last_frame[4]
    index_tip = last_frame[8]
    middle_tip = last_frame[12]
    ring_tip = last_frame[16]
    pinky_tip = last_frame[20]

    # Distances between fingertips
    d_thumb_index = float(np.linalg.norm(thumb_tip - index_tip))
    d_thumb_ring = float(np.linalg.norm(thumb_tip - ring_tip))

    # Extensions from wrist
    ext_index = float(np.linalg.norm(index_tip - wrist))
    ext_middle = float(np.linalg.norm(middle_tip - wrist))
    ext_ring = float(np.linalg.norm(ring_tip - wrist))
    ext_pinky = float(np.linalg.norm(pinky_tip - wrist))

    # 2. Open Palm: ALL 4 fingers (index, middle, ring, pinky) extended wide
    if ext_index > 0.18 and ext_middle > 0.18 and ext_ring > 0.18 and ext_pinky > 0.18 and d_thumb_index > 0.10:
        return "open_palm", 0.96

    # 3. Sling Ring: Index finger extended high while other fingers are curled OR circular index motion
    index_pts = seq[:, 8 * 3 : 8 * 3 + 3]  # (30, 3)
    x_range = float(np.ptp(index_pts[:, 0]))
    y_range = float(np.ptp(index_pts[:, 1]))
    if (ext_index > 0.18 and ext_middle < 0.18 and ext_ring < 0.18) or (x_range > 0.08 and y_range > 0.08):
        return "sling_ring", 0.95

    # 4. Pinch Pull: Thumb & Index pinched close together (<0.08)
    if d_thumb_index < 0.08:
        return "pinch_pull", 0.94

    # 5. Mudra Hold: Thumb tip touches Ring tip or Index tip while other fingers extended
    if d_thumb_ring < 0.10 or (d_thumb_index < 0.10 and ext_middle > 0.15):
        return "mudra_hold", 0.95

    # Return none if pose doesn't match a distinct gesture (do NOT default to open_palm)
    return "none", 0.0


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
    """PyTorch inference backend (CPU/CUDA) with kinematic geometry fusion."""

    def __init__(self, checkpoint_path: str | Path = "models/checkpoints/gesture_classifier.pt") -> None:
        import torch
        from src.gesture.model import GestureCNN1D, GestureLSTM

        self.cp_path = Path(checkpoint_path)
        self.device = torch.device("cpu")

        if not self.cp_path.exists():
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
        # 1. Run 3D kinematic hand geometry and motion trajectory analysis
        k_class, k_conf = classify_hand_kinematics(window)
        if k_class != "none":
            return k_class, k_conf

        # 2. PyTorch neural network forward pass fallback
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
        k_class, k_conf = classify_hand_kinematics(window)
        if k_class != "none":
            return k_class, k_conf

        if self.fallback_backend is not None:
            return self.fallback_backend.predict(window)

        logits = self._trt_engine.predict(window)
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
