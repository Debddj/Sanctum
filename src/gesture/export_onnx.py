"""Export trained gesture classifier from PyTorch to ONNX format.

The ONNX model is the input to TensorRT engine building in Phase 7.
"""

from __future__ import annotations

# TODO: Implement ONNX export
# - Load trained .pt checkpoint
# - Create dummy input matching (1, window_size, feature_dim)
# - torch.onnx.export with dynamic_axes for batch dimension
# - Verify with onnxruntime
# - Save to models/checkpoints/gesture_classifier.onnx

if __name__ == "__main__":
    raise NotImplementedError("ONNX export not yet implemented — see Phase 7")
