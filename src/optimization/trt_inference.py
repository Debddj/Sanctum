"""TensorRT runtime wrapper for gesture classifier inference.

Drop-in replacement for the PyTorch inference path, using a serialized
TensorRT engine for lower latency.
"""

from __future__ import annotations

# TODO: Implement TensorRT inference
# - Load serialized .engine file
# - Allocate CUDA device/host memory
# - Run inference with proper input/output binding
# - Return class predictions matching PyTorch model interface

if __name__ == "__main__":
    raise NotImplementedError("TensorRT inference not yet implemented — see Phase 7")
