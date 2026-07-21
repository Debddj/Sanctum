"""Build a TensorRT engine from an ONNX gesture classifier model.

Reads configuration from configs/model/trt_build.yaml. Supports FP16 and INT8
precision modes with optional calibration.
"""

from __future__ import annotations

# TODO: Implement TensorRT engine build
# - Parse trt_build.yaml
# - Load ONNX model
# - Configure TensorRT builder (precision, workspace, dynamic shapes)
# - INT8 calibration if precision=int8
# - Serialize engine to models/trt_engines/
# - Log build statistics

if __name__ == "__main__":
    raise NotImplementedError("TensorRT engine build not yet implemented — see Phase 7")
