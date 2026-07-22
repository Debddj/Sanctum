"""TensorRT runtime wrapper for gesture classifier inference.

Drop-in replacement for the PyTorch inference path, using serialized TensorRT engines
for ultra-low latency gesture classification on CUDA targets.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np


class TensorRTInference:
    """TensorRT engine execution context wrapper."""

    def __init__(self, engine_path: Path | str = "models/trt_engines/gesture_classifier.engine") -> None:
        self.engine_path = Path(engine_path)
        self.context = None
        self.engine = None

        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit

            self.trt = trt
            self.cuda = cuda

            logger = trt.Logger(trt.Logger.WARNING)
            with open(self.engine_path, "rb") as f:
                runtime = trt.Runtime(logger)
                self.engine = runtime.deserialize_cuda_engine(f.read())
                self.context = self.engine.create_execution_context()
        except Exception as e:
            print(f"[TRT Inference] Unable to initialize TensorRT execution context: {e}")

    def predict(self, input_sequence: np.ndarray) -> np.ndarray:
        """Run TensorRT inference on landmark input sequence (1, 30, 63).

        Args:
            input_sequence: Array of shape (1, 30, 63) or (30, 63).

        Returns:
            Logits array of shape (1, 4).
        """
        if self.context is None:
            # CPU Fallback dummy output if TRT runtime unavailable
            return np.zeros((1, 4), dtype=np.float32)

        input_data = np.ascontiguousarray(input_sequence, dtype=np.float32)
        if input_data.ndim == 2:
            input_data = np.expand_dims(input_data, axis=0)

        output = np.empty((1, 4), dtype=np.float32)

        # Allocate CUDA device memory
        d_input = self.cuda.mem_alloc(input_data.nbytes)
        d_output = self.cuda.mem_alloc(output.nbytes)

        # Transfer host to device
        self.cuda.memcpy_htod(d_input, input_data)

        # Execute TensorRT context
        bindings = [int(d_input), int(d_output)]
        self.context.execute_v2(bindings)

        # Transfer device to host
        self.cuda.memcpy_dtoh(output, d_output)
        return output
