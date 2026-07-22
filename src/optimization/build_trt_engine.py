"""Build a TensorRT engine from an ONNX gesture classifier model.

Parses ONNX model graph, configures TensorRT builder flags (FP16/INT8),
and serializes the compiled engine binary to disk.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_engine(
    onnx_path: Path | str = "models/checkpoints/gesture_classifier.onnx",
    engine_path: Path | str = "models/trt_engines/gesture_classifier.engine",
    precision: str = "fp16",
    workspace_mb: int = 1024,
) -> Path:
    """Build and serialize TensorRT engine from ONNX model."""
    onnx_file = Path(onnx_path)
    if not onnx_file.is_absolute():
        onnx_file = PROJECT_ROOT / onnx_path

    out_file = Path(engine_path)
    if not out_file.is_absolute():
        out_file = PROJECT_ROOT / engine_path

    out_file.parent.mkdir(parents=True, exist_ok=True)

    if not onnx_file.exists():
        raise FileNotFoundError(f"ONNX model not found at {onnx_file}")

    try:
        import tensorrt as trt
    except ImportError:
        print("[TRT Build] TensorRT Python package not installed. Dummy engine file created for non-CUDA platforms.")
        out_file.write_bytes(b"DUMMY_TRT_ENGINE_PLACEHOLDER")
        return out_file

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)

    # TensorRT 10+ handles explicit batch implicitly; 8.x uses EXPLICIT_BATCH flag
    if hasattr(trt, "NetworkDefinitionCreationFlag") and hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(flags)
    else:
        network = builder.create_network()

    parser = trt.OnnxParser(network, logger)

    with open(onnx_file, "rb") as f:
        if not parser.parse(f.read()):
            for idx in range(parser.num_errors):
                print(f"[TRT Error] {parser.get_error(idx)}")
            raise RuntimeError("Failed to parse ONNX model for TensorRT engine build.")

    config = builder.create_builder_config()

    # Workspace limit compatibility across TensorRT 8.x, 10.x, 11.x
    if hasattr(config, "set_memory_pool_limit") and hasattr(trt, "MemoryPoolType"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * (1 << 20))
    elif hasattr(config, "max_workspace_size"):
        config.max_workspace_size = workspace_mb * (1 << 20)

    # Precision flags compatibility across TensorRT 8.x, 10.x, 11.x
    if precision.lower() == "fp16":
        fp16_flag = getattr(trt.BuilderFlag, "FP16", None) or getattr(trt.BuilderFlag, "PRECISION_FP16", None)
        if fp16_flag is not None:
            config.set_flag(fp16_flag)
    elif precision.lower() == "int8":
        int8_flag = getattr(trt.BuilderFlag, "INT8", None) or getattr(trt.BuilderFlag, "PRECISION_INT8", None)
        if int8_flag is not None:
            config.set_flag(int8_flag)

    print(f"[TRT Build] Compiling TensorRT engine (precision={precision}, workspace={workspace_mb}MB)...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        raise RuntimeError("TensorRT engine compilation failed.")

    # Convert to bytes if HostMemory object returned
    if hasattr(serialized_engine, "tobytes"):
        engine_bytes = serialized_engine.tobytes()
    else:
        engine_bytes = bytes(serialized_engine)

    with open(out_file, "wb") as f:
        f.write(engine_bytes)

    print(f"[TRT Build] Successfully built and saved TensorRT engine to {out_file}")
    return out_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build TensorRT engine from ONNX.")
    parser.add_argument("--onnx", type=str, default="models/checkpoints/gesture_classifier.onnx")
    parser.add_argument("--output", type=str, default="models/trt_engines/gesture_classifier.engine")
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp16", "fp32", "int8"])
    parser.add_argument("--workspace-mb", type=int, default=1024)
    args = parser.parse_args()

    build_engine(args.onnx, args.output, args.precision, args.workspace_mb)
