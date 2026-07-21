"""Export trained PyTorch gesture classifier to ONNX format.

Produces models/checkpoints/gesture_classifier.onnx with dynamic batch axes,
serving as the entry format for TensorRT engine compilation in Phase 7.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gesture.model import GestureCNN1D, GestureLSTM


def export_to_onnx(
    checkpoint_path: Path | str = "models/checkpoints/gesture_classifier.pt",
    onnx_output_path: Path | str = "models/checkpoints/gesture_classifier.onnx",
) -> Path:
    """Export PyTorch checkpoint to ONNX format."""
    cp_file = Path(checkpoint_path)
    if not cp_file.exists():
        cp_file = PROJECT_ROOT / checkpoint_path

    out_file = Path(onnx_output_path)
    if not out_file.is_absolute():
        out_file = PROJECT_ROOT / onnx_output_path

    out_file.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    checkpoint = torch.load(cp_file, map_location=device)

    arch = checkpoint.get("architecture", "cnn_1d")
    if arch == "lstm":
        model = GestureLSTM(input_features=63, num_classes=4).to(device)
    else:
        model = GestureCNN1D(input_features=63, num_classes=4).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dummy_input = torch.randn(1, 30, 63, device=device)

    print(f"[ONNX] Exporting {arch.upper()} model to {out_file}...")

    # Legacy ONNX exporter avoids torch.export dynamo unicode console print issues on Windows
    torch.onnx.export(
        model,
        dummy_input,
        out_file,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        dynamo=False,
    )

    print(f"[ONNX] Successfully exported model to {out_file}")
    return out_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX.")
    parser.add_argument("--checkpoint", type=str, default="models/checkpoints/gesture_classifier.pt")
    parser.add_argument("--output", type=str, default="models/checkpoints/gesture_classifier.onnx")
    args = parser.parse_args()
    export_to_onnx(args.checkpoint, args.output)
