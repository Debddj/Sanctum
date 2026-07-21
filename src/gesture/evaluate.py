"""Gesture classifier evaluation script.

Evaluates trained checkpoint on test split, measures PyTorch CPU baseline latency,
and outputs classification metrics and confusion matrix data.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gesture.dataset import GestureDataset
from src.gesture.model import GestureCNN1D, GestureLSTM

GESTURE_NAMES = ["sling_ring", "mudra_hold", "pinch_pull", "open_palm"]


def evaluate_checkpoint(
    checkpoint_path: Path | str = "models/checkpoints/gesture_classifier.pt",
) -> dict:
    """Evaluate trained gesture model on test set."""
    cp_file = Path(checkpoint_path)
    if not cp_file.exists():
        cp_file = PROJECT_ROOT / checkpoint_path

    if not cp_file.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {cp_file}")

    device = torch.device("cpu")  # Baseline evaluation on CPU
    checkpoint = torch.load(cp_file, map_location=device)

    arch = checkpoint.get("architecture", "cnn_1d")
    if arch == "lstm":
        model = GestureLSTM(input_features=63, num_classes=4).to(device)
    else:
        model = GestureCNN1D(input_features=63, num_classes=4).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    splits_dir = PROJECT_ROOT / "data" / "splits"
    seq_dir = PROJECT_ROOT / "data" / "landmark_sequences"
    test_manifest = splits_dir / "test_manifest.json"

    test_dataset = GestureDataset(manifest_path=test_manifest, sequence_dir=seq_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    all_preds = []
    all_targets = []
    latencies_ms = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)

            # High resolution latency measurement
            start_ns = time.perf_counter_ns()
            logits = model(x)
            end_ns = time.perf_counter_ns()

            latencies_ms.append((end_ns - start_ns) / 1e6)
            pred = logits.argmax(dim=1).item()

            all_preds.append(pred)
            all_targets.append(y.item())

    all_preds_arr = np.array(all_preds)
    all_targets_arr = np.array(all_targets)

    acc = np.mean(all_preds_arr == all_targets_arr)
    mean_lat = float(np.mean(latencies_ms))
    p50_lat = float(np.percentile(latencies_ms, 50))
    p95_lat = float(np.percentile(latencies_ms, 95))
    p99_lat = float(np.percentile(latencies_ms, 99))

    # Confusion matrix
    conf_matrix = np.zeros((4, 4), dtype=int)
    for t, p in zip(all_targets, all_preds):
        conf_matrix[t, p] += 1

    print("=== Sanctum Gesture Classifier Evaluation ===")
    print(f"Architecture: {arch.upper()}")
    print(f"Test Accuracy: {acc * 100:.2f}% ({np.sum(all_preds_arr == all_targets_arr)}/{len(all_targets)})")
    print(f"Inference Latency (CPU): mean={mean_lat:.3f}ms | p50={p50_lat:.3f}ms | p95={p95_lat:.3f}ms | p99={p99_lat:.3f}ms")
    print("\nConfusion Matrix:")
    print("Pred -> ", " ".join(f"{g:>10}" for g in GESTURE_NAMES))
    for idx, row in enumerate(conf_matrix):
        print(f"{GESTURE_NAMES[idx]:>10}: " + " ".join(f"{val:>10}" for val in row))

    return {
        "accuracy": acc,
        "mean_latency_ms": mean_lat,
        "p50_latency_ms": p50_lat,
        "p95_latency_ms": p95_lat,
        "p99_latency_ms": p99_lat,
        "confusion_matrix": conf_matrix.tolist(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate gesture classifier.")
    parser.add_argument("--checkpoint", type=str, default="models/checkpoints/gesture_classifier.pt")
    args = parser.parse_args()
    evaluate_checkpoint(args.checkpoint)
