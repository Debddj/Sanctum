"""Gesture classifier training script.

Trains the 1D-CNN or LSTM classifier using landmark sequence datasets
and saves the trained checkpoint to models/checkpoints/gesture_classifier.pt.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gesture.dataset import GestureDataset
from src.gesture.model import GestureCNN1D, GestureLSTM


def train_model(config_path: Path | str = "configs/model/classifier.yaml") -> Path:
    """Train gesture classifier model and return saved checkpoint path."""
    config_file = Path(config_path)
    if not config_file.exists():
        config_file = PROJECT_ROOT / config_path

    with open(config_file) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Using execution device: {device}")

    # Paths
    splits_dir = PROJECT_ROOT / "data" / "splits"
    seq_dir = PROJECT_ROOT / "data" / "landmark_sequences"
    checkpoint_dir = PROJECT_ROOT / "models" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "gesture_classifier.pt"

    train_manifest = splits_dir / "train_manifest.json"
    val_manifest = splits_dir / "val_manifest.json"

    if not train_manifest.exists():
        raise FileNotFoundError(
            f"Dataset manifest not found at {train_manifest}. Run scripts/record_gesture_samples.py first."
        )

    # Datasets & DataLoaders
    batch_size = config.get("training", {}).get("batch_size", 32)
    train_dataset = GestureDataset(manifest_path=train_manifest, sequence_dir=seq_dir)
    val_dataset = GestureDataset(manifest_path=val_manifest, sequence_dir=seq_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model selection (default CNN1D per design decision)
    arch = config.get("model", {}).get("architecture", "cnn_1d")
    num_classes = config.get("model", {}).get("num_classes", 4)
    features = config.get("model", {}).get("input_features", 63)

    if arch == "lstm":
        model = GestureLSTM(input_features=features, num_classes=num_classes).to(device)
    else:
        model = GestureCNN1D(input_features=features, num_classes=num_classes).to(device)

    # Optimizer & Loss
    epochs = config.get("training", {}).get("epochs", 50)
    lr = config.get("training", {}).get("learning_rate", 0.001)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * len(y_batch)
            preds = logits.argmax(dim=1)
            correct_train += (preds == y_batch).sum().item()
            total_train += len(y_batch)

        scheduler.step()

        # Validation
        model.eval()
        total_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                logits = model(x_val)
                loss = criterion(logits, y_val)

                total_val_loss += loss.item() * len(y_val)
                preds = logits.argmax(dim=1)
                correct_val += (preds == y_val).sum().item()
                total_val += len(y_val)

        avg_train_loss = total_train_loss / max(1, total_train)
        train_acc = correct_train / max(1, total_train)
        avg_val_loss = total_val_loss / max(1, total_val)
        val_acc = correct_val / max(1, total_val)

        if epoch % 5 == 0 or epoch == epochs:
            print(
                f"[Epoch {epoch:02d}/{epochs}] "
                f"Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.4f}"
            )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": avg_val_loss,
                    "val_acc": val_acc,
                    "architecture": arch,
                },
                checkpoint_path,
            )

    print(f"[Train] Complete! Saved best model checkpoint to {checkpoint_path}")
    return checkpoint_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train gesture classifier.")
    parser.add_argument("--config", type=str, default="configs/model/classifier.yaml")
    args = parser.parse_args()
    train_model(args.config)
