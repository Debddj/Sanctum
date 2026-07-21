"""PyTorch Dataset over landmark sequences for gesture classification.

Loads pre-extracted, normalized landmark sequences from disk and serves them
as (sequence, label) pairs for training and evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class GestureDataset(Dataset):
    """Dataset of landmark sequences labeled with gesture classes."""

    def __init__(
        self,
        manifest_path: str | Path,
        sequence_dir: str | Path,
        window_size: int = 30,
        feature_dim: int = 63,
        transform: Optional[object] = None,
    ) -> None:
        """
        Args:
            manifest_path: Path to a JSON manifest listing {file, label} entries.
            sequence_dir: Directory containing .npy sequence files.
            window_size: Expected sequence length.
            feature_dim: Expected feature dimension per frame.
            transform: Optional transform applied to each sequence.
        """
        self.sequence_dir = Path(sequence_dir)
        self.window_size = window_size
        self.feature_dim = feature_dim
        self.transform = transform

        with open(manifest_path) as f:
            self.entries: list[dict] = json.load(f)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        entry = self.entries[idx]
        seq_path = self.sequence_dir / entry["file"]
        label = entry["label"]

        sequence = np.load(seq_path).astype(np.float32)
        assert sequence.shape == (self.window_size, self.feature_dim)

        if self.transform is not None:
            sequence = self.transform(sequence)

        return torch.from_numpy(sequence), label
