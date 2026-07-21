"""Gesture classifier training loop.

Handles data loading, training, validation, checkpointing, and
learning rate scheduling. Reads configuration from configs/model/classifier.yaml.
"""

from __future__ import annotations

# TODO: Implement training loop
# - Load classifier.yaml config
# - Initialize model (CNN1D or LSTM based on config)
# - Set up DataLoaders from GestureDataset
# - Training loop with cosine annealing LR schedule
# - Early stopping based on validation loss
# - Checkpoint saving to models/checkpoints/
# - TensorBoard / wandb logging (optional)

if __name__ == "__main__":
    raise NotImplementedError("Training loop not yet implemented — see Phase 2")
