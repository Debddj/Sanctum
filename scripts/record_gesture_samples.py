"""Data collection CLI — record labeled gesture samples from webcam.

Usage:
    python scripts/record_gesture_samples.py --gesture sling_ring --output data/landmark_sequences/
"""

from __future__ import annotations

# TODO: Implement in Phase 2
# - Parse CLI args (gesture class, output dir, duration)
# - Initialize webcam stream and MediaPipe Hands
# - Display live preview with landmark overlay
# - On keypress, start recording normalized landmark sequences
# - Save as .npy files with metadata JSON
# - Build train/val/test split manifests

if __name__ == "__main__":
    raise NotImplementedError("Data collection CLI not yet implemented — see Phase 2")
