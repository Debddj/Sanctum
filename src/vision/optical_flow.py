"""Optical flow computation — Farneback (baseline) and RAFT (optional).

Used by the time-reversal effect to compute motion vectors for smooth
backward playback blending.
"""

from __future__ import annotations

# TODO: Implement in Phase 4
# - Farneback optical flow via cv2.calcOpticalFlowFarneback
# - Optional RAFT model from torchvision for higher quality
# - Return dense flow field for frame interpolation during rewind

if __name__ == "__main__":
    raise NotImplementedError("Optical flow not yet implemented — see Phase 4")
