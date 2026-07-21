"""Real-time person segmentation for background replacement.

Starts with MediaPipe Selfie Segmentation for real-time performance;
SAM is available as a quality fallback if mask quality is the bottleneck.
"""

from __future__ import annotations

# TODO: Implement in Phase 5
# - MediaPipe Selfie Segmentation for real-time person mask
# - Optional SAM integration for higher quality (at latency cost)
# - Return binary/alpha mask per frame
# - Pipeline orchestrator composites against swapped background

if __name__ == "__main__":
    raise NotImplementedError("Segmentation not yet implemented — see Phase 5")
