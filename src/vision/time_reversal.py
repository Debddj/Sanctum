"""Time-reversal effect — rewind trigger and ring-buffer playback.

On gesture trigger, plays the ring buffer backward and hands frames
to the frontend as a texture stream rather than live camera feed.
"""

from __future__ import annotations

# TODO: Implement in Phase 4
# - Listen for time_reversal gesture dispatch
# - Switch pipeline from live feed to ring buffer reverse playback
# - Apply optical flow blending for smooth rewind
# - Transition back to live feed on completion or gesture release

if __name__ == "__main__":
    raise NotImplementedError("Time reversal not yet implemented — see Phase 4")
