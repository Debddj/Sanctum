"""Pipeline orchestrator — stage sequencing and effect dispatch.

Coordinates the flow: capture → landmarks → normalization → classifier
→ effect dispatch → WebSocket output. Reads pipeline.yaml for stage toggles.
"""

from __future__ import annotations

# TODO: Implement in Phase 0/1
# - Load pipeline.yaml configuration
# - Initialize enabled stages
# - Per-frame processing loop:
#   1. Read frame from WebcamStream
#   2. Extract landmarks via MediaPipeHands
#   3. Normalize landmarks
#   4. Push to SequenceWindow
#   5. Run classifier when window is ready
#   6. Dispatch effect based on gesture class
#   7. Send state to connected WebSocket clients
# - Integrate latency_tracer for per-stage timing
