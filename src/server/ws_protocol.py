"""Frame-synced WebSocket message protocol.

Defines the message schema exchanged between the FastAPI server and the
Three.js frontend over the WebSocket connection.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional, Any

from pydantic import BaseModel


class MessageType(str, Enum):
    """WebSocket message types."""

    LANDMARKS = "landmarks"
    GESTURE = "gesture"
    EFFECT_STATE = "effect_state"
    FRAME_DATA = "frame_data"
    CALIBRATION = "calibration"
    ERROR = "error"


class LandmarkMessage(BaseModel):
    """Hand landmark data and camera video frame for a single frame."""

    type: MessageType = MessageType.LANDMARKS
    frame_id: int
    timestamp_ms: float
    hands: list[dict]  # List of {handedness, landmarks: [[x,y,z], ...]}
    image_b64: Optional[str] = None  # Base64 JPEG data URL for browser video display


class GestureMessage(BaseModel):
    """Gesture classification result."""

    type: MessageType = MessageType.GESTURE
    frame_id: int
    gesture_class: str
    confidence: float
    action: str  # dispatch action from gestures.yaml


class EffectStateMessage(BaseModel):
    """Current VFX effect state for the renderer."""

    type: MessageType = MessageType.EFFECT_STATE
    frame_id: int
    active_effects: list[Any] = []
    portal: Optional[dict] = None  # centroid, normal, radius, lifetime
    time_reversal: Optional[dict] = None  # active, progress
    segmentation: Optional[dict] = None  # mask_available, background_id
