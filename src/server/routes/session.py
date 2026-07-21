"""Session management API routes."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/session", tags=["session"])


@router.post("/start")
async def start_session() -> dict:
    """Start a new AR VFX session."""
    # TODO: Initialize pipeline, allocate resources
    return {"session_id": "placeholder", "status": "started"}


@router.post("/stop")
async def stop_session() -> dict:
    """Stop the current session and release resources."""
    # TODO: Tear down pipeline, release camera
    return {"status": "stopped"}


@router.get("/status")
async def session_status() -> dict:
    """Get current session status."""
    # TODO: Return pipeline state, FPS, active effects
    return {"status": "idle", "fps": 0, "active_effects": []}
