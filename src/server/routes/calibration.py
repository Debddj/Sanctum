"""Calibration API routes."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/calibration", tags=["calibration"])


@router.post("/start")
async def start_calibration() -> dict:
    """Begin hand tracking calibration sequence."""
    # TODO: Guide user through calibration poses
    return {"status": "calibrating"}


@router.get("/status")
async def calibration_status() -> dict:
    """Get calibration state."""
    # TODO: Return calibration quality metrics
    return {"calibrated": False, "quality": None}
