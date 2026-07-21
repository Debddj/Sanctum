"""FastAPI application entrypoint.

Sets up the ASGI app, mounts WebSocket endpoints, and initializes the
pipeline orchestrator on startup.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from src.server.pipeline_orchestrator import PipelineOrchestrator
from src.server.routes import calibration, session

orchestrator = PipelineOrchestrator()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize and tear down pipeline resources."""
    print("[Sanctum] Server starting, initializing pipeline orchestrator...")
    orchestrator.start()
    yield
    print("[Sanctum] Server shutting down, stopping pipeline orchestrator...")
    orchestrator.stop()


app = FastAPI(
    title="Sanctum",
    description="Gesture-driven AR VFX engine",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(session.router)
app.include_router(calibration.router)


@app.get("/health")
async def health() -> dict:
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "sanctum",
        "pipeline_running": orchestrator.is_running,
        "active_clients": len(orchestrator.active_websockets),
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Main WebSocket endpoint for frame-synced landmark and effect streaming."""
    await websocket.accept()
    await orchestrator.register_websocket(websocket)
    try:
        while True:
            raw_data = await websocket.receive_text()
            try:
                msg = json.loads(raw_data)
                # Echo / Ack for connection validation & latency ping tests
                await websocket.send_json({"type": "ack", "received": msg})
            except json.JSONDecodeError:
                await websocket.send_json({"type": "ack", "received": raw_data})
    except WebSocketDisconnect:
        await orchestrator.unregister_websocket(websocket)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
