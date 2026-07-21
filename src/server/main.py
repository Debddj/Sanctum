"""FastAPI application entrypoint.

Sets up the ASGI app, mounts WebSocket endpoints, and initializes the
pipeline orchestrator on startup.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize and tear down pipeline resources."""
    # TODO: Initialize pipeline orchestrator, webcam stream, MediaPipe
    print("Sanctum server starting...")
    yield
    # TODO: Clean up resources
    print("Sanctum server shutting down.")


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


@app.get("/health")
async def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok", "service": "sanctum"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Main WebSocket endpoint for frame-synced landmark and effect streaming."""
    await websocket.accept()
    try:
        while True:
            # TODO: Stream landmarks + effect state to frontend
            data = await websocket.receive_text()
            await websocket.send_json({"type": "ack", "data": data})
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
