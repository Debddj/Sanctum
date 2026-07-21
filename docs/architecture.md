# Sanctum Architecture

## System Overview

Sanctum is a real-time augmented reality VFX engine that translates hand gestures into cinematic visual effects. The system is split into a Python backend (FastAPI + MediaPipe + PyTorch/TensorRT) and a JavaScript frontend (Vite + Three.js), connected via WebSocket.

## Pipeline Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   Webcam     │────▶│  MediaPipe Hands  │────▶│  Normalization   │
│   Capture    │     │  (21 landmarks)   │     │  (wrist-relative)│
└──────────────┘     └──────────────────┘     └────────┬─────────┘
                                                        │
                                                        ▼
┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Effect      │◀────│ Gesture Classifier│◀────│ Sliding Window   │
│  Dispatch    │     │ (CNN1D / LSTM)    │     │ (30 frames)      │
└──────┬───────┘     └──────────────────┘     └──────────────────┘
       │
       ▼
┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  WebSocket   │────▶│  Three.js Scene   │────▶│  Portal Shader / │
│  Transport   │     │  Manager          │     │  VFX Renderer    │
└──────────────┘     └──────────────────┘     └──────────────────┘
```

## Module Boundaries

### `src/capture/` — Frame Acquisition
- Threaded webcam reader decoupled from processing pipeline
- Ring buffer for time-reversal frame storage

### `src/landmarks/` — Hand Tracking
- MediaPipe Hands wrapper (dual-hand)
- Wrist-relative normalization (makes classifier distance-invariant)
- Sliding window for temporal gesture sequences

### `src/gesture/` — Classification
- 1D-CNN and LSTM architectures
- Dataset, training, evaluation, ONNX export
- Designed for before/after benchmarking against TensorRT

### `src/optimization/` — Performance
- TensorRT engine build and inference (FP16/INT8)
- CUDA particle simulation kernel
- Kept as a separate module for the before/after narrative

### `src/vision/` — Computer Vision Effects
- Optical flow (Farneback / RAFT)
- Time-reversal playback
- Person segmentation (MediaPipe / SAM)

### `src/server/` — Backend
- FastAPI with WebSocket transport
- Pipeline orchestrator coordinates all stages
- REST routes for session and calibration management

### `src/profiling/` — Instrumentation
- Per-stage latency tracing with nanosecond precision
- Report generation for baseline vs. optimized comparison

### `frontend/` — Renderer
- Vite + Three.js
- WebSocket client for real-time landmark and effect streaming
- Custom GLSL shaders for portal VFX

## Design Decisions

1. **Optimization as a first-class module**: `src/optimization/` is separate from `src/gesture/` so the before/after comparison in `benchmarks/` is reviewable
2. **Server-side particle simulation**: CUDA kernel computes particle state; frontend just renders instanced particles (separates compute from WebGL)
3. **Farneback first, RAFT if needed**: Start with lower-latency optical flow; only upgrade if quality demands it
4. **MediaPipe Selfie Seg first, SAM if needed**: Same latency-first principle for segmentation
