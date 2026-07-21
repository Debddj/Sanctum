# Sanctum — Gesture-Driven AR VFX Engine

A real-time augmented reality visual effects engine that translates hand gestures into cinematic VFX — portals, time reversal, and multiverse compositing — powered by MediaPipe, PyTorch, TensorRT, custom CUDA kernels, and Three.js.

## Overview

Sanctum captures webcam video, extracts hand landmarks via MediaPipe, classifies temporal gesture sequences using a 1D-CNN/LSTM classifier, and dispatches VFX events to a Three.js frontend over WebSocket. The pipeline is profiled end-to-end with per-stage latency tracing, then optimized with TensorRT (FP16/INT8) for the classifier and custom CUDA kernels for particle simulation.

## Architecture

```
Webcam → Frame Capture → MediaPipe Hands → Landmark Normalization
    → Sliding Window → Gesture Classifier → Effect Dispatch
    → WebSocket → Three.js Renderer (Portal / Time Reversal / Compositing)
```

## Key Features

- **Hand Tracking**: Dual-hand MediaPipe landmark extraction with wrist-relative, scale-invariant normalization
- **Gesture Classification**: 1D-CNN / LSTM classifier over sliding-window landmark sequences
- **Portal VFX**: Radial noise portal shader driven by hand centroid and normal vector
- **Time Reversal**: Ring-buffer frame playback with optical flow blending (Farneback / RAFT)
- **Segmentation**: Real-time background masking via MediaPipe Selfie Segmentation (SAM fallback)
- **TensorRT Optimization**: ONNX → TensorRT engine build (FP16/INT8), before/after latency benchmarks
- **CUDA Particle Sim**: Per-particle position/velocity updates in a custom CUDA kernel, exposed via pybind11
- **Profiling**: Per-stage span timers with baseline vs. optimized CSV reports

## Project Structure

```
sanctum/
├── src/               # Python backend — capture, landmarks, gesture, optimization, server, profiling
├── frontend/          # Vite + Three.js frontend — WebSocket client, portal shaders
├── configs/           # Pipeline, gesture, and model configuration (YAML)
├── data/              # Raw recordings, landmark sequences, train/val/test splits
├── models/            # Classifier checkpoints and TensorRT engines
├── notebooks/         # EDA, model development, and profiling notebooks
├── benchmarks/        # Latency CSVs and profiling reports
├── tests/             # Unit and end-to-end tests
├── docker/            # Dockerfiles and docker-compose
├── docs/              # Architecture docs and gesture specifications
└── scripts/           # Setup, data collection, and demo scripts
```

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA 12.x + cuDNN 8.x
- TensorRT 8.6+
- Node.js 18+
- Docker (optional, recommended)

### Development Setup

```bash
# Clone and set up Python environment
git clone <repo-url> sanctum && cd sanctum
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Copy environment config
cp .env.example .env

# Set up frontend
cd frontend && npm install && cd ..

# Run the demo
./scripts/run_demo.sh
```

### Docker

```bash
docker compose -f docker/docker-compose.yml up --build
```

## Implementation Phases

| Phase | Focus | Key Outputs |
|-------|-------|------------|
| 0 | Scaffolding | Package skeleton, Docker, WebSocket round-trip |
| 1 | Hand tracking | Landmark extraction, normalization, debug overlay |
| 2 | Gesture classifier | Dataset, 1D-CNN/LSTM training, baseline latency |
| 3 | Portal VFX | Radial noise shader, centroid/normal uniforms |
| 4 | Time reversal | Ring buffer, optical flow, rewind playback |
| 5 | Segmentation | Background masking, multiverse compositing |
| 6 | Profiling | Per-stage latency traces, baseline CSV |
| 7 | Optimization | TensorRT, CUDA kernels, optimized CSV |
| 8 | Integration | End-to-end demo, reproducible CI test |

## License

MIT
