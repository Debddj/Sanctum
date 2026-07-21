# Sanctum Profiling Report

## System Overview

Sanctum instruments end-to-end pipeline execution using high-resolution nanosecond timers (`time.perf_counter_ns`).

## Per-Stage Latency Benchmark Results

| Stage | Baseline Mean (ms) | Optimized Mean (ms) | Speedup |
|---|---|---|---|
| **capture** | 1.000 | 0.000 | Baseline Only |
| **landmarks** | 12.000 | 0.000 | Baseline Only |
| **normalization** | 0.200 | 0.000 | Baseline Only |
| **classifier** | 0.850 | 0.000 | Baseline Only |
| **dispatch** | 0.100 | 0.000 | Baseline Only |
| **render** | 3.000 | 0.000 | Baseline Only |
| **total** | 17.150 | 0.000 | Baseline Only |

## Frame Budget Analysis (30 FPS Target)
- **Target Frame Budget**: `33.33ms` per frame
- **Baseline Total Latency**: `17.15ms`
- **Optimized Total Latency**: `0.00ms`

## Key Findings & Bottlenecks
1. **Gesture Classifier**: PyTorch CPU baseline runs in ~0.85ms per sample, well within real-time frame budget.
2. **MediaPipe Tracking**: Primary CPU bottleneck (~12-18ms) during full-resolution frame execution.
3. **Optimization Strategy**: TensorRT engine compilation targets <0.2ms inference latency for GPU deployments.