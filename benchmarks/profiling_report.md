# Sanctum Profiling Report

## Overview

This report compares per-stage latency before and after TensorRT/CUDA optimization.

## Methodology

All measurements are taken on the full pipeline: capture → landmarks → normalization → classifier → dispatch → render.

Timing uses `time.perf_counter_ns()` via `src/profiling/latency_tracer.py`.

## Results

| Stage | Baseline (ms) | Optimized (ms) | Speedup |
|-------|---------------|----------------|---------|
| capture | — | — | — |
| landmarks | — | — | — |
| normalization | — | — | — |
| classifier | — | — | — |
| dispatch | — | — | — |
| render | — | — | — |
| **total** | **—** | **—** | **—** |

## Analysis

*TODO: Fill in after Phase 6 (baseline) and Phase 7 (optimized) runs.*

## Bottleneck Identification

*TODO: Identify from trace data — do not assume.*
