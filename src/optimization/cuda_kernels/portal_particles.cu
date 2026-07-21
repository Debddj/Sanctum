/**
 * Portal particle simulation — CUDA kernel for per-particle position/velocity update.
 *
 * Moves the particle simulation off the WebGL vertex shader and onto the CUDA GPU,
 * splitting simulation (GPU-heavy compute) from rendering (WebGL). The server computes
 * particle state per frame and streams the buffer to the frontend.
 *
 * TODO: Implement in Phase 7
 * - Define Particle struct: position (float3), velocity (float3), lifetime, color
 * - __global__ update_particles kernel: integrate velocity, apply forces (spiral, gravity)
 * - Handle particle spawn/respawn based on portal centroid and hand normal
 * - Expose via pybind11 in bindings.cpp
 */

// Placeholder — implementation in Phase 7
