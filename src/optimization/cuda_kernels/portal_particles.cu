/**
 * Portal particle simulation — CUDA kernel for per-particle position/velocity update.
 *
 * Performs high-performance parallel particle physics (swirl forces, noise velocity integration,
 * lifetime updates) on CUDA GPU threads.
 */

#include <cuda_runtime.h>
#include <math.h>

struct Particle {
    float px, py, pz; // Position
    float vx, vy, vz; // Velocity
    float life;       // Current life (1.0 -> 0.0)
    float max_life;   // Total lifespan in seconds
};

extern "C" __global__ void update_portal_particles_kernel(
    Particle* particles,
    int num_particles,
    float dt,
    float center_x,
    float center_y,
    float center_z,
    float swirl_speed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    Particle p = particles[idx];

    // Update life
    p.life -= dt;

    if (p.life <= 0.0f) {
        // Respawn particle at portal perimeter
        float angle = ((float)idx / (float)num_particles) * 6.28318530718f;
        float radius = 0.5f;
        p.px = center_x + radius * cosf(angle);
        p.py = center_y + radius * sinf(angle);
        p.pz = center_z;

        // Tangential velocity for swirling motion
        p.vx = -swirl_speed * sinf(angle);
        p.vy = swirl_speed * cosf(angle);
        p.vz = 0.1f * sinf(angle * 3.0f);
        p.life = p.max_life;
    } else {
        // Integrate position
        p.px += p.vx * dt;
        p.py += p.vy * dt;
        p.pz += p.vz * dt;

        // Centripetal acceleration toward center
        float dx = center_x - p.px;
        float dy = center_y - p.py;
        p.vx += dx * 0.5f * dt;
        p.vy += dy * 0.5f * dt;
    }

    particles[idx] = p;
}

void launch_update_particles(
    Particle* d_particles,
    int num_particles,
    float dt,
    float cx,
    float cy,
    float cz,
    float swirl
) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_particles + threadsPerBlock - 1) / threadsPerBlock;
    update_portal_particles_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_particles, num_particles, dt, cx, cy, cz, swirl
    );
}
