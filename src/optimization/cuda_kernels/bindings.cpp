/**
 * pybind11 bindings for CUDA portal particle simulation.
 *
 * Exposes launch_update_particles CUDA kernel to Python as a C++ extension module.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Forward declaration of C++/CUDA launcher
struct Particle {
    float px, py, pz;
    float vx, vy, vz;
    float life;
    float max_life;
};

extern void launch_update_particles(
    Particle* d_particles,
    int num_particles,
    float dt,
    float cx,
    float cy,
    float cz,
    float swirl
);

void update_particles_cpp(
    uintptr_t particle_ptr,
    int count,
    float dt,
    float cx,
    float cy,
    float cz,
    float swirl
) {
    Particle* d_particles = reinterpret_cast<Particle*>(particle_ptr);
    launch_update_particles(d_particles, count, dt, cx, cy, cz, swirl);
}

PYBIND11_MODULE(portal_particles_cuda, m) {
    m.doc() = "CUDA particle simulation module for Sanctum AR VFX";
    m.def(
        "update_particles",
        &update_particles_cpp,
        "Update particle positions and velocities using CUDA GPU kernel",
        py::arg("particle_ptr"),
        py::arg("count"),
        py::arg("dt"),
        py::arg("cx"),
        py::arg("cy"),
        py::arg("cz"),
        py::arg("swirl")
    );
}
