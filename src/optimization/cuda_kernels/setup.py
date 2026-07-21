"""Build script for the CUDA particle simulation extension.

Compiles portal_particles.cu and bindings.cpp into a Python extension
module using setuptools and CUDA toolkit.
"""

from setuptools import setup, Extension

# TODO: Implement in Phase 7
# - Use torch.utils.cpp_extension.CUDAExtension or manual nvcc setup
# - Compile portal_particles.cu + bindings.cpp
# - Link against CUDA runtime and pybind11

if __name__ == "__main__":
    raise NotImplementedError("CUDA extension build not yet implemented — see Phase 7")
