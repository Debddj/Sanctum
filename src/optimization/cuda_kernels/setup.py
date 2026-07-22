"""Build script for portal_particles_cuda pybind11 CUDA extension.

Compiles portal_particles.cu and bindings.cpp into a native C++ extension module.
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="portal_particles_cuda",
    ext_modules=[
        CUDAExtension(
            name="portal_particles_cuda",
            sources=[
                "bindings.cpp",
                "portal_particles.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"],
            },
        ),
    ],
    cmdclass={
        "build_ext": BuildExtension
    },
)
