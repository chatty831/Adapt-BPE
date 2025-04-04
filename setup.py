import os
import sys

import pybind11
from setuptools import Extension, setup


def get_pybind_include():
    """Helper to get both pybind11 includes: normal and user."""
    return [pybind11.get_include(), pybind11.get_include(True)]


ext_modules = [
    Extension(
        name="bpe_module",
        sources=["bpe_bindings.cpp", "bpe.cpp"],
        include_dirs=get_pybind_include() + ["."],  # "." if your .hpp files are local
        language="c++",
        extra_compile_args=["-std=c++17"],
        extra_link_args=[],
    ),
]

setup(
    name="bpe_module",
    version="0.3",
    ext_modules=ext_modules,
)
