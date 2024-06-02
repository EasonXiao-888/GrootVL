# --------------------------------------------------------
# GrootVL: Tree Topology is All You Need in State Space Model
# Modified by Yicheng Xiao
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------


import os
import glob

import torch

from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import CUDAExtension

from setuptools import find_packages
from setuptools import setup

requirements = ["torch", "torchvision"]


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "tree_scan", "csrc")

    main_file = glob.glob(os.path.join(extensions_dir, "vision.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "*.cu"))

    sources = main_file
    extension = CppExtension
    define_macros = []

    if torch.cuda.is_available() and CUDA_HOME is not None:
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args = {
            "cxx": ["-O3"],
            "nvcc": [
                "-O3",
                "-DCUDA_HAS_FP16=1",
                "-D__CUDA_NO_HALF_OPERATORS__",
                "-D__CUDA_NO_HALF_CONVERSIONS__",
                "-D__CUDA_NO_HALF2_OPERATORS__"
            ]
        }
    else:
        raise NotImplementedError("Cuda is not available.")

    sources = [os.path.join(extensions_dir, s) for s in sources]
    include_dirs = [extensions_dir]
    ext_modules = [
        extension(
            "tree_scan._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args
        )
    ]
    if not os.path.exists("tree_scan"):
        os.makedirs("tree_scan")
    return ext_modules


setup(
    name="tree_scan",
    version="1.0",
    author="Lin Song & Yicheng Xiao",
    packages=find_packages(exclude=("configs", "tests",)),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
