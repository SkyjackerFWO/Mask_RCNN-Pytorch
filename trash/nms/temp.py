import os
import torch
from torch.utils.cpp_extension import BuildExtension
from setuptools import setup, Extension
from torch.utils import cpp_extension

sources = ['src/nms.c']
headers = ['src/nms.h']
defines = []
with_cuda = True

if torch.cuda.is_available():
    print('Including CUDA code.')
    sources += ['src/nms_cuda.c']
    headers += ['src/nms_cuda.h']
    defines += [('WITH_CUDA', None)]
    with_cuda = True


ext_modules = [
    CppExtension(
        '_ext_extension.cpp', ['extension.cpp'],
        extra_compile_args=CXX_FLAGS),
    CppExtension(
        '_ext_extension.ort', ['ort_extension.cpp'],
        extra_compile_args=CXX_FLAGS),
    CppExtension(
        '_ext_extension.rng', ['rng_extension.cpp'],
        extra_compile_args=CXX_FLAGS),
]
this_file = os.path.dirname(os.path.realpath(__file__))
print(this_file)
extra_objects = ['src/cuda/nms_kernel.cu.o']
extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]

# ffi = BuildExtension(
#     name = '_ext.nms',
#     headers=headers,
#     sources=sources,
#     define_macros=defines,
#     relative_to=__file__,
#     with_cuda=with_cuda,
#     extra_objects=extra_objects
# )


setup(
        name='cuda_extension',
        ext_modules=[
            cpp_extension.CUDAExtension(
                    name = '_ext.nms',
                    # headers=headers,
                    sources=sources,
                    define_macros=defines,
                    extra_objects=extra_objects)
        ],
        cmdclass={
            'build_ext': BuildExtension
        })
