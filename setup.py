from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

ext = [Extension("source.fast.elastic_kernel", sources=["./source/fast/elastic_kernel.pyx"] )]
ext = [Extension("source.fast.get_physical_points", sources=["./source/fast/get_physical_points.pyx"] )]

setup(
   name = "codim1",
   cmdclass = {'build_ext' : build_ext},
   include_dirs = [np.get_include()],
   ext_modules=ext
)
