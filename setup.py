import setuptools
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

cython_package = 'codim1.fast'
ext = [
        Extension(cython_package + ".elastic_kernel", sources=["./codim1/fast/elastic_kernel.pyx"]),
        Extension(cython_package + ".get_physical_points", sources=["./codim1/fast/get_physical_points.pyx"]),
        Extension(cython_package + ".integration", sources=["./codim1/fast/integration.pyx"])
      ]

ext = cythonize(ext)

setuptools.setup(
   name = "codim1",
   packages = ['codim1'],
   include_dirs = [np.get_include()],
   ext_modules=ext
)
