from imp import find_module
import glob
from os.path import join
import setuptools
from setuptools.extension import Extension
import numpy as np

# -g compiles with debugging information.
# -O0 means compile with no optimization, try -O3 for blazing speed
compile_args = ['-g', '-O3', '-std=c++0x']
fast_package = 'codim1.fast'
ext = []
ext.append(Extension(fast_package + '_ext',
                  glob.glob('codim1/fast/*.cpp'),
                  include_dirs=['codim1/fast'],
                  library_dirs=['/'],
                  libraries=['boost_python'],
                  extra_compile_args=compile_args))

setuptools.setup(
   name = "codim1",
   version = '0.0.0',
   author = 'Ben Thompson',
   packages = ['codim1'],
   include_dirs = [np.get_include()],
   ext_modules=ext
)
