from setuptools import setup
from Cython.Build import cythonize

setup(
    name='inversion_utils',
    ext_modules=cythonize("inversion_utils.pyx"),
    zip_safe=False,
    script_args=["build_ext", "--inplace"]
)


'''
from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy
#include_dirs=[numpy.get_include()]
#extensions = [Extension("utils_cy", ["utils_cy.pyx"], include_dirs=[numpy.get_include()])]

#setup(name='utils_cy', ext_modules=cythonize(extensions), script_args=["build_ext", "--inplace"])

# command: python3 setup.py build_ext --inplace

extensions = [Extension("inversion_utils", ["inversion_utils.pyx"])]

setup(name='inversion_utils', ext_modules=cythonize(extensions), script_args=["build_ext", "--inplace"])
'''