from distutils.core import setup, Extension
import os
import numpy

os.environ['CC'] = 'g++'

_wolf = Extension('_wolf', ['wolf.cpp', 'wolf.i'],
					include_dirs=[numpy.get_include(), '.'],
					swig_opts=['-c++'],
					extra_link_args=['-stdlib=libc++'],
					extra_compile_args=['-I/Library/Developer/CommandLineTools/usr/include/c++/v1'])

setup(name='wolf', version='1.0', ext_modules=[_wolf], py_modules=['wolf'])