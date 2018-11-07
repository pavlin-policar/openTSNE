import os
import sys
import tempfile
from distutils import ccompiler
from distutils.errors import CompileError, LinkError
from os.path import join

import setuptools
from setuptools import setup, Extension

try:
    import numpy
except ImportError:
    from subprocess import call
    call(['pip', 'install', 'numpy'])

import numpy as np

USE_CYTHON = os.environ.get('USE_CYTHON', False)
ext = 'pyx' if USE_CYTHON else 'c'

IS_WINDOWS = sys.platform.startswith('win')
openmp_opt = '/openmp' if IS_WINDOWS else '-fopenmp'
optim_opt = '/Ox' if IS_WINDOWS else '-O3'


def has_c_library(library, extension='.c'):
    """Check whether a C/C++ library is available on the system to the compiler.

    Parameters
    ----------
    library: str
        The library we want to check for e.g. if we are interested in FFTW3, we
        want to check for `fftw3.h`, so this parameter will be `fftw3`.
    extension: str
        If we want to check for a C library, the extension is `.c`, for C++
        `.cc`, `.cpp` or `.cxx` are accepted.

    Returns
    -------
    bool
        Whether or not the library is available.

    """
    with tempfile.TemporaryDirectory(dir='.') as directory:
        name = join(directory, '%s.%s' % (library, extension))
        with open(name, 'w') as f:
            f.write('#include <%s.h>\n' % library)
            f.write('int main() {}\n')

        # Get a compiler instance
        compiler = ccompiler.new_compiler()
        assert isinstance(compiler, ccompiler.CCompiler)

        try:
            # Try to compile the file using the C compiler
            compiler.link_executable(compiler.compile([name]), name)
            return True
        except (CompileError, LinkError):
            return False


extensions = [
    Extension('fastTSNE.quad_tree', ['fastTSNE/quad_tree.%s' % ext],
              extra_compile_args=[openmp_opt, optim_opt],
              extra_link_args=[openmp_opt, optim_opt],
              include_dirs=[np.get_include()],
              ),
    Extension('fastTSNE._tsne', ['fastTSNE/_tsne.%s' % ext],
              extra_compile_args=[openmp_opt, optim_opt],
              extra_link_args=[openmp_opt, optim_opt],
              include_dirs=[np.get_include()],
              ),
    Extension('fastTSNE.kl_divergence', ['fastTSNE/kl_divergence.%s' % ext],
              extra_compile_args=[openmp_opt, optim_opt],
              extra_link_args=[openmp_opt, optim_opt],
              include_dirs=[np.get_include()],
              ),
]

# Check if we have access to FFTW3 and if so, use that implementation
if has_c_library('fftw3'):
    lm_opt = '/lm' if IS_WINDOWS else '-lm'
    fftw3_opt = '/lfftw3' if IS_WINDOWS else '-lfftw3'
    extensions.append(
        Extension('fastTSNE._matrix_mul.matrix_mul',
                  ['fastTSNE/_matrix_mul/matrix_mul_fftw3.%s' % ext],
                  extra_compile_args=[openmp_opt, optim_opt, fftw3_opt, lm_opt],
                  extra_link_args=[openmp_opt, optim_opt, fftw3_opt, lm_opt],
                  include_dirs=[np.get_include()],
                  )
        )
else:
    extensions.append(
        Extension('fastTSNE._matrix_mul.matrix_mul',
                  ['fastTSNE/_matrix_mul/matrix_mul_numpy.%s' % ext],
                  extra_compile_args=[openmp_opt, optim_opt],
                  extra_link_args=[openmp_opt, optim_opt],
                  include_dirs=[np.get_include()],
                  )
    )

if USE_CYTHON:
    from Cython.Build import cythonize

    extensions = cythonize(extensions)

setup(
    name='fastTSNE',
    description='',
    license='BSD-3-Clause',
    author='Pavlin Poličar',
    author_email='pavlin.g.p@gmail.com',
    version='0.2.11',
    url='https://github.com/pavlin-policar/fastTSNE',
    packages=setuptools.find_packages(),
    ext_modules=extensions,
    install_requires=[
        'numpy>1.14',
        'numba>=0.38.1',
        'scikit-learn>=0.20',
        'scipy',
    ],
)
