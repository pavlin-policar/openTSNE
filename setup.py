import os
import sys
from setuptools import setup, Extension

import setuptools

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
optim_opt = '/O3' if IS_WINDOWS else '-O3'

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

if USE_CYTHON:
    from Cython.Build import cythonize

    extensions = cythonize(extensions)

setup(
    name='fastTSNE',
    description='',
    license='BSD-3-Clause',
    author='Pavlin Poličar',
    author_email='pavlin.g.p@gmail.com',
    version='0.2.9',
    url='https://github.com/pavlin-policar/fastTSNE',
    packages=setuptools.find_packages(),
    ext_modules=extensions,
    install_requires=[
        'numpy>1.14',
        'numba>=0.38.1',
        'scikit-learn>=0.19,<0.19.99',
        'scipy',
    ],
)
