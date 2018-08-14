import os

import numpy as np
import setuptools
from setuptools import setup, Extension

USE_CYTHON = os.environ.get('USE_CYTHON', False)
ext = 'pyx' if USE_CYTHON else 'c'

extensions = [
    Extension('fastTSNE.quad_tree', ['fastTSNE/quad_tree.%s' % ext],
              extra_compile_args=['-fopenmp', '-O3'],
              extra_link_args=['-fopenmp', '-O3'],
              include_dirs=[np.get_include()],
              ),
    Extension('fastTSNE._tsne', ['fastTSNE/_tsne.%s' % ext],
              extra_compile_args=['-fopenmp', '-lfftw3', '-O3'],
              extra_link_args=['-fopenmp', '-lfftw3', '-O3'],
              include_dirs=[np.get_include()],
              ),
    Extension('fastTSNE.kl_divergence', ['fastTSNE/kl_divergence.%s' % ext],
              extra_compile_args=['-fopenmp', '-O3'],
              extra_link_args=['-fopenmp', '-O3'],
              include_dirs=[np.get_include()],
              ),
]

if USE_CYTHON:
    from Cython.Build import cythonize
    extensions = cythonize(extensions)

setup(
    name='fastTSNE',
    description='',
    author='Pavlin Poličar',
    author_email='pavlin.g.p@gmail.com',
    version='0.1.8',
    url='https://github.com/pavlin-policar/fastTSNE',
    packages=setuptools.find_packages(),
    ext_modules=extensions,
    install_requires=[
        'numpy==1.14.4',
        'pyFFTW==0.10.4',
        'pynndescent==0.2.0',
        'scikit-learn==0.19.1',
        'scipy==1.1.0',
    ]
)
