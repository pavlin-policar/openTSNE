import os

import numpy as np
import setuptools
from setuptools import setup, Extension

USE_CYTHON = os.environ.get('USE_CYTHON', False)
ext = 'pyx' if USE_CYTHON else 'c'

extensions = [
    Extension('tsne.quad_tree', ['tsne/quad_tree.%s' % ext],
              extra_compile_args=['-fopenmp', '-O3'],
              extra_link_args=['-fopenmp', '-O3'],
              include_dirs=[np.get_include()],
              ),
    Extension('tsne._tsne', ['tsne/_tsne.%s' % ext],
              extra_compile_args=['-fopenmp', '-lfftw3', '-O3'],
              extra_link_args=['-fopenmp', '-lfftw3', '-O3'],
              include_dirs=[np.get_include()],
              ),
    Extension('tsne.kl_divergence', ['tsne/kl_divergence.%s' % ext],
              extra_compile_args=['-fopenmp', '-O3'],
              extra_link_args=['-fopenmp', '-O3'],
              include_dirs=[np.get_include()],
              ),
]

if USE_CYTHON:
    from Cython.Build import cythonize
    extensions = cythonize(extensions)

print(extensions)

setup(
    name='t-SNE',
    description='',
    author='Pavlin Poličar',
    author_email='pavlin.g.p@gmail.com',
    version='0.1.1',
    url='https://github.com/pavlin-policar/tSNE',
    packages=setuptools.find_packages(),
    ext_modules=extensions,
)
