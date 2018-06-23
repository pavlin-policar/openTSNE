from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np


extensions = [
    Extension('tsne.quad_tree', ['tsne/quad_tree.pyx'],
              extra_compile_args=['-fopenmp', '-O3'],
              extra_link_args=['-fopenmp', '-O3'],
              include_dirs=[np.get_include()],
              ),
    Extension('tsne._tsne', ['tsne/_tsne.pyx'],
              extra_compile_args=['-fopenmp', '-lfftw3', '-O3'],
              extra_link_args=['-fopenmp', '-lfftw3', '-O3'],
              include_dirs=[np.get_include()],
              ),
    Extension('tsne._kl_divergence', ['tsne/_kl_divergence.pyx'],
              extra_compile_args=['-fopenmp', '-O3'],
              extra_link_args=['-fopenmp', '-O3'],
              include_dirs=[np.get_include()],
              ),
]

setup(
    name='t-SNE',
    description='',
    author='Pavlin Poličar',
    author_email='pavlin.g.p@gmail.com',
    ext_modules=cythonize(extensions),
)
