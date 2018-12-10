Installation
============

Conda
-----

The recommended installation method for fastTSNE is using ``conda`` and can be easily installed from ``conda-forge`` with

.. code-block:: text

    conda install --channel conda-forge fasttsne

`Conda package <https://anaconda.org/conda-forge/fasttsne>`_

PyPi
----

fastTSNE is also available through ``pip`` and can be installed with

.. code-block:: text

    pip install fasttsne

`PyPi package <https://pypi.org/project/fastTSNE/>`_

Note, however, that fastTSNE requires a C/C++ compiler. ``numpy`` must also be installed. If it is not available, we will attempt to install it before proceeding with the installation.

In order for fastTSNE to utilize multiple threads, the C/C++ compiler must also implement ``OpenMP``. In practice, almost all compilers implement this with the exception of older version of ``clang`` on OSX systems.

To squeeze the most out of fastTSNE, you may also consider installing FFTW3 prior to installation. FFTW3 implements the Fast Fourier Transform, which is heavily used in fastTSNE. If FFTW3 is not available, fastTSNE will use numpy's implementation of the FFT, which is slightly slower than FFTW. The difference is only noticeable with large data sets containing millions of data points.
