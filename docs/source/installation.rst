Installation
============

Conda
-----

openTSNE can be easily installed from ``conda-forge`` with

.. code-block:: text

    conda install --channel conda-forge opentsne

`Conda package <https://anaconda.org/conda-forge/opentsne>`_

PyPi
----

openTSNE is also available through ``pip`` and can be installed with

.. code-block:: text

    pip install opentsne

`PyPi package <https://pypi.org/project/openTSNE/>`_

Note, however, that openTSNE requires a C/C++ compiler. ``numpy`` must also be installed.

In order for openTSNE to utilize multiple threads, the C/C++ compiler must also implement ``OpenMP``. In practice, almost all compilers implement this with the exception of older version of ``clang`` on OSX systems.

To squeeze the most out of openTSNE, you may also consider installing FFTW3 prior to installation. FFTW3 implements the Fast Fourier Transform, which is heavily used in openTSNE. If FFTW3 is not available, openTSNE will use numpy's implementation of the FFT, which is slightly slower than FFTW. The difference is only noticeable with large data sets containing millions of data points.
