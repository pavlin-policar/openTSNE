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


Installing from source
----------------------

If you wish to install openTSNE from source, please run

.. code-block:: text

    python setup.py install

in the root directory to install the appropriate dependencies and compile the necessary binary files.

Please note that openTSNE requires a C/C++ compiler to be available on the system. Additionally, numpy must be pre-installed in the active environment.

In order for openTSNE to utilize multiple threads, the C/C++ compiler must support ``OpenMP``. In practice, almost all compilers implement this with the exception of older version of ``clang`` on OSX systems.

To squeeze the most out of openTSNE, you may also consider installing FFTW3 prior to installation. FFTW3 implements the Fast Fourier Transform, which is heavily used in openTSNE. If FFTW3 is not available, openTSNE will use numpy’s implementation of the FFT, which is slightly slower than FFTW. The difference is only noticeable with large data sets containing millions of data points.
