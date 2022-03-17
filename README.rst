openTSNE
========

|Build Status| |ReadTheDocs Badge| |License Badge|

openTSNE is a modular Python implementation of t-Distributed Stochasitc Neighbor Embedding (t-SNE) [1]_, a popular dimensionality-reduction algorithm for visualizing high-dimensional data sets. openTSNE incorporates the latest improvements to the t-SNE algorithm, including the ability to add new data points to existing embeddings [2]_, massive speed improvements [3]_ [4]_, enabling t-SNE to scale to millions of data points and various tricks to improve global alignment of the resulting visualizations [5]_.

.. figure:: docs/source/images/macosko_2015.png
   :alt: Macosko 2015 mouse retina t-SNE embedding
   :align: center

   A visualization of 44,808 single cell transcriptomes obtained from the mouse retina [6]_ embedded using the multiscale kernel trick to better preserve the global aligment of the clusters.

- `Documentation <http://opentsne.readthedocs.io>`__
- `User Guide and Tutorial <https://opentsne.readthedocs.io/en/latest/tsne_algorithm.html>`__
- Examples: `basic <https://opentsne.readthedocs.io/en/latest/examples/01_simple_usage/01_simple_usage.html>`__, `advanced <https://opentsne.readthedocs.io/en/latest/examples/02_advanced_usage/02_advanced_usage.html>`__, `preserving global alignment <https://opentsne.readthedocs.io/en/latest/examples/03_preserving_global_structure/03_preserving_global_structure.html>`__, `embedding large data sets <https://opentsne.readthedocs.io/en/latest/examples/04_large_data_sets/04_large_data_sets.html>`__
- `Speed benchmarks <https://opentsne.readthedocs.io/en/latest/benchmarks.html>`__

Installation
------------

openTSNE requires Python 3.7 or higher in order to run.

Conda
~~~~~

openTSNE can be easily installed from ``conda-forge`` with

::

   conda install --channel conda-forge opentsne

`Conda package <https://anaconda.org/conda-forge/opentsne>`__

PyPi
~~~~

openTSNE is also available through ``pip`` and can be installed with

::

   pip install opentsne

`PyPi package <https://pypi.org/project/openTSNE>`__

Installing from source
~~~~~~~~~~~~~~~~~~~~~~

If you wish to install openTSNE from source, please run

::

   pip install .


in the root directory to install the appropriate dependencies and compile the necessary binary files.

Please note that openTSNE requires a C/C++ compiler to be available on the system.

In order for openTSNE to utilize multiple threads, the C/C++ compiler
must support ``OpenMP``. In practice, almost all compilers
implement this with the exception of older version of ``clang`` on OSX
systems.

To squeeze the most out of openTSNE, you may also consider installing
FFTW3 prior to installation. FFTW3 implements the Fast Fourier
Transform, which is heavily used in openTSNE. If FFTW3 is not available,
openTSNE will use numpy’s implementation of the FFT, which is slightly
slower than FFTW. The difference is only noticeable with large data sets
containing millions of data points.

A hello world example
---------------------

Getting started with openTSNE is very simple. First, we'll load up some data using scikit-learn

.. code:: python

   from sklearn import datasets

   iris = datasets.load_iris()
   x, y = iris["data"], iris["target"]

then, we'll import and run

.. code:: python

   from openTSNE import TSNE

   embedding = TSNE().fit(x)

Citation
--------

If you make use of openTSNE for your work we would appreciate it if you would cite the paper

.. code::

    @article {Poli{\v c}ar731877,
        author = {Poli{\v c}ar, Pavlin G. and Stra{\v z}ar, Martin and Zupan, Bla{\v z}},
        title = {openTSNE: a modular Python library for t-SNE dimensionality reduction and embedding},
        year = {2019},
        doi = {10.1101/731877},
        publisher = {Cold Spring Harbor Laboratory},
        URL = {https://www.biorxiv.org/content/early/2019/08/13/731877},
        eprint = {https://www.biorxiv.org/content/early/2019/08/13/731877.full.pdf},
        journal = {bioRxiv}
    }
    
openTSNE implements two efficient algorithms for t-SNE. Please consider citing the original authors of the algorithm that you use. If you use FIt-SNE (default), then the citation is [4]_ below, but if you use Barnes-Hut the citation is [3]_. 


References
----------

.. [1] Van Der Maaten, Laurens, and Hinton, Geoffrey. `“Visualizing data using
    t-SNE.” <http://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf>`__
    Journal of Machine Learning Research 9.Nov (2008): 2579-2605.
.. [2] Poličar, Pavlin G., Martin Stražar, and Blaž Zupan. `“Embedding to Reference t-SNE Space Addresses Batch Effects in Single-Cell Classification.” <https://link.springer.com/article/10.1007/s10994-021-06043-1>`__ Machine Learning (2021): 1-20.
.. [3] Van Der Maaten, Laurens. `“Accelerating t-SNE using tree-based algorithms.”
    <http://www.jmlr.org/papers/volume15/vandermaaten14a/vandermaaten14a.pdf>`__
    Journal of Machine Learning Research 15.1 (2014): 3221-3245.
.. [4] Linderman, George C., et al. `"Fast interpolation-based t-SNE for improved
    visualization of single-cell RNA-seq data." <https://www.nature.com/articles/s41592-018-0308-4>`__ Nature Methods 16.3 (2019): 243.
.. [5] Kobak, Dmitry, and Berens, Philipp. `“The art of using t-SNE for single-cell transcriptomics.” <https://www.nature.com/articles/s41467-019-13056-x>`__
    Nature Communications 10, 5416 (2019).
.. [6] Macosko, Evan Z., et al. \ `“Highly parallel genome-wide expression profiling of
    individual cells using nanoliter droplets.”
    <https://www.sciencedirect.com/science/article/pii/S0092867415005498>`__
    Cell 161.5 (2015): 1202-1214.

.. |Build Status| image:: https://dev.azure.com/pavlingp/openTSNE/_apis/build/status/Test?branchName=master
   :target: https://dev.azure.com/pavlingp/openTSNE/_build/latest?definitionId=1&branchName=master
.. |ReadTheDocs Badge| image:: https://readthedocs.org/projects/opentsne/badge/?version=latest
   :target: https://opentsne.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status
.. |License Badge| image:: https://img.shields.io/badge/License-BSD%203--Clause-blue.svg
   :target: https://opensource.org/licenses/BSD-3-Clause
