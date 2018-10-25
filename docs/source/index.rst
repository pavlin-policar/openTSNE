fastTSNE: Fast, parallel implementations of t-SNE
=================================================

t-Distributed Stochastic Neighbor Embedding or t-SNE for short is a popular dimensionality reduction technique suited for visualizing high dimensional data sets.

.. figure:: images/zeisel_2018.png

   A visualization of 160,796 single cell transcriptomes from the mouse nervous system [Zeisel 2018] computed in under 2 minutes using FFT accelerated interpolation and approximate nearest neighbors.

The goal of this project is to have fast implementations of t-SNE in one place, without any external C/C++ dependencies. This makes the package very easy to install and include in other projects.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   basic_usage


.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api
