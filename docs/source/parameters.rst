.. _parameter-guide:

Parameter guide
===============


Perplexity
----------
Perplexity is perhaps the most important tunable parameter and can reveal different aspects of the data. Considered loosely, it can be thought of as the balance between preserving the global and the local structure of the data. A more direct way to think about perplexity is that it is the continuous analogy to the :math:`k` number of nearest neighbors for which we will preserve distances for each data point.

In most implementations, perplexity defaults to 30. This focuses the attention of t-SNE on preserving the distances to its 30 nearest neighbors and puts far less weight on preserving distances to the remaining points. For data sets with a small number of points e.g. 100, this will uncover the global structure quite well since each point will preserve distances to a third of the data set.

For larger data sets, e.g. 10,000 points, considering 30 nearest neighbors will do a poor job of preserving global structure. Using a higher value, e.g. 500, will do a fairly good job for of uncovering the global structure. For larger data sets still e.g. 500k or 1 million samples, this is typically not enough and can take quite a long time to run. Luckily, various tricks can be used to improve global structure [5]_.

Note that perplexity linearly impacts runtime i.e. higher values of
perplexity will incur longer execution time.


Exaggeration
------------


Learning Rate
-------------


Momentum
--------


References
----------
.. [5] Kobak, Dmitry, and Philipp Berens. "The art of using t-SNE for single-cell transcriptomics." bioRxiv (2018): 453449.
