.. _parameter-guide:

Parameter guide
===============


Perplexity
----------
Perplexity is perhaps the most important tunable parameter and can reveal different aspects of the data. Considered loosely, it can be thought of as the balance between preserving the global and the local structure of the data. A more direct way to think about perplexity is that it is the continuous analogy to the :math:`k` number of nearest neighbors for which we will preserve distances for each data point.

In most implementations, perplexity defaults to 30. This focuses the attention of t-SNE on preserving the distances to its 30 nearest neighbors and puts far less weight on preserving distances to the remaining points. For data sets with a small number of points e.g. 100, this will uncover the global structure quite well since each point will preserve distances to a third of the data set.

For larger data sets, e.g. 10,000 points, considering 30 nearest neighbors will do a poor job of preserving global structure. Using a higher value, e.g. 500, will do a fairly good job for of uncovering the global structure. For larger data sets still e.g. 500k or 1 million samples, this is typically not enough and can take quite a long time to run. Luckily, various tricks can be used to improve global structure [4]_.

Note that perplexity linearly impacts runtime i.e. higher values of
perplexity will incur longer execution time.


Exaggeration
------------

The exaggeration factor is typically used during the early exaggeration phase. This factor increases the attractive forces between points and allows points to move around more freely, finding their corresponding neighbors more easily. The most typical value of exaggeration during the early exaggeration phase is 12, but higher values have also been shown to work in combination with different learning rates [5]_.

Exaggeration can also be used in the normal optimization regime to form more densely packed clusters, making the separation between clusters more visible [4]_.

Optimization parameters
-----------------------

t-SNE uses a variation of gradient descent optimizationn proceedure that incorporates momentum and speeds up convergence of the embedding [6]_.

learning_rate: float
    The learning rate controls the step size of the gradient updates. This typically ranges from 100 to 1000, but usually the default works well enough.

    When dealing with large data sets on the order of 1 million points or more, it may be necessary to increase the learning rate to 1000 or to increase the number of iterations [4]_.

momentum: float
    Gradient descent with momentum keeps a sum exponentially decaying weights from previous iterations, speeding up convergence. In early stages of the optmization, this is typically set to a lower value (0.5 in most implementations) since points generally move around quite a bit in this phase and increased after the initial early exaggeration phase (typically to 0.8) to speed up convergence.


Barnes-Hut parameters
---------------------

Please refer to :ref:`barnes-hut` for a description of the Barnes-Hut algorithm.

theta: float
    The trade-off parameter between accuracy and speed.


Interpolation parameters
------------------------

Please refer to :ref:`fit-sne` for a description of the interpolation-based algorithm.

n_interpolation_points: int
    The number of interpolation points to use within each grid cell. It is highly recommended leaving this at the default value due to the Runge phenomenon described above.

min_num_intervals: int
    This value indicates what the minimum number of intervals/cells should be in any dimension.

ints_in_interval: float
    Our implementation dynamically determines the number of cells such that the accuracy for any given interval remains fixed. This value indicates the size of the interval/cell in any dimension e.g. setting this value to 3 indicates that all the cells should have side length of 3.


References
----------
.. [4] Kobak, Dmitry, and Philipp Berens. "The art of using t-SNE for single-cell transcriptomics." bioRxiv (2018): 453449.

.. [5] Linderman, George C., and Stefan Steinerberger. "Clustering with t-SNE, provably." arXiv preprint arXiv:1706.02582 (2017).

.. [6] Jacobs, Robert A. "Increased rates of convergence through learning rate adaptation." Neural networks 1.4 (1988): 295-307.
