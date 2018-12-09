How t-SNE works
===============

What are the steps?


Barnes-Hut t-SNE
----------------

Parameters
~~~~~~~~~~

theta: float
    Only used when ``negative_gradient_method="bh"`` or its other aliases.
    This is the trade-off parameter between speed and accuracy of the tree
    approximation method. Typical values range from 0.2 to 0.8. The value 0
    indicates that no approximation is to be made and produces exact results
    also producing longer runtime. See [2]_ for more details.


Interpolation-based t-SNE
-------------------------

Parameters
~~~~~~~~~~

n_interpolation_points: int
    Only used when ``negative_gradient_method="fft"`` or its other aliases.
    The number of interpolation points to use within each grid cell for
    interpolation based t-SNE. It is highly recommended leaving this value
    at the default 3 as otherwise the interpolation may suffer from the
    Runge phenomenon. Theoretically, increasing this number will result in
    higher approximation accuracy, but practically, this can also be done
    with the ``ints_in_interval`` parameter, which does not suffer from the
    Runge phenomenon and should always be preferred. This is described in
    detail by Linderman [2]_.

min_num_intervals: int
    Only used when ``negative_gradient_method="fft"`` or its other aliases.
    The interpolation approximation method splits the embedding space into a
    grid, where the number of grid cells is governed by
    ``ints_in_interval``. Sometimes, especially during early stages of
    optimization, that number may be too small, and we may want better
    accuracy. The number of intervals used will thus always be at least the
    number specified here. Note that larger values will produce more precise
    approximations but will have longer runtime.

ints_in_interval: float
    Only used when ``negative_gradient_method="fft"`` or its other aliases.
    Since the coordinate range of the embedding changes during optimization,
    this value tells us how many integers should appear in a single e.g.
    setting this value to 3 means that the intervals will appear as follows:
    [0, 3), [3, 6), ... Lower values will need more intervals to fill the
    space, e.g. 1.5 will produce 4 intervals [0, 1.5), [1.5, 3), ...
    Therefore lower values will produce more intervals, producing more
    interpolation points which in turn produce better approximation at the
    cost of longer runtime.


References
----------
