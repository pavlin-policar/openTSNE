import logging
import multiprocessing
from collections import Iterable

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from sklearn.utils import check_random_state

from . import _tsne
from .affinity import Affinities, NearestNeighborAffinities
from .quad_tree import QuadTree

EPSILON = np.finfo(np.float64).eps

log = logging.getLogger(__name__)


def _check_callbacks(callbacks):
    if callbacks is not None:
        # If list was passed, make sure all of them are actually callable
        if isinstance(callbacks, Iterable):
            if any(not callable(c) for c in callbacks):
                raise ValueError('`callbacks` must contain callable objects!')
        # The gradient descent method deals with lists
        elif callable(callbacks):
            callbacks = (callbacks,)
        else:
            raise ValueError('`callbacks` must be a callable object!')

    return callbacks


def _handle_nice_params(optim_params: dict) -> None:
    """Convert the user friendly params into something the optimizer can
    understand."""
    # Handle callbacks
    optim_params['callbacks'] = _check_callbacks(optim_params['callbacks'])
    optim_params['use_callbacks'] = optim_params['callbacks'] is not None

    # Handle negative gradient method
    negative_gradient_method = optim_params['negative_gradient_method']
    if callable(negative_gradient_method):
        negative_gradient_method = negative_gradient_method
    elif negative_gradient_method in {'bh', 'BH', 'barnes-hut'}:
        negative_gradient_method = kl_divergence_bh
    elif negative_gradient_method in {'fft', 'FFT', 'interpolation'}:
        negative_gradient_method = kl_divergence_fft
    else:
        raise ValueError('Unrecognized gradient method. Please choose one of '
                         'the supported methods or provide a valid callback.')
    optim_params['negative_gradient_method'] = negative_gradient_method

    # Handle number of jobs
    n_jobs = optim_params['n_jobs']
    if n_jobs < 0:
        n_cores = multiprocessing.cpu_count()
        # Add negative number of n_jobs to the number of cores, but increment by
        # one because -1 indicates using all cores, -2 all except one, and so on
        n_jobs = n_cores + n_jobs + 1

    # If the number of jobs, after this correction is still <= 0, then the user
    # probably thought they had more cores, so we'll default to 1
    if n_jobs <= 0:
        log.warning('`n_jobs` receieved value %d but only %d cores are available. '
                    'Defaulting to single job.' % (optim_params['n_jobs'], n_cores))
        n_jobs = 1

    optim_params['n_jobs'] = n_jobs


class OptimizationInterrupt(InterruptedError):
    """Optimization was interrupted by a callback

    Parameters
    ----------
    error: float
        The latest KL divergence of the embedding.
    final_embedding: Union[TSNEEmbedding, PartialTSNEEmbedding]
        The latest embedding. Is either a partial or full embedding, depending
        on where the error was raised.

    """
    def __init__(self, error, final_embedding):
        super().__init__()
        self.error = error
        self.final_embedding = final_embedding


class PartialTSNEEmbedding(np.ndarray):
    """A partial embedding is created when we take an existing
    :class:`TSNEEmbedding` and add new data to it. It differs from the typical
    embedding in that it would be unwise to add even more data to only the
    subset of already approximated data.

    If we would like to add new data multiple times to the existing embedding,
    we can simply do so by using the original embedding.

    """

    def __new__(cls, embedding, reference_embedding, P, gradient_descent_params):
        obj = np.asarray(embedding, dtype=np.float64, order='C').view(PartialTSNEEmbedding)

        obj.reference_embedding = reference_embedding
        obj.P = P
        obj.gradient_descent_params = gradient_descent_params

        obj.kl_divergence = None

        return obj

    def optimize(self, n_iter, inplace=False, propagate_exception=False,
                 **gradient_descent_params):
        """Run optmization on the embedding for a given nubmer of steps.

        Parameters
        ----------
        n_iter: int
            The number of optimization iterations.
        inplace: bool
            Whether or not to create a copy of the embedding or to perform
            updates inplace.
        propagate_exception: bool
            The optimization process can be interrupted using callbacks. This
            flag indicates whether we should propagate that exception or to
            simply stop optimization and return the resulting embedding.
        **gradient_descent_params: dict
            Any parameters accepted by :func:`gradient_descent` can be specified
            here for finer control of the optimization process.

        Returns
        -------
        PartialTSNEEmbedding
            An optimized partial t-SNE embedding.

        Raises
        ------
        OptimizationInterrupt
            If a callback stops the optimization and the ``propagate_exception``
            flag is set, then an exception is raised.

        """
        # Typically we want to return a new embedding and keep the old one intact
        if inplace:
            embedding = self
        else:
            embedding = PartialTSNEEmbedding(
                np.copy(self), self.reference_embedding, self.P, self.gradient_descent_params,
            )

        # If optimization parameters were passed to this funciton, prefer those
        # over the defaults specified in the TSNE object
        optim_params = dict(self.gradient_descent_params)
        optim_params.update(gradient_descent_params)
        _handle_nice_params(optim_params)
        optim_params['n_iter'] = n_iter

        try:
            error, embedding = gradient_descent(
                embedding=embedding, reference_embedding=self.reference_embedding,
                P=self.P, **optim_params)

        except OptimizationInterrupt as ex:
            log.info('Optimization was interrupted with callback.')
            if propagate_exception:
                raise ex
            error, embedding = ex.error, ex.final_embedding

        embedding.kl_divergence = error

        return embedding


class TSNEEmbedding(np.ndarray):
    """t-SNE embedding

    Parameters
    ----------
    affinities: Affinities
        An affinity index which can be used to compute the affinities of new
        points to the points in the existing embedding. The affinity index also
        contains the affinity matrix :math:`P` used during optimization.
    gradient_descent_params: dict
        Specifies all the parameters to use for gradient descent.
    random_state: Optional[Union[int, RandomState]]
        The random state parameter follows the convention used in scikit-learn.
        If the value is an int, random_state is the seed used by the random
        number generator. If the value is a RandomState instance, then it will
        be used as the random number generator. If the value is None, the random
        number generator is the RandomState instance used by `np.random`.

    Attributes
    ----------
    kl_divergence: float
        The KL divergence or error of the embedding.

    """

    def __new__(cls, embedding, affinities, gradient_descent_params, random_state=None):
        obj = np.asarray(embedding, dtype=np.float64, order='C').view(TSNEEmbedding)

        obj.affinities = affinities  # type: Affinities
        obj.gradient_descent_params = gradient_descent_params  # type: dict
        obj.random_state = random_state

        obj.kl_divergence = None

        return obj

    def optimize(self, n_iter, inplace=False, propagate_exception=False,
                 **gradient_descent_params):
        """Run optmization on the embedding for a given nubmer of steps.

        Parameters
        ----------
        n_iter: int
            The number of optimization iterations.
        inplace: bool
            Whether or not to create a copy of the embedding or to perform
            updates inplace.
        propagate_exception: bool
            The optimization process can be interrupted using callbacks. This
            flag indicates whether we should propagate that exception or to
            simply stop optimization and return the resulting embedding.
        **gradient_descent_params: dict
            Any parameters accepted by :func:`gradient_descent` can be specified
            here for finer control of the optimization process.

        Returns
        -------
        TSNEEmbedding
            An optimized t-SNE embedding.

        Raises
        ------
        OptimizationInterrupt
            If a callback stops the optimization and the ``propagate_exception``
            flag is set, then an exception is raised.

        """
        # Typically we want to return a new embedding and keep the old one intact
        if inplace:
            embedding = self
        else:
            embedding = TSNEEmbedding(np.copy(self), self.affinities,
                                      self.gradient_descent_params)

        # If optimization parameters were passed to this funciton, prefer those
        # over the defaults specified in the TSNE object
        optim_params = dict(self.gradient_descent_params)
        optim_params.update(gradient_descent_params)
        _handle_nice_params(optim_params)
        optim_params['n_iter'] = n_iter

        try:
            error, embedding = gradient_descent(
                embedding=embedding, P=self.affinities.P, **optim_params)

        except OptimizationInterrupt as ex:
            log.info('Optimization was interrupted with callback.')
            if propagate_exception:
                raise ex
            error, embedding = ex.error, ex.final_embedding

        embedding.kl_divergence = error

        return embedding

    def transform(self, X, perplexity=None, initialization='weighted',
                  early_exaggeration=2, early_exaggeration_iter=100,
                  initial_momentum=0.2, n_iter=300, final_momentum=0.4,
                  **gradient_descent_params):
        """Embed new points into the existing embedding.

        This procedure optimizes each point only with respect to the existing
        embedding i.e. it ignores any interactions between the points in ``X``
        among themselves.

        Parameters
        ----------
        X: np.ndarray
            The data matrix to be added to the existing embedding.
        perplexity: float
            Perplexity can be thought of as the continuous k number of neighbors
            to consider for each data point. To avoid confusion, note that
            perplexity linearly impacts runtime.
        initialization: Union[np.ndarray, str]
            The initial point positions to be used in the embedding space. Can
            be a precomputed numpy array, ``random`` or ``weighted``. In all
            cases, ``weighted`` should be preferred. It positions each point in
            the weighted mean position of it's nearest neighbors in the existing
            embedding. Typically, few optimization steps are needed for good
            embeddings. ``random`` positions all new points in the center of the
            embedding and should be used for demonstration only.
        early_exaggeration: float
            The early exaggeration factor.
        early_exaggeration_iter: int
            The number of iterations to run in the early exaggeration phase.
        initial_momentum: float
            As in regular t-SNE, optimization uses momentum for faster
            convergence. This value controls the momentum used during the
            *early exaggeration* phase.
        n_iter: int
            The number of iterations to run in the normal optimization regime.
        final_momentum: float
            As in regular t-SNE, optimization uses momentum for faster
            convergence. This value controls the momentum used during the normal
            regime and*late exaggeration* phase.
        **gradient_descent_params: dict
            Any parameters accepted by :func:`gradient_descent` can be specified
            here for finer control of the optimization process.

        Returns
        -------
        PartialTSNEEmbedding
            The positions of the new points in the embedding space.

        """
        embedding = self.prepare_partial(X, perplexity=perplexity, initialization=initialization)

        optim_params = dict(gradient_descent_params)

        try:
            # Early exaggeration with lower momentum to allow points to find more
            # easily move around and find their neighbors
            optim_params['momentum'] = initial_momentum
            optim_params['exaggeration'] = early_exaggeration
            optim_params['n_iter'] = early_exaggeration_iter
            embedding.optimize(inplace=True, propagate_exception=True, **optim_params)

            # Restore actual affinity probabilities and increase momentum to get
            # final, optimized embedding
            optim_params['momentum'] = final_momentum
            optim_params['exaggeration'] = None
            optim_params['n_iter'] = n_iter
            embedding.optimize(inplace=True, propagate_exception=True, **optim_params)

        except OptimizationInterrupt as ex:
            log.info('Optimization was interrupted with callback.')
            embedding = ex.final_embedding

        return embedding

    def prepare_partial(self, X, initialization='median', **affinity_params):
        """Prepare the partial embedding which can be optimized.

        In addition to generating initial coordinates via
        :meth:`generate_partial_coordinates`, this also precomputes the affinity
        matrix :math:`P`, which is used throughout the optimization.

        Parameters
        ----------
        X : np.ndarray
            The data matrix to be added to the existing embedding.
        initialization: Union[np.ndarray, str]
            The initial point positions to be used in the embedding space. Can
            be a precomputed numpy array, ``random``, ``weighted`` or ``median``.
            In all cases, ``weighted`` or ``median`` should be preferred. These
            positions each point at the weighted mean or median position of it's
            nearest neighbors in the existing embedding. Typically, few
            optimization steps are needed for good embeddings. ``weighted`` and
            ``median`` produce very similar results.

            ``random`` positions all new points in the center of the embedding
            and should be used for demonstration only.
        **affinity_params: dict
            Additional params to be passed to the ``Affinity.to_new`` method.

        Returns
        -------
        PartialTSNEEmbedding
            An unoptimized :class:`PartialTSNEEmbedding` object, prepared for
            optimization.

        """
        P = self.affinities.to_new(X, **affinity_params)

        # Extract neighbor indices and their affinities as a dense matrix from P
        # so they can be used in weighted initialization schemes
        P = P.tocsr()
        k_neighbors = int(P.data.shape[0] / P.shape[0])
        neighbors = P.indices.copy().reshape((P.shape[0], k_neighbors))
        distances = P.data.copy().reshape((P.shape[0], k_neighbors))

        embedding = self.generate_partial_coordinates(
            X, initialization, neighbors, distances,
        )

        return PartialTSNEEmbedding(
            embedding, reference_embedding=self, P=P,
            gradient_descent_params=self.gradient_descent_params,
        )

    def generate_partial_coordinates(self, X, initialization, neighbors, distances):
        """Generate initial coordinates for each data point.

        Unlike :meth:`prepare_partial`, this returns the initial coordinates in
        a :class:`np.ndarray` object.

        Parameters
        ----------
        X : np.ndarray
            The data matrix to be added to the existing embedding.
        initialization: Union[np.ndarray, str]
            The initial point positions to be used in the embedding space. Can
            be a precomputed numpy array, ``random`` or ``weighted``. In all
            cases, ``weighted`` should be preferred. It positions each point in
            the weighted mean position of it's nearest neighbors in the existing
            embedding. Typically, few optimization steps are needed for good
            embeddings. ``random`` positions all new points in the center of the
            embedding and should be used for demonstration only.
        neighbors: np.ndarray
            For every new point to be added in the embedding, we need to know
            the indices of its nearest neighbor points in the existing
            embedding. This is the typical format returned by scikit-learn's
            nearest neighbor methods. This is needed when
            ``initialization='weighted'``.
        distances: np.ndarray
            For every new point to be added in the embedding, we need to know
            the distance to its nearest neighbor points in the existing
            embedding. This is the typical format returned by scikit-learn's
            nearest neighbor methods. This is needed when
            ``initialization='weighted'``.

        Returns
        -------
        np.ndarray
            Initial positions for each data point.

        """
        n_samples = X.shape[0]
        n_components = self.shape[1]

        # If initial positions are given in an array, use a copy of that
        if isinstance(initialization, np.ndarray):
            if initialization.shape[0] != n_samples:
                raise ValueError(
                    'The provided initialization contains a different number'
                    'of samples (%d) than the data provided (%d).' % (
                        initialization.shape[0], n_samples)
                )
            if initialization.shape[1] != n_components:
                raise ValueError(
                    'The provided initialization contains a different number '
                    'of components (%d) than the embedding (%d).' % (
                        initialization.shape[1], n_components)
                )
            embedding = np.array(initialization)

        # Random initialization with isotropic normal distribution
        elif initialization == 'random':
            random_state = check_random_state(self.random_state)
            embedding = random_state.normal(0, 1e-2, (n_samples, n_components))

        elif initialization == 'weighted':
            embedding = np.zeros((n_samples, n_components))
            for i in range(n_samples):
                embedding[i] = np.average(self[neighbors[i]], axis=0, weights=distances[i])

        elif initialization == 'median':
            embedding = np.median(self[neighbors], axis=1)

        else:
            raise ValueError('Unrecognized initialization scheme `%s`.' % initialization)

        return embedding


class TSNE:
    """t-Distributed Stochastic Neighbor Embedding

    Parameters
    ----------
    n_components: int
        The dimension of the embedding space.
    perplexity: float
        Perplexity can be thought of as the continuous :math:`k` number of
        neighbors to consider for each data point. To avoid confusion, note that
        perplexity linearly impacts runtime.
    learning_rate: float
        The learning rate for the t-SNE optimization steps. Typical values range
        from 1 to 1000. Setting the learning rate too low or too high may result
        in the points forming a "ball". This is also known as the crowding
        problem.
    early_exaggeration_iter: int
        The number of iterations to run in the *early exaggeration* phase.
    early_exaggeration: float
        The early exaggeration factor. Typical values range from 12 to 32,
        however larger values have also been found to work well with specific
        values of learning rate. See Linderman and Steinberger [3]_ for more
        details.
    n_iter: int
        The number of iterations to run in the normal optimization regime.
    late_exaggeration_iter: int
        The number of iterations to run in the *late exaggeration* phase. This
        last phase of the optimization can improve separability of clusters, but
        is rarely used in practice. See Linderman et al. [4]_ for more details.
    late_exaggeration: float
        The late exaggeration factor. See Linderman et al. [4]_ for more details.
    theta: float
        Only used when ``negative_gradient_method='bh'`` or its other aliases.
        This is the trade-off parameter between speed and accuracy of the tree
        approximation method. Typical values range from 0.2 to 0.8. The value 0
        indicates that no approximation is to be made and produces exact results
        also producing longer runtime. See [2]_ for more details.
    n_interpolation_points: int
        Only used when ``negative_gradient_method='fft'`` or its other aliases.
        The number of interpolation points to use within each grid cell for
        interpolation based t-SNE. It is highly recommended leaving this value
        at the default 3 as otherwise the interpolation may suffer from the
        Runge phenomenon. Theoretically, increasing this number will result in
        higher approximation accuracy, but practically, this can also be done
        with the ``ints_in_interval`` parameter, which does not suffer from the
        Runge phenomenon and should always be preferred. This is described in
        detail by Linderman [2]_.
    min_num_intervals: int
        Only used when ``negative_gradient_method='fft'`` or its other aliases.
        The interpolation approximation method splits the embedding space into a
        grid, where the number of grid cells is governed by
        ``ints_in_interval``. Sometimes, especially during early stages of
        optimization, that number may be too small, and we may want better
        accuracy. The number of intervals used will thus always be at least the
        number specified here. Note that larger values will produce more precise
        approximations but will have longer runtime.
    ints_in_interval: float
        Only used when ``negative_gradient_method='fft'`` or its other aliases.
        Since the coordinate range of the embedding changes during optimization,
        this value tells us how many integers should appear in a single e.g.
        setting this value to 3 means that the intervals will appear as follows:
        [0, 3), [3, 6), ... Lower values will need more intervals to fill the
        space, e.g. 1.5 will produce 4 intervals [0, 1.5), [1.5, 3), ...
        Therefore lower values will produce more intervals, producing more
        interpolation points which in turn produce better approximation at the
        cost of longer runtime.
    initialization: Union[np.ndarray, str]
        The initial point positions to be used in the embedding space. Can be a
        precomputed numpy array, ``pca`` or ``random``. Please note that when
        passing in a precomputed positions, it is highly recommended that the
        point positions have small variance (var(Y) < 0.0001), otherwise you may
        get poor embeddings.
    metric: str
        The metric to be used to compute affinities between points in the
        original space.
    metric_params: Optional[dict]
        Additional keyword arguments for the metric function.
    initial_momentum: float
        t-SNE optimization uses momentum for faster convergence. This value
        controls the momentum used during the *early optimization* phase.
    final_momentum: float
        t-SNE optimization uses momentum for faster convergence. This value
        controls the momentum used during the normal regime and *late
        exaggeration* phase.
    n_jobs: int
        The number of jobs to run in parallel. This follows the scikit-learn
        convention, ``-1`` meaning all processors, ``-2`` meaning all but one
        processor and so on.
    neighbors: str
        Specifies the nearest neighbor method to use. Can be either ``exact`` or
        ``approx``. ``exact`` uses space partitioning binary trees from
        scikit-learn while ``approx`` makes use of nearest neighbor descent.
        Note that ``approx`` has a bit of overhead and will be slower on smaller
        data sets than exact search.
    negative_gradient_method: str
        Specifies the negative gradient approximation method to use. For smaller
        data sets, the Barnes-Hut approximation is appropriate [2]_ and can be
        set using one of the following aliases: ``bh``, ``BH`` or
        ``barnes-hut``. Note that the time complexity of Barnes-Hut scales as
        :math:`\mathcal{O}(N \log N)`. For larger data sets, the FFT accelerated
        interpolation method is more appropriate and can be set using one of the
        following aliases: ``fft``, ``FFT`` or ``ìnterpolation`` [4]_. Note that
        this method scales linearly in the number of points
        :math:`\mathcal{O}(N)` and its complexity is governed by the number of
        interpolation points.
    callbacks: Optional[Union[Callable, List[Callable]]]
        We can pass callbacks, that will be run every ``callbacks_every_iters``
        iterations. Each callback should accept three parameters, the first is
        the current iteration number, the second is the current KL divergence
        error and the last is the current embedding. The callback may return
        ``True`` in order to stop the optimization.
    callbacks_every_iters: int
        How many iterations should run between each time a callback is invoked.
    random_state: Optional[Union[int, RandomState]]
        The random state parameter follows the convention used in scikit-learn.
        If the value is an int, random_state is the seed used by the random
        number generator. If the value is a RandomState instance, then it will
        be used as the random number generator. If the value is None, the random
        number generator is the RandomState instance used by `np.random`.

    References
    ----------
    .. [1] Maaten, Laurens van der, and Geoffrey Hinton. "Visualizing data using
        t-SNE." Journal of machine learning research 9.Nov (2008): 2579-2605.
    .. [2] Van Der Maaten, Laurens. "Accelerating t-SNE using tree-based
        algorithms." The Journal of Machine Learning Research 15.1 (2014):
        3221-3245.
    .. [3] Linderman, George C., and Stefan Steinerberger. "Clustering with
        t-SNE, provably." arXiv preprint arXiv:1706.02582 (2017).
    .. [4] Linderman, George C., et al. "Efficient Algorithms for t-distributed
        Stochastic Neighborhood Embedding." arXiv preprint arXiv:1712.09005
        (2017).

    """

    def __init__(self, n_components=2, perplexity=30, learning_rate=100,
                 early_exaggeration_iter=250, early_exaggeration=12,
                 n_iter=750, late_exaggeration_iter=0, late_exaggeration=1.2,
                 theta=0.5, n_interpolation_points=3, min_num_intervals=10,
                 ints_in_interval=1, initialization='pca', metric='euclidean',
                 metric_params=None, initial_momentum=0.5, final_momentum=0.8,
                 n_jobs=1, neighbors='exact', negative_gradient_method='bh',
                 callbacks=None, callbacks_every_iters=50, random_state=None):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.early_exaggeration = early_exaggeration
        self.early_exaggeration_iter = early_exaggeration_iter
        self.n_iter = n_iter
        self.late_exaggeration = late_exaggeration
        self.late_exaggeration_iter = late_exaggeration_iter
        self.theta = theta
        self.n_interpolation_points = n_interpolation_points
        self.min_num_intervals = min_num_intervals
        self.ints_in_interval = ints_in_interval
        self.initialization = initialization
        self.metric = metric
        self.metric_params = metric_params
        self.initial_momentum = initial_momentum
        self.final_momentum = final_momentum
        self.n_jobs = n_jobs
        self.neighbors_method = neighbors
        self.negative_gradient_method = negative_gradient_method

        self.callbacks = callbacks
        self.callbacks_every_iters = callbacks_every_iters

        self.random_state = random_state

    def fit(self, X):
        """Fit a t-SNE embedding for a given data set.

        Runs the standard t-SNE optimization, consisting of the early
        exaggeration phase, normal optimization phase and, optionally, the late
        exaggeration phase.

        Parameters
        ----------
        X : np.ndarray
            The data matrix to be embedded.

        Returns
        -------
        TSNEEmbedding
            A fully optimized t-SNE embedding.

        """
        embedding = self.prepare_initial(X)

        try:
            # Early exaggeration with lower momentum to allow points to find more
            # easily move around and find their neighbors
            embedding.optimize(
                n_iter=self.early_exaggeration_iter, exaggeration=self.early_exaggeration,
                momentum=self.initial_momentum, inplace=True, propagate_exception=True,
            )

            # Restore actual affinity probabilities and increase momentum to get
            # final, optimized embedding
            embedding.optimize(
                n_iter=self.n_iter, momentum=self.final_momentum, inplace=True,
                propagate_exception=True,
            )

            # Use the trick described in [4]_ to get more separated clusters of
            # points by applying a late exaggeration phase
            embedding.optimize(
                n_iter=self.late_exaggeration_iter, exaggeration=self.late_exaggeration,
                momentum=self.final_momentum, inplace=True, propagate_exception=True,
            )

        except OptimizationInterrupt as ex:
            log.info('Optimization was interrupted with callback.')
            embedding = ex.final_embedding

        return embedding

    def prepare_initial(self, X, initialization=None, affinities=None):
        """Prepare the initial embedding which can be optimized.

        In addition to generating initial coordinates via
        :meth:`generate_initial_coordinates`, this also precomputes the affinity
        matrix :math:`P`, which is used throughout the optimization.

        Parameters
        ----------
        X : np.ndarray
            The data matrix to be embedded.
        initialization : Optional[Union[np.ndarray, str]]
            Initial positions for each data point. Note that the initialization
            must contain the same number of samples as X and must have the
            correct number of components. If the initialization method is not
            specified, the value passed to the constructor will be used.
        affinities: Optional[Affinities]
            Affinities for the input data samples. For the typical use of t-SNE
            this parameter can be ignored since the method will default to
            perplexity based nearest neighbor affinities.

        Returns
        -------
        TSNEEmbedding
            An unoptimized :class:`TSNEEmbedding` object, prepared for
            optimization.

        """
        # Get some initial coordinates for the embedding
        y_coords = self.generate_initial_coordinates(X, initialization=initialization)

        # If affinities are not given, we'll use the typical perplexity based
        # nearest neighbor affinities
        if affinities is None:
            affinities = NearestNeighborAffinities(
                X, self.perplexity, method=self.neighbors_method,
                metric=self.metric, metric_params=self.metric_params, n_jobs=self.n_jobs,
            )
        elif not isinstance(affinities, Affinities):
            raise TypeError(
                '`affinities` must be an instance of the `%s`. Got an instance '
                'of `%s` instead' % (Affinities.__name__, affinities.__class__.__name__)
            )

        gradient_descent_params = {
            # Degrees of freedom of the Student's t-distribution. The
            # suggestion degrees_of_freedom = n_components - 1 comes from [3]_.
            'dof':  max(self.n_components - 1, 1),

            'negative_gradient_method': self.negative_gradient_method,
            'learning_rate': self.learning_rate,
            # By default, use the momentum used in unexaggerated phase
            'momentum': self.final_momentum,

            # Barnes-Hut params
            'theta': self.theta,
            # Interpolation params
            'n_interpolation_points': self.n_interpolation_points,
            'min_num_intervals': self.min_num_intervals,
            'ints_in_interval': self.ints_in_interval,

            'n_jobs': self.n_jobs,
            # Callback params
            'callbacks': self.callbacks,
            'callbacks_every_iters': self.callbacks_every_iters,
        }

        return TSNEEmbedding(y_coords, affinities, gradient_descent_params, self.random_state)

    def generate_initial_coordinates(self, X, initialization=None):
        """Generate initial coordinates for each data point.

        Unlike :meth:`prepare_initial`, this returns the initial coordinates in
        a :class:`np.ndarray` object.

        Parameters
        ----------
        X : np.ndarray
            The data matrix to be embedded.
        initialization : Optional[Union[np.ndarray, str]]
            Initial positions for each data point. Note that the initialization
            must contain the same number of samples as X and must have the
            correct number of components. If the initialization method is not
            specified, the value passed to the constructor will be used.

        Returns
        -------
        np.ndarray
            Initial positions for each data point.

        """
        if initialization is None:
            initialization = self.initialization

        # If initial positions are given in an array, use a copy of that
        if isinstance(initialization, np.ndarray):
            if initialization.shape[0] != X.shape[0]:
                raise ValueError(
                    'The provided initialization contains a different number '
                    'of samples (%d) than the data provided (%d).' % (
                        initialization.shape[0], X.shape[0])
                )
            if initialization.shape[1] != self.n_components:
                raise ValueError(
                    'The provided initialization contains a different number '
                    'of components (%d) than the embedding (%d).' % (
                        initialization.shape[1], self.n_components)
                )

            embedding = np.array(initialization)

            variance = np.var(embedding, axis=0)
            if any(variance > 1e-4):
                log.warning(
                    'Variance of embedding is greater than 0.0001. Initial '
                    'embeddings with high variance may have display poor convergence.'
                )

            return embedding

        # Initialize the embedding using a PCA projection into the desired
        # number of components
        elif initialization == 'pca':
            pca = PCA(n_components=self.n_components, random_state=self.random_state)
            embedding = pca.fit_transform(X)
            # The PCA embedding may have high variance, which leads to poor convergence
            normalization = np.std(embedding[:, 0]) * 100
            embedding /= normalization

            return embedding

        # Random initialization with isotropic normal distribution
        elif initialization == 'random':
            random_state = check_random_state(self.random_state)
            return random_state.normal(0, 1e-2, (X.shape[0], self.n_components))

        else:
            raise ValueError('Unrecognized initialization scheme `%s`.' % initialization)


def kl_divergence_bh(embedding, P, dof, bh_params, reference_embedding=None,
                     should_eval_error=False, n_jobs=1, **_):
    gradient = np.zeros_like(embedding, dtype=np.float64, order='C')

    # In the event that we wish to embed new points into an existing embedding
    # using simple optimization, we compute optimize the new embedding points
    # w.r.t. the existing embedding. Otherwise, we want to optimize the
    # embedding w.r.t. itself
    if reference_embedding is None:
        reference_embedding = embedding

    # Compute negative gradient
    tree = QuadTree(reference_embedding)
    sum_Q = _tsne.estimate_negative_gradient_bh(
        tree, embedding, gradient, **bh_params, dof=dof, num_threads=n_jobs)
    del tree

    # Compute positive gradient
    sum_P, kl_divergence_ = _tsne.estimate_positive_gradient_nn(
        P.indices, P.indptr, P.data, embedding, reference_embedding, gradient,
        dof, num_threads=n_jobs, should_eval_error=should_eval_error,
    )

    gradient *= 2 * (dof + 1) / dof
    # Computing positive gradients summed up only unnormalized q_ijs, so we
    # have to include normalziation term separately
    if should_eval_error:
        kl_divergence_ += sum_P * np.log(sum_Q + EPSILON)

    return kl_divergence_, gradient


def kl_divergence_fft(embedding, P, dof, fft_params, reference_embedding=None,
                      should_eval_error=False, n_jobs=1, **_):
    gradient = np.zeros_like(embedding, dtype=np.float64, order='C')

    # Compute negative gradient.
    if embedding.ndim == 1 or embedding.shape[1] == 1:
        if reference_embedding is not None:
            sum_Q = _tsne.estimate_negative_gradient_fft_1d_with_reference(
                embedding.ravel(), reference_embedding.ravel(), gradient.ravel(), **fft_params)
        else:
            sum_Q = _tsne.estimate_negative_gradient_fft_1d(
                embedding.ravel(), gradient.ravel(), **fft_params)
    elif embedding.shape[1] == 2:
        if reference_embedding is not None:
            sum_Q = _tsne.estimate_negative_gradient_fft_2d_with_reference(
                embedding, reference_embedding, gradient, **fft_params)
        else:
            sum_Q = _tsne.estimate_negative_gradient_fft_2d(
                embedding, gradient, **fft_params)
    else:
        raise RuntimeError('Interpolation based t-SNE for >2 dimensions is '
                           'currently unsupported (and generally a bad idea)')

    # The positive gradient function needs a reference embedding always
    if reference_embedding is None:
        reference_embedding = embedding

    # Compute positive gradient
    sum_P, kl_divergence_ = _tsne.estimate_positive_gradient_nn(
        P.indices, P.indptr, P.data, embedding, reference_embedding, gradient,
        dof, num_threads=n_jobs, should_eval_error=should_eval_error,
    )

    gradient *= 2 * (dof + 1) / dof
    if should_eval_error:
        kl_divergence_ += sum_P * np.log(sum_Q + EPSILON)

    return kl_divergence_, gradient


def gradient_descent(embedding, P, dof, n_iter, negative_gradient_method,
                     learning_rate, momentum, exaggeration=None, min_gain=0.01,
                     min_grad_norm=1e-8, theta=0.5, n_interpolation_points=3,
                     min_num_intervals=10, ints_in_interval=10,
                     reference_embedding=None, n_jobs=1, use_callbacks=False,
                     callbacks=None, callbacks_every_iters=50):
    """Perform batch gradient descent with momentum and gains.

    Parameters
    ----------
    embedding : np.ndarray
        The current embedding Y in the desired space.
    P : csr_matrix
        Joint probability matrix :math:`P_{ij}`.
    dof : float
        Degrees of freedom of the Student's t-distribution.
    n_iter : int
        Number of iterations to run the optimization for.
    negative_gradient_method : Callable[..., Tuple[float, np.ndarray]]
        The callable takes the embedding as arguments and returns the (or an
        approximation to) KL divergence of the current embedding and the
        gradient of the embedding, with which to update the point locations.
    learning_rate : float
        The learning rate for t-SNE. Typical values range from 1 to 1000.
        Setting the learning rate too high will result in the crowding problem
        where all the points form a ball in the center of the space.
    momentum : float
        The momentum generates a weight for previous gradients that decays
        exponentially.
    exaggeration : float
        The exaggeration term is used to increase the attractive forces during
        the first steps of the optimization. This enables points to move more
        easily through others, helping find their true neighbors quicker.
    min_gain : float
        Minimum individual gain for each parameter.
    min_grad_norm : float
        If the gradient norm is below this threshold, the optimization will be
        stopped. In practice, this almost never happens.
    theta : float
        This is the trade-off parameter between speed and accuracy of the
        Barnes-Hut approximation of the negative forces. Setting a lower value
        will produce more accurate results, while setting a higher value will
        search through less of the space providing a rougher approximation.
        Scikit-learn recommends values between 0.2-0.8. This value is ignored
        unless the Barnes-Hut algorithm is used for gradients.
    n_interpolation_points : int
        The number of interpolation points to use for FFT accelerated
        interpolation based tSNE. It is recommended leaving this value at the
        default=3 as otherwise the interpolation may suffer from the Runge
        phenomenon. This value is ignored unless the interpolation based
        algorithm is used.
    min_num_intervals : int
        The minimum number of intervals into which we split our embedding. A
        larger value will produce better embeddings at the cost of performance.
        This value is ignored unless the interpolation based algorithm is used.
    ints_in_interval : float
        Since the coordinate range of the embedding will certainly change
        during optimization, this value tells us how many integer values should
        appear in a single interval. This number of intervals affect the
        embedding quality at the cost of performance. Less ints per interval
        will incur a larger number of intervals.
    reference_embedding : Optional[np.ndarray]
        If we are adding points to an existing embedding, we have to compute
        the gradients and errors w.r.t. the existing embedding.
    n_jobs : int
        Number of threads.
    use_callbacks : bool
    callbacks : Callable[[int, float, np.ndarray] -> bool]
        The callback should accept three parameters, the first is the current
        iteration, the second is the current KL divergence error and the last
        is the current embedding. The callback should return a boolean value
        indicating whether or not to stop optimization i.e. True to stop.
    callbacks_every_iters : int
        How often should the callback be called.

    Returns
    -------
    float
        The KL divergence of the optimized embedding.
    np.ndarray
        The optimized embedding Y.

    Raises
    ------
    OptimizationInterrupt
        If the provided callback interrupts the optimization, this is raised.

    """
    assert isinstance(embedding, np.ndarray), \
        '`embedding` must be an instance of `np.ndarray`. Got `%s` instead' \
        % type(embedding)

    if reference_embedding is not None:
        assert isinstance(reference_embedding, np.ndarray), \
            '`reference_embedding` must be an instance of `np.ndarray`. Got ' \
            '`%s` instead' % type(reference_embedding)

    update = np.zeros_like(embedding)
    gains = np.ones_like(embedding)

    bh_params = {'theta': theta}
    fft_params = {'n_interpolation_points': n_interpolation_points,
                  'min_num_intervals': min_num_intervals,
                  'ints_in_interval': ints_in_interval}

    # Lie about the P values for bigger attraction forces
    if exaggeration is None:
        exaggeration = 1

    if exaggeration != 1:
        P *= exaggeration

    # Notify the callbacks that the optimization is about to start
    if isinstance(callbacks, Iterable):
        for callback in callbacks:
            # Only call function if present on object
            getattr(callback, 'optimzation_about_to_start', lambda: ...)()

    for iteration in range(n_iter):
        should_call_callback = use_callbacks and (iteration + 1) % callbacks_every_iters == 0
        should_eval_error = should_call_callback

        error, gradient = negative_gradient_method(
            embedding, P, dof=dof, bh_params=bh_params, fft_params=fft_params,
            reference_embedding=reference_embedding, n_jobs=n_jobs,
            should_eval_error=should_eval_error,
        )

        # Correct the KL divergence w.r.t. the exaggeration if needed
        if should_eval_error and exaggeration != 1:
            error = error / exaggeration - np.log(exaggeration)

        if should_call_callback:
            # Continue only if all the callbacks say so
            should_stop = any((bool(c(iteration + 1, error, embedding)) for c in callbacks))
            if should_stop:
                # Make sure to un-exaggerate P so it's not corrupted in future runs
                if exaggeration != 1:
                    P /= exaggeration
                raise OptimizationInterrupt(error=error, final_embedding=embedding)

        # Update the embedding using the gradient
        grad_direction_flipped = np.sign(update) != np.sign(gradient)
        grad_direction_same = np.invert(grad_direction_flipped)
        gains[grad_direction_flipped] += 0.2
        gains[grad_direction_same] = gains[grad_direction_same] * 0.8 + min_gain
        update = momentum * update - learning_rate * gains * gradient
        embedding += update

        # Zero-mean the embedding
        embedding -= np.mean(embedding, axis=0)

        if np.linalg.norm(gradient) < min_grad_norm:
            log.info('Gradient norm eps reached. Finished.')
            break

    # Make sure to un-exaggerate P so it's not corrupted in future runs
    if exaggeration != 1:
        P /= exaggeration

    # The error from the loop is the one for the previous, non-updated
    # embedding. We need to return the error for the actual final embedding, so
    # compute that at the end before returning
    error, _ = negative_gradient_method(
        embedding, P, dof=dof, bh_params=bh_params, fft_params=fft_params,
        reference_embedding=reference_embedding, n_jobs=n_jobs,
        should_eval_error=True,
    )

    return error, embedding
