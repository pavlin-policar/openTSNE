import logging
from collections import Iterable
from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA

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


class OptimizationInterrupt(InterruptedError):
    def __init__(self, error: float, final_embedding: np.ndarray) -> None:
        super().__init__()
        self.error = error
        self.final_embedding = final_embedding


class PartialTSNEEmbedding(np.ndarray):
    """A partial embedding is created when we take an existing `TSNEEmbedding`
    and add new data to it. It differs from the typical embedding in that it
    would be unwise to add even more data to only the subset of already
    approximated data. Therefore, we don't allow this and save the computation
    of a nearest neighbor index.

    If we would like to add new data multiple times to the existing embedding,
    we can simply do so on the original embedding.

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
    def __new__(cls, embedding, affinities, gradient_descent_params, pca_init=None):
        obj = np.asarray(embedding, dtype=np.float64, order='C').view(TSNEEmbedding)

        obj.affinities = affinities  # type: Affinities
        obj.gradient_descent_params = gradient_descent_params  # type: dict
        obj.pca_init = pca_init  # type: Optional[PCA]

        obj.kl_divergence = None

        return obj

    def optimize(self, n_iter, inplace=False, propagate_exception=False,
                 **gradient_descent_params):
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
        embedding = self.prepare_partial(
            X, perplexity=perplexity, initialization=initialization)

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

    def prepare_partial(self, X, initialization='weighted', perplexity=None):
        """Get the initial positions of some new data to be fitted w.r.t. the
        existing embedding.

        Parameters
        ----------
        X : np.ndarray
        initialization : Optional[Union[str, np.ndarray]]
        perplexity : Optional[float]

        Returns
        -------
        PartialTSNEEmbedding

        """
        P, neighbors, distances = self.affinities.to_new(
            X, return_distances=True, perplexity=perplexity,
        )

        embedding = self.__generate_partial_coordinates(
            X, initialization, neighbors, distances,
        )

        return PartialTSNEEmbedding(
            embedding, reference_embedding=self, P=P,
            gradient_descent_params=self.gradient_descent_params,
        )

    def __generate_partial_coordinates(self, X, initialization, neighbors, distances):
        n_samples = X.shape[0]
        n_components = self.shape[1]

        # If initial positions are given in an array, use a copy of that
        if isinstance(initialization, np.ndarray):
            assert initialization.shape[0] == X.shape[0], \
                'The provided initialization contains a different number of ' \
                'samples (%d) than the data provided (%d).' % (
                    initialization.shape[0], X.shape[0])
            embedding = np.array(initialization)

        # Initialize the embedding using a PCA projection into the desired
        # number of components
        elif initialization == 'pca':
            assert self.pca_init is not None, \
                'The initial embedded was not initialized with `pca`, so ' \
                'there is no projection model!'
            embedding = self.pca_init.fit_transform(X)

        # Random initialization with isotropic normal distribution
        elif initialization == 'random':
            embedding = np.random.normal(0, 1e-2, (X.shape[0], n_components))

        elif initialization == 'weighted':
            embedding = np.zeros((n_samples, n_components))
            for i in range(n_samples):
                embedding[i] = np.average(self[neighbors[i]], axis=0, weights=distances[i])

        return embedding


class TSNE:
    def __init__(self, n_components=2, perplexity=30, learning_rate=100,
                 early_exaggeration_iter=250, early_exaggeration=12,
                 n_iter=750, late_exaggeration_iter=0, late_exaggeration=1.2,
                 theta=0.5, n_interpolation_points=3, min_num_intervals=10,
                 ints_in_interval=1, initialization='pca', metric='euclidean',
                 initial_momentum=0.5, final_momentum=0.8, n_jobs=1,
                 neighbors='exact', negative_gradient_method='bh',
                 callbacks=None, callbacks_every_iters=50):
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
        self.initial_momentum = initial_momentum
        self.final_momentum = final_momentum
        self.n_jobs = n_jobs
        self.neighbors_method = neighbors
        self.negative_gradient_method = negative_gradient_method

        self.callbacks = callbacks
        self.callbacks_every_iters = callbacks_every_iters

    def fit(self, X: np.ndarray) -> TSNEEmbedding:
        """Perform t-SNE dimensionality reduction.

        Parameters
        ----------
        X : np.ndarray

        Returns
        -------
        TSNEEmbedding

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

    def prepare_initial(self, X, initialization=None):
        """Prepare the initial embedding which can be optimized in steps.

        Parameters
        ----------
        X : np.ndarray
        initialization : Optional[Union[np.ndarray, str]]

        Returns
        -------
        TSNEEmbedding

        """
        # Get some initial coordinates for the embedding
        y_coords = self.generate_initial_coordinates(X, initialization=initialization)

        # If using PCA init, we need to keep the model to project new data
        if isinstance(y_coords, tuple):
            pca_init_model, y_coords = y_coords
        else:
            pca_init_model = None

        # Compute the affinities for the input data
        affinities = NearestNeighborAffinities(
            X, self.perplexity, method=self.neighbors_method,
            metric=self.metric, n_jobs=self.n_jobs,
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

        return TSNEEmbedding(y_coords, affinities, gradient_descent_params,
                             pca_init=pca_init_model)

    def generate_initial_coordinates(self, X, initialization=None):
        """Get initial coordinates for the new embedding for the data set.

        Parameters
        ----------
        X : np.ndarray
        initialization : Optional[Union[np.ndarray, str]]

        Returns
        -------
        np.ndarray

        """
        initialization = initialization or self.initialization

        # If initial positions are given in an array, use a copy of that
        if isinstance(initialization, np.ndarray):
            assert initialization.shape[0] == X.shape[0], \
                'The provided initialization contains a different number of ' \
                'samples (%d) than the data provided (%d).' % (
                    initialization.shape[0], X.shape[0])
            return np.array(initialization)

        # Initialize the embedding using a PCA projection into the desired
        # number of components
        elif initialization == 'pca':
            pca = PCA(n_components=self.n_components)
            return pca, pca.fit_transform(X)

        # Random initialization with isotropic normal distribution
        elif initialization == 'random':
            return np.random.normal(0, 1e-2, (X.shape[0], self.n_components))

        else:
            raise ValueError('Unrecognized initialization scheme')


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
        Joint probability matrix P_{ij}.
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
            callback.optimzation_about_to_start()

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
            error = (error - np.sum(P) * np.log(exaggeration)) / exaggeration

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
