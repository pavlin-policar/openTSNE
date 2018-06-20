import logging
import time

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA

from tsne import _tsne
from tsne.nearest_neighbors import KNNIndex, KDTree, NNDescent
from tsne.quad_tree import QuadTree

EPSILON = np.finfo(np.float64).eps

log = logging.getLogger(__name__)


class OptimizationInterrupt(KeyboardInterrupt):
    def __init__(self, error: float, final_embedding: np.ndarray) -> None:
        super().__init__()
        self.error = error
        self.final_embedding = final_embedding


class TSNE:
    def __init__(self, n_components=2, perplexity=30, learning_rate=10,
                 early_exaggeration_iter=250, early_exaggeration=12,
                 n_iter=750, late_exaggeration_iter=0, late_exaggeration=1.2,
                 angle=0.5, n_interpolation_points=3, min_num_intervals=10,
                 ints_in_inverval=10, initialization='pca', metric='sqeuclidean',
                 initial_momentum=0.5, final_momentum=0.8, n_jobs=1,
                 neighbors='exact', negative_gradient_method='bh',
                 callback=None, callback_every_iters=50):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.early_exaggeration = early_exaggeration
        self.early_exaggeration_iter = early_exaggeration_iter
        self.n_iter = n_iter
        self.late_exaggeration = late_exaggeration
        self.late_exaggeration_iter = late_exaggeration_iter
        self.angle = angle
        self.n_interpolation_points = n_interpolation_points
        self.min_num_intervals = min_num_intervals
        self.ints_in_inverval = ints_in_inverval
        self.initialization = initialization
        self.metric = metric
        self.initial_momentum = initial_momentum
        self.final_momentum = final_momentum
        self.n_jobs = n_jobs
        self.neighbors_method = neighbors
        self.negative_gradient_method = negative_gradient_method

        if callback is not None and not callable(callback):
            raise ValueError('`callback` must be a callable object!')
        self.use_callback = callback is not None
        self.callback = callback
        self.callback_every_iters = callback_every_iters

    def fit(self, X: np.ndarray, neighbors: np.ndarray = None,
            distances: np.ndarray = None) -> np.ndarray:
        """Perform t-SNE dimensionality reduction.

        Parameters
        ----------
        X : np.ndarray
        neighbors : Optional[np.ndarray]
            Often times, we want to run t-SNE multiple times. If computing the
            nearest neighbors takes a long time, re-using those results for
            different runs will drastically speed up the process. When
            providing precomputed neighbors, be sure to include `distances` as
            well as well as to verify they are of the same shape.
            Contains indices of the k nearest neighbors (or approximate nn).
        distances : Optional[np.ndarray]
            See `neighbors` parameter. Contains the distances to the k nearest
            neighbors (or approximate nn).

        Returns
        -------
        np.ndarray
            The t-SNE embedding.

        """
        n_samples, n_dims = X.shape

        # If no early exaggeration value is proposed, use the update scheme
        # proposed in [2]_ which has some desirable properties
        if self.early_exaggeration is None:
            early_exaggeration = n_samples / 10
            learning_rate = 1
        else:
            early_exaggeration = self.early_exaggeration
            learning_rate = self.learning_rate

        # Check for valid perplexity value
        if n_samples - 1 < 3 * self.perplexity:
            perplexity = (n_samples - 1) / 3
            log.warning('Perplexity value is too high. Using perplexity %.2f' % perplexity)
        else:
            perplexity = self.perplexity

        k_neighbors = min(n_samples - 1, int(3 * perplexity))

        # Find each points' nearest neighbors or use the precomputed ones, if
        # provided
        if neighbors is not None and distances is not None:
            assert neighbors.shape == distances.shape, \
                'The `distances` and `neighbors` dimensions must match exactly!'
            logging.info('Nearest neighbors provided, using precomputed neighbors.')
        else:
            _, neighbors, distances = self.__get_nearest_neighbors(X, k_neighbors)

        start = time.time()
        # Compute the symmetric joint probabilities of points being neighbors
        P = joint_probabilities_nn(neighbors, distances, perplexity, n_jobs=self.n_jobs)
        print('joint probabilities', time.time() - start)

        # Initialize the embedding as a C contigous array for our fast cython
        # implementation
        embedding = self.get_initial_embedding_for(X)
        embedding = np.asarray(embedding, dtype=np.float64, order='C')

        # Degrees of freedom of the Student's t-distribution. The suggestion
        # degrees_of_freedom = n_components - 1 comes from [3]_.
        degrees_of_freedom = max(self.n_components - 1, 1)

        # Optimization
        # Determine which method will be used for optimization
        if callable(self.negative_gradient_method):
            gradient_method = self.negative_gradient_method
        elif self.negative_gradient_method in {'bh', 'BH', 'barnes-hut'}:
            gradient_method = kl_divergence_bh
        elif self.negative_gradient_method in {'fft', 'FFT', 'interpolation'}:
            gradient_method = kl_divergence_fft
        else:
            raise ValueError('Invalid gradient scheme! Please choose one of '
                             'the supported strings or provide a valid callback.')

        callback_params = {'use_callback': self.use_callback,
                           'callback': self.callback,
                           'callback_every_iters': self.callback_every_iters}
        fft_params = {'n_interpolation_points': self.n_interpolation_points,
                      'min_num_intervals': self.min_num_intervals,
                      'ints_in_interval': self.ints_in_inverval}

        try:
            # Early exaggeration with lower momentum to allow points to find more
            # easily move around and find their neighbors
            P *= early_exaggeration
            error, embedding = gradient_descent(
                embedding=embedding, P=P, dof=degrees_of_freedom,
                gradient_method=gradient_method, n_iter=self.early_exaggeration_iter,
                learning_rate=learning_rate, momentum=self.initial_momentum,
                theta=self.angle, n_jobs=self.n_jobs, **callback_params, **fft_params,
            )

            # Restore actual affinity probabilities and increase momentum to get
            # final, optimized embedding
            P /= early_exaggeration
            error, embedding = gradient_descent(
                embedding=embedding, P=P, dof=degrees_of_freedom,
                gradient_method=gradient_method, n_iter=self.n_iter,
                learning_rate=learning_rate, momentum=self.final_momentum,
                theta=self.angle, n_jobs=self.n_jobs, **callback_params, **fft_params,
            )

            # Use the trick described in [4]_ to get more separated clusters of
            # points by applying a late exaggeration phase
            P *= self.late_exaggeration
            error, embedding = gradient_descent(
                embedding=embedding, P=P, dof=degrees_of_freedom,
                gradient_method=gradient_method, n_iter=self.late_exaggeration_iter,
                learning_rate=learning_rate, momentum=self.final_momentum,
                theta=self.angle, n_jobs=self.n_jobs, **callback_params, **fft_params,
            )

        except OptimizationInterrupt as ex:
            log.info('Optimization was interrupted with callback.')
            embedding = ex.final_embedding

        return embedding

    def get_optimizer_for(self, data: np.ndarray):
        n_samples = data.shape[0]

        # Check for valid perplexity value
        if n_samples - 1 < 3 * self.perplexity:
            perplexity = (n_samples - 1) / 3
            log.warning('Perplexity value is too high. Using perplexity %.2f' % perplexity)
        else:
            perplexity = self.perplexity

        k_neighbors = min(n_samples - 1, int(3 * perplexity))
        start = time.time()
        knn_index, neighbors, distances = self.__get_nearest_neighbors(data, k_neighbors)
        print('NN search', time.time() - start)

        start = time.time()
        # Compute the symmetric joint probabilities of points being neighbors
        P = joint_probabilities_nn(neighbors, distances, perplexity, n_jobs=self.n_jobs)
        print('joint probabilities', time.time() - start)

        # Degrees of freedom of the Student's t-distribution. The suggestion
        # degrees_of_freedom = n_components - 1 comes from [3]_.
        degrees_of_freedom = max(self.n_components - 1, 1)

        # Determine which method will be used for optimization
        if callable(self.negative_gradient_method):
            gradient_method = self.negative_gradient_method
        elif self.negative_gradient_method in {'bh', 'BH', 'barnes-hut'}:
            gradient_method = kl_divergence_bh
        elif self.negative_gradient_method in {'fft', 'FFT', 'interpolation'}:
            gradient_method = kl_divergence_fft
        else:
            raise ValueError('Unrecognized gradient method. Please choose one of '
                             'the supported methods or provide a valid callback.')

        gradient_descent_params = {
            'dof': degrees_of_freedom,
            'gradient_method': gradient_method,
            'learning_rate': self.learning_rate,
            # By default, use the momentum used in unexaggerated phase
            'momentum': self.final_momentum,

            # Barnes-Hut params
            'theta': self.angle,
            # Interpolation params
            'n_interpolation_points': self.n_interpolation_points,
            'min_num_intervals': self.min_num_intervals,
            'ints_in_interval': self.ints_in_inverval,

            'n_jobs': self.n_jobs,
            # Callback params
            'use_callback': self.use_callback,
            'callback': self.callback,
            'callback_every_iters': self.callback_every_iters,
        }

        return TSNEOptimizer(
            perplexity=perplexity, knn_index=knn_index, P=P,
            gradient_descent_params=gradient_descent_params,
        )

    def get_initial_embedding_for(self, X: np.ndarray) -> np.ndarray:
        # If initial positions are given in an array, use a copy of that
        if isinstance(self.initialization, np.ndarray):
            return np.array(self.initialization)

        # Initialize the embedding using a PCA projection into the desired
        # number of components
        elif self.initialization == 'pca':
            pca = PCA(n_components=self.n_components)
            return pca.fit_transform(X)

        # Random initialization
        elif self.initialization == 'random':
            return np.random.normal(0, 1e-2, (X.shape[0], self.n_components))

        else:
            raise ValueError('Unrecognized initialization scheme')

    def __get_nearest_neighbors(self, X, k_neighbors):
        """Compute the nearest neighbors for a given dataset.

        Note that the perplexity value must be valid. This is not checked here.

        Parameters
        ----------
        X : np.ndarray
            The reference data against which to search for neighbors.
        k_neighbors: int

        Return
        ------
        KNNIndex
            A nearest neighbor index.
        np.ndarray
            The indices of the k nearest neighbors.
        np.ndarray
            The distances of the k nearest neighbors, corresponding to the
            indices. See above.

        """
        methods = {'exact': KDTree, 'approx': NNDescent}
        if callable(self.neighbors_method):
            assert isinstance(KNNIndex), \
                'The nearest neighbor algorithm you provided does not ' \
                'inherit the `KNNIndex` class!'
            knn_index_cls = self.neighbors_method

        elif self.neighbors_method not in methods:
            raise ValueError('Unrecognized nearest neighbor algorithm `%s`. '
                             'Please choose one of the supported methods or '
                             'provide a valid `KNNIndex` instance.')
        else:
            knn_index_cls = methods[self.neighbors_method]

        knn_index = knn_index_cls(metric=self.metric, n_jobs=self.n_jobs)

        knn_index.build(X)
        neighbors, distances = knn_index.query_train(X, k=k_neighbors)

        return knn_index, neighbors, distances


class TSNEEmbedding(np.ndarray):
    def __new__(cls, embedding):
        obj = np.asarray(embedding, dtype=np.float64, order='C').view(TSNEEmbedding)

        obj.kl_divergence = None
        obj.pBIC = None

        return obj


class TSNEOptimizer:
    def __init__(self, perplexity, knn_index, P, gradient_descent_params):
        self.perplexity = perplexity
        self.knn_index = knn_index
        self.P = P
        self.gradient_descent_params = gradient_descent_params

        self.kl_divergence = None
        self.pBIC = None

    def optimize(self, embedding, n_iter, inplace=False,
                 propagate_exception=False, **gradient_descent_params):
        assert isinstance(embedding, np.ndarray), \
            '`embedding` must be an instance of `np.ndarray`. Got `%s` instead' \
            % type(embedding)

        # Make sure to interpret the embedding as a tSNE embedding
        embedding = embedding.view(TSNEEmbedding)

        # Typically we want to return a new embedding and keep the old one intact
        if not inplace:
            embedding = TSNEEmbedding(embedding)

        # If optimization parameters were passed to this funciton, prefer those
        # over the defaults specified in the TSNE object
        optim_params = dict(self.gradient_descent_params)
        optim_params.update(gradient_descent_params)
        optim_params['n_iter'] = n_iter

        try:
            error, embedding = gradient_descent(
                embedding=embedding, P=self.P, **optim_params)

        except OptimizationInterrupt as ex:
            log.info('Optimization was interrupted with callback.')
            if propagate_exception:
                raise ex
            error, embedding = ex.error, ex.final_embedding

        # Optimization done - time to compute error metrics
        n_samples = embedding.shape[0]
        embedding.kl_divergence = error
        embedding.pBIC = 2 * error + np.log(n_samples) * self.perplexity / n_samples

        return embedding


def joint_probabilities_nn(neighbors, distances, perplexity, symmetrize=True,
                           n_reference_samples=None, n_jobs=1):
    """Compute the conditional probability matrix P_{j|i}.

    This method computes an approximation to P using the nearest neighbors.

    Parameters
    ----------
    neighbors : np.ndarray
        A `n_samples * k_neighbors` matrix containing the indices to each
        points' nearest neighbors in descending order.
    distances : np.ndarray
        A `n_samples * k_neighbors` matrix containing the distances to the
        neighbors at indices defined in the neighbors parameter.
    perplexity : double
        The desired perplexity of the probability distribution.
    symmetrize : bool
        Whether to symmetrize the probability matrix or not. Symmetrizing is
        used for typical t-SNE, but does not make sense when embedding new data
        into an existing embedding.
    n_reference_samples : int
        The number of samples in the existing (reference) embedding. Needed to
        properly construct the sparse P matrix.
    n_jobs : int
        Number of threads.

    Returns
    -------
    csr_matrix
        A `n_samples * n_reference_samples` matrix containing the probabilities
        that a new sample would appear as a neighbor of a reference point.

    """
    n_samples, k_neighbors = distances.shape

    if n_reference_samples is None:
        n_reference_samples = n_samples

    # Compute asymmetric pairwise input similarities
    conditional_P = _tsne.compute_gaussian_perplexity(
        distances, perplexity, num_threads=n_jobs)

    P = csr_matrix((conditional_P.ravel(), neighbors.ravel(),
                    range(0, n_samples * k_neighbors + 1, k_neighbors)),
                   shape=(n_samples, n_reference_samples))

    # Symmetrize the probability matrix
    if symmetrize:
        P = (P + P.T) / 2

    # Convert weights to probabilities using pair-wise normalization scheme
    P /= np.sum(P)

    return P


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

    # In the event that we wish to embed new points into an existing embedding
    # using simple optimization, we compute optimize the new embedding points
    # w.r.t. the existing embedding. Otherwise, we want to optimize the
    # embedding w.r.t. itself
    if reference_embedding is None:
        reference_embedding = embedding

    # Compute negative gradient
    if embedding.ndim == 1 or embedding.shape[1] == 1:
        sum_Q = _tsne.estimate_negative_gradient_fft_1d(
            embedding.ravel(), gradient.ravel(), **fft_params)
    elif embedding.shape[1] == 2:
        sum_Q = _tsne.estimate_negative_gradient_fft_2d(
            embedding, gradient, **fft_params)
    else:
        raise RuntimeError('Interpolation based t-SNE for >2 dimensions is '
                           'currently unsupported (and generally a bad idea)')

    # Compute positive gradient
    sum_P, kl_divergence_ = _tsne.estimate_positive_gradient_nn(
        P.indices, P.indptr, P.data, embedding, reference_embedding, gradient,
        dof, num_threads=n_jobs, should_eval_error=should_eval_error,
    )

    gradient *= 2 * (dof + 1) / dof
    if should_eval_error:
        kl_divergence_ += sum_P * np.log(sum_Q + EPSILON)

    return kl_divergence_, gradient


def gradient_descent(embedding, P, dof, n_iter, gradient_method, learning_rate,
                     momentum, exaggeration=1, min_gain=0.01,
                     min_grad_norm=1e-8, theta=0.5, n_interpolation_points=3,
                     min_num_intervals=10, ints_in_interval=10,
                     reference_embedding=None, n_jobs=1, use_callback=False,
                     callback=None, callback_every_iters=50):
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
    gradient_method : Callable[..., Tuple[float, np.ndarray]]
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
    use_callback : bool
    callback : Callable[[float, np.ndarray] -> bool]
        The callback should accept two parameters, the first is the error of
        the current iteration (the KL divergence), the second is the current
        embedding. The callback should return a boolean value indicating
        whether or not to continue optimization i.e. False to stop.
    callback_every_iters : int
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

    if reference_embedding is None:
        reference_embedding = embedding
        assert isinstance(reference_embedding, np.ndarray), \
            '`reference_embedding` must be an instance of `np.ndarray`. Got ' \
            '`%s` instead' % type(reference_embedding)

    error = 0
    update = np.zeros_like(embedding)
    gains = np.ones_like(embedding)

    bh_params = {'theta': theta}
    fft_params = {'n_interpolation_points': n_interpolation_points,
                  'min_num_intervals': min_num_intervals,
                  'ints_in_interval': ints_in_interval}

    # Lie about the P values for bigger attraction forces
    if exaggeration != 1:
        P *= exaggeration

    for iteration in range(n_iter):
        should_call_callback = use_callback and (iteration + 1) % callback_every_iters == 0
        is_last_iteration = iteration == n_iter - 1
        # We want to provide the error to the callback and a final error if
        # we're at the final iteration, regardless of logging
        should_eval_error = (iteration + 1) % 50 == 0 or should_call_callback or is_last_iteration

        error, gradient = gradient_method(
            embedding, P, dof=dof, bh_params=bh_params, fft_params=fft_params,
            reference_embedding=reference_embedding, n_jobs=n_jobs,
            should_eval_error=should_eval_error,
        )

        if should_eval_error:
            # TODO: Verify this with the KL divergence function
            # Correct the KL divergence w.r.t. the exaggeration
            error = (-np.sum(P) * np.log(exaggeration) + error) / exaggeration

            print('Iteration % 4d, error %.4f' % (iteration + 1, error))

        grad_direction_flipped = np.sign(update) != np.sign(gradient)
        grad_direction_same = np.invert(grad_direction_flipped)
        gains[grad_direction_flipped] += 0.2
        gains[grad_direction_same] = gains[grad_direction_same] * 0.8 + min_gain
        update = momentum * update - learning_rate * gains * gradient
        embedding += update

        # Zero-mean the embedding
        embedding -= np.mean(embedding, axis=0)

        if should_call_callback:
            should_continue = bool(callback(error, embedding))
            if not should_continue:
                # Make sure to un-exaggerate P so it's not corrupted in future runs
                if exaggeration != 1:
                    P /= exaggeration
                raise OptimizationInterrupt(error=error, final_embedding=embedding)

        if np.linalg.norm(gradient) < min_grad_norm:
            log.info('Gradient norm eps reached. Finished.')
            break

    # Make sure to un-exaggerate P so it's not corrupted in future runs
    if exaggeration != 1:
        P /= exaggeration

    return error, embedding


def sqeuclidean(x, y):
    return np.sum((x - y) ** 2)


def kl_divergence(P, embedding):
    """Compute the KL divergence for a given embedding and P_{ij}s."""
    n_samples, n_dims = embedding.shape

    pairwise_distances = np.empty((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            pairwise_distances[i, j] = sqeuclidean(embedding[i], embedding[j])

    Q_unnormalized = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            if i != j:
                Q_unnormalized[i, j] = 1 / (1 + pairwise_distances[i, j])

    sum_Q = np.sum(Q_unnormalized)

    Q = Q_unnormalized / sum_Q

    kl_divergence_ = 0
    for i in range(n_samples):
        for j in range(n_samples):
            if P[i, j] > 0:
                kl_divergence_ += P[i, j] * np.log(P[i, j] / (Q[i, j] + EPSILON))

    return kl_divergence_
