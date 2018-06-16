import logging
import time

import numpy as np
from Orange.data import Domain, ContinuousVariable, Table
from Orange.projection import Projector, Projection
from annoy import AnnoyIndex
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from tsne import _tsne
from .quad_tree import QuadTree

EPSILON = np.finfo(np.float64).eps

log = logging.getLogger(__name__)


class TSNEModel(Projection):
    def __init__(self, existing_data, existing_embedding, metric,
                 perplexity=30, learning_rate=10, exaggeration=4,
                 exaggeration_iter=50, angle=0.5, momentum=0.5,
                 n_jobs=1):
        self.data = existing_data
        self.embedding = existing_embedding
        self.n_components = existing_embedding.X.shape[1]
        self.metric = metric
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.exaggeration = exaggeration
        self.exaggeration_iter = exaggeration_iter
        self.angle = angle
        self.momentum = momentum
        self.n_jobs = n_jobs

    def __call__(self, data: Table, **kwargs) -> Table:
        # If we want to transform new data, ensure that we use correct domain
        if data.domain != self.pre_domain:
            data = data.transform(self.pre_domain)

        embedding = self.transform(data.X, **kwargs)
        return Table(self.embedding.domain, embedding, data.Y, data.metas)

    def transform(self, X, n_iter=200, learning_rate=None, perplexity=None,
                  init='closest', exaggeration=None, exaggeration_iter=None):
        n_new_samples, n_new_dims = X.shape
        n_reference_samples, n_reference_dims = self.data.X.shape

        assert n_new_dims == n_reference_dims, \
            'X dimensions do not match the dimensions of the original data. ' \
            'Expected %d, got %d!' % (n_reference_dims, n_new_dims)

        # Use parameter values that were used during learning if no new ones
        # are specified
        if learning_rate is None:
            learning_rate = self.learning_rate
        if perplexity is None:
            perplexity = self.perplexity
        if exaggeration is None:
            exaggeration = self.exaggeration
        if exaggeration_iter is None:
            exaggeration_iter = self.exaggeration_iter

        # If no early exaggeration value is proposed, use the update scheme
        # proposed in [2]_ which has some desirable properties
        if exaggeration is None:
            exaggeration = n_new_samples / 10
            learning_rate = 1

        # Check that perplexity isn't larger than the possible existing
        # embedding data
        if n_reference_samples < 3 * perplexity:
            perplexity = n_reference_samples / 3
            log.warning('Perplexity value is too high. Using perplexity %.2f'
                        % perplexity)
        else:
            perplexity = perplexity

        k_neighbors = min(n_reference_samples, int(3 * perplexity))

        # Find the nearest neighbors from the new points to the existing points
        start = time.time()
        knn = NearestNeighbors(algorithm='auto', metric=self.metric)
        knn.fit(self.data.X)
        distances, neighbors = knn.kneighbors(X, n_neighbors=k_neighbors)
        del knn
        print('NN search', time.time() - start)

        # Compute the joint probabilities. Don't symmetrize, because P will be
        # a `n_samples * n_new_samples` matrix, which is not necessarily square
        start = time.time()
        P = joint_probabilities(
            distances, neighbors, perplexity, symmetrize=False,
            n_reference_samples=n_reference_samples, n_jobs=self.n_jobs,
        )
        print('joint probabilities', time.time() - start)

        # Initialize the embedding as a C contigous array for our fast cython
        # implementation
        embedding = self.get_initial_embedding_for(X, init, neighbors, distances)
        embedding = np.asarray(embedding, dtype=np.float64, order='C')

        # Degrees of freedom of the Student's t-distribution
        degrees_of_freedom = max(self.n_components - 1, 1)

        # Optimization
        # Early exaggeration with lower momentum to allow points to find more
        # easily move around and find their neighbors
        P *= exaggeration
        embedding = gradient_descent(
            embedding, P, degrees_of_freedom, n_iter=exaggeration_iter,
            learning_rate=learning_rate, momentum=self.momentum,
            theta=self.angle, reference_embedding=self.embedding.X,
            n_jobs=self.n_jobs,
        )

        # Restore actual affinity probabilities and increase momentum to get
        # final, optimized embedding
        P /= exaggeration
        embedding = gradient_descent(
            embedding, P, degrees_of_freedom, n_iter=n_iter,
            learning_rate=learning_rate, momentum=self.momentum,
            theta=self.angle, reference_embedding=self.embedding.X,
            n_jobs=self.n_jobs,
        )

        return embedding

    def get_initial_embedding_for(self, X, method, neighbors, distances):
        n_new_samples = X.shape[0]

        # If initial positions are given in an array, use a copy of that
        if isinstance(method, np.ndarray):
            return np.array(self.init)

        # Use the weighted mean position of the points closes neighbors
        elif method == 'closest':
            embedding = np.zeros((n_new_samples, self.n_components))
            for i in range(n_new_samples):
                embedding[i] = np.average(
                    self.embedding.X[neighbors[i]], axis=0, weights=distances[i])
            return embedding

        # Random initialization
        elif method == 'random':
            return np.random.randn(n_new_samples, self.n_components)

        else:
            raise ValueError('Unrecognized initialization scheme')


class TSNE(Projector):
    name = 't-SNE'

    def __init__(self, n_components=2, perplexity=30, learning_rate=10,
                 early_exaggeration_iter=250, early_exaggeration=12,
                 n_iter=750, late_exaggeration_iter=0, late_exaggeration=1.2,
                 angle=0.5, init='pca', metric='sqeuclidean',
                 initial_momentum=0.5, final_momentum=0.8, n_jobs=1,
                 neighbors='exact', grad='bh', preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.early_exaggeration = early_exaggeration
        self.early_exaggeration_iter = early_exaggeration_iter
        self.n_iter = n_iter
        self.late_exaggeration = late_exaggeration
        self.late_exaggeration_iter = late_exaggeration_iter
        self.angle = angle
        self.init = init
        self.metric = metric
        self.initial_momentum = initial_momentum
        self.final_momentum = final_momentum
        self.n_jobs = n_jobs
        self.neighbors_method = neighbors
        self.gradient_method = grad

    def __call__(self, data: Table) -> TSNEModel:
        data = self.preprocess(data)

        # Create the t-SNE embedding
        embedding = self.fit(data.X, data.Y)

        # Put the embedding into a table
        tsne_cols = [ContinuousVariable('t-SNE-%d' % (i + 1)) for i in range(self.n_components)]
        embedding_domain = Domain(tsne_cols, data.domain.class_vars, data.domain.metas)
        embedding_table = Table(embedding_domain, embedding, data.Y, data.metas)

        # Build the t-SNE model
        model = TSNEModel(
            data, embedding_table, metric=self.metric,
            perplexity=self.perplexity, learning_rate=self.learning_rate,
            exaggeration=self.early_exaggeration, angle=self.angle,
            momentum=self.initial_momentum, n_jobs=self.n_jobs
        )
        model.pre_domain = data.domain
        model.name = '%s (%s)' % (self.name, data.name)
        return model

    def fit(self, X: np.ndarray, Y: np.ndarray = None) -> np.ndarray:
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

        # Need -2: -1 because indexing starts at zero, -1 because neighbor
        # methods, find query point as closest with distance zero
        k_neighbors = min(n_samples - 1, int(3 * perplexity))

        start = time.time()
        # Find each points' nearest neighbors
        if self.neighbors_method == 'exact':
            knn = NearestNeighbors(algorithm='auto', metric=self.metric, n_jobs=self.n_jobs)
            knn.fit(X)
            distances, neighbors = knn.kneighbors(n_neighbors=k_neighbors)
            del knn
        elif self.neighbors_method == 'approx':
            index = AnnoyIndex(n_dims, metric=self.metric)
            for i in range(n_samples):
                index.add_item(i, X[i])
            index.build(50)

            search_neighbors = max(n_samples - 1, k_neighbors + 1)
            neighbors = np.zeros((n_samples, search_neighbors), dtype=int)
            distances = np.zeros((n_samples, search_neighbors), dtype=float)
            for i in range(n_samples):
                neighbors[i], distances[i] = index.get_nns_by_item(
                    i, search_neighbors, include_distances=True)
            neighbors, distances = neighbors[:, 1:], distances[:, 1:]

        print('NN search', time.time() - start)

        start = time.time()
        # Compute the symmetric joint probabilities of points being neighbors
        P = joint_probabilities(distances, neighbors, perplexity, n_jobs=self.n_jobs)
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
        if callable(self.gradient_method):
            gradient_method = self.gradient_method
        elif self.gradient_method in {'bh', 'BH', 'barnes-hut'}:
            gradient_method = kl_divergence_bh
        elif self.gradient_method in {'fft', 'FFT', 'interpolation'}:
            gradient_method = kl_divergence_fft
        else:
            raise ValueError('Invalid gradient scheme! Please choose one of '
                             'the supported strings or provide a valid callback.')

        # Early exaggeration with lower momentum to allow points to find more
        # easily move around and find their neighbors
        P *= early_exaggeration
        embedding = gradient_descent(
            embedding=embedding, P=P, dof=degrees_of_freedom,
            gradient_method=gradient_method, n_iter=self.early_exaggeration_iter,
            learning_rate=learning_rate, momentum=self.initial_momentum,
            theta=self.angle, n_jobs=self.n_jobs,
        )

        # Restore actual affinity probabilities and increase momentum to get
        # final, optimized embedding
        P /= early_exaggeration
        embedding = gradient_descent(
            embedding=embedding, P=P, dof=degrees_of_freedom,
            gradient_method=gradient_method, n_iter=self.n_iter,
            learning_rate=learning_rate, momentum=self.final_momentum,
            theta=self.angle, n_jobs=self.n_jobs,
        )

        # Use the trick described in [4]_ to get more separated clusters of
        # points by applying a late exaggeration phase
        P *= self.late_exaggeration
        embedding = gradient_descent(
            embedding=embedding, P=P, dof=degrees_of_freedom,
            gradient_method=gradient_method, n_iter=self.late_exaggeration_iter,
            learning_rate=learning_rate, momentum=self.final_momentum,
            theta=self.angle, n_jobs=self.n_jobs,
        )

        return embedding

    def get_initial_embedding_for(self, X):
        # If initial positions are given in an array, use a copy of that
        if isinstance(self.init, np.ndarray):
            return np.array(self.init)

        # Initialize the embedding using a PCA projection into the desired
        # number of components
        elif self.init == 'pca':
            pca = PCA(n_components=self.n_components)
            return pca.fit_transform(X)

        # Random initialization
        elif self.init == 'random':
            return np.random.normal(0, 1e-2, (X.shape[0], self.n_components))

        else:
            raise ValueError('Unrecognized initialization scheme')


def joint_probabilities(distances, neighbors, perplexity, symmetrize=True,
                        n_reference_samples=None, n_jobs=1):
    """Compute the conditional probability matrix P_{j|i}.

    This method computes an approximation to P using just the nearest
    neighbors.

    Parameters
    ----------
    distances : np.ndarray
        A `n_samples * k_neighbors` matrix containing the distances to the
        neighbors at indices defined in the neighbors parameter.
    neighbors : np.ndarray
        A `n_samples * k_neighbors` matrix containing the indices to each
        points' nearest neighbors in descending order.
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


def kl_divergence_bh(embedding, P, dof, theta, reference_embedding=None,
                     should_eval_error=False, n_jobs=1, **_):
    """Compute the gradient of the t-SNE objective function D_{KL}(P || Q).

    Parameters
    ----------
    embedding : np.ndarray
        The current embedding Y in the desired space.
    P : csr_matrix
        Joint probability matrix P_{ij}.
    dof : float
        Degrees of freedom of the Student's t-distribution.
    theta : float
        This is the trade-off parameter between speed and accuracy of the
        Barnes-Hut approximation of the negative forces. Setting a lower value
        will produce more accurate results, while setting a higher value will
        search through less of the space providing a rougher approximation.
        Scikit-learn recommends values between 0.2-0.8.
    reference_embedding : Optional[np.ndarray]
        If we are adding points to an existing embedding, we have to compute
        the gradients and errors w.r.t. the existing embedding.
    should_eval_error : bool
        Evaluating the KL divergence error at every iteration severely impacts
        performance.
    n_jobs : int
        Number of threads.

    Returns
    -------
    float
        KL divergence if the `should_eval_error` flag is set, otherwise 0.
    np.ndarray
        The gradient of the error

    """
    gradient = np.zeros_like(embedding, dtype=np.float64, order='C')

    # In the event that we wish to embed new points into an existing embedding
    # using simple optimization, we compute optimize the new embedding points
    # w.r.t. the existing embedding. Otherwise, we want to optimize the
    # embedding w.r.t. itself
    if reference_embedding is None:
        reference_embedding = embedding

    # Compute negative gradient
    tree = QuadTree(reference_embedding)
    sum_Q = _tsne.compute_negative_gradients_bh(
        tree, embedding, gradient, theta=theta, dof=dof, num_threads=n_jobs)
    del tree

    # Compute positive gradient
    sum_P, kl_divergence_ = _tsne.compute_positive_gradients(
        P.indices, P.indptr, P.data, embedding, reference_embedding, gradient,
        dof, num_threads=n_jobs, should_eval_error=should_eval_error,
    )

    gradient *= 2 * (dof + 1) / dof
    # Computing positive gradients summed up only unnormalized q_ijs, so we
    # have to include normalziation term separately
    if should_eval_error:
        kl_divergence_ += sum_P * np.log(sum_Q + EPSILON)

    return kl_divergence_, gradient


def kl_divergence_fft(embedding, P, dof, reference_embedding=None,
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
        sum_Q = _tsne.compute_negative_gradients_fft_1d(embedding, gradient)
    elif embedding.shape[1] == 2:
        sum_Q = _tsne.compute_negative_gradients_fft_2d(embedding, gradient)
    else:
        raise RuntimeError('Interpolation based t-SNE for >2 dimensions is '
                           'currently unsupported (and generally a bad idea)')

    # Compute positive gradient
    sum_P, kl_divergence_ = _tsne.compute_positive_gradients(
        P.indices, P.indptr, P.data, embedding, reference_embedding, gradient,
        dof, num_threads=n_jobs, should_eval_error=should_eval_error,
    )

    gradient *= 2 * (dof + 1) / dof
    if should_eval_error:
        kl_divergence_ += sum_P * np.log(sum_Q + EPSILON)

    return kl_divergence_, gradient


def gradient_descent(embedding, P, dof, n_iter, gradient_method, learning_rate,
                     momentum, min_gain=0.01, min_grad_norm=1e-8, theta=0.5,
                     reference_embedding=None, n_jobs=1):
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
        Scikit-learn recommends values between 0.2-0.8.
    reference_embedding : Optional[np.ndarray]
        If we are adding points to an existing embedding, we have to compute
        the gradients and errors w.r.t. the existing embedding.
    n_jobs : int
        Number of threads.

    Returns
    -------
    np.ndarray
        The optimized embedding Y.

    """
    assert isinstance(embedding, np.ndarray), \
        '`embedding` must be an instance of `np.ndarray`. Got `%s` instead' \
        % type(embedding)

    if reference_embedding is None:
        reference_embedding = embedding
        assert isinstance(reference_embedding, np.ndarray), \
            '`reference_embedding` must be an instance of `np.ndarray`. Got ' \
            '`%s` instead' % type(reference_embedding)

    update = np.zeros_like(embedding)
    gains = np.ones_like(embedding)

    for iteration in range(n_iter):
        should_eval_error = (iteration + 1) % 50 == 0

        error, gradient = kl_divergence_fft(
            embedding, P, dof, reference_embedding, n_jobs=n_jobs,
            should_eval_error=should_eval_error,
        )
        error1, gradient = kl_divergence_bh(embedding, P, dof=dof, theta=theta, should_eval_error=True)
        print(error, error1)

        if should_eval_error:
            print('Iteration % 4d, error %.4f' % (iteration + 1, error))

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

    return embedding


def sqeuclidean(x, y):
    return np.sum((x - y) ** 2)


def kl_divergence(P, embedding):
    n_samples, n_dims = embedding.shape

    pairwise_distances = np.empty((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            pairwise_distances[i, j] = sqeuclidean(embedding[i], embedding[j])

    Q = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            if i != j:
                Q[i, j] = 1 / (1 + pairwise_distances[i, j])

    sum_Q = np.sum(Q)

    Q_normalized = Q / sum_Q

    kl_divergence_ = 0
    for i in range(n_samples):
        for j in range(n_samples):
            if P[i, j] > 0:
                # kl_divergence_ += P[i, j] * np.log(P[i, j] / (Q_normalized[i, j] + EPSILON)
                kl_divergence_ += P[i, j] * np.log(P[i, j] / (Q[i, j] + EPSILON))

    kl_divergence_ += np.sum(P) * np.log(sum_Q + EPSILON)

    return kl_divergence_
