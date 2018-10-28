import logging

import numpy as np
from scipy.sparse import csr_matrix

from . import _tsne
from .nearest_neighbors import BallTree, NNDescent, KNNIndex, VALID_METRICS

log = logging.getLogger(__name__)


class Affinities:
    """Compute the affinities among some initial data and new data.

    t-SNE takes as input an affinity matrix :math:`P`, and does not really care
    about anything else about the data. This means we can use t-SNE for any data
    where we are able to express interactions between samples with an affinity
    matrix.

    Attributes
    ----------
    P: array_like
        The affinity matrix expressing interactions between all data samples.

    """

    def __init__(self):
        self.P = None

    def to_new(self, data):
        """Compute the affinities of new data points to the existing ones.

        This is especially useful for `transform` where we need the conditional
        probabilities from the existing to the new data.

        """


class NearestNeighborAffinities(Affinities):
    """Compute affinities using the nearest neighbors defined by perplexity.

    Parameters
    ----------
    data: np.ndarray
        The data matrix.
    perplexity: float
        Perplexity can be thought of as the continuous :math:`k` number of
        neighbors to consider for each data point. To avoid confusion, note that
        perplexity linearly impacts runtime.
    method: str
        Specifies the nearest neighbor method to use. Can be either ``exact`` or
        ``approx``. ``exact`` uses space partitioning binary trees from
        scikit-learn while ``approx`` makes use of nearest neighbor descent.
        Note that ``approx`` has a bit of overhead and will be slower on smaller
        data sets than exact search.
    metric: str
        The metric to be used to compute affinities between points in the
        original space.
    metric_params: Optional[dict]
        Additional keyword arguments for the metric function.
    symmetrize: bool
        Symmetrize affinity matrix. Standard t-SNE symmetrizes the interactions
        but when embedding new data, symmetrization is not performed.
    n_jobs: int
        The number of jobs to run in parallel. This follows the scikit-learn
        convention, ``-1`` meaning all processors, ``-2`` meaning all but one
        processor and so on.
    random_state: Optional[Union[int, RandomState]]
        The random state parameter follows the convention used in scikit-learn.
        If the value is an int, random_state is the seed used by the random
        number generator. If the value is a RandomState instance, then it will
        be used as the random number generator. If the value is None, the random
        number generator is the RandomState instance used by `np.random`.

    """

    def __init__(self, data, perplexity=30, method='approx', metric='euclidean',
                 metric_params=None, symmetrize=True, n_jobs=1, random_state=None):
        self.n_samples = data.shape[0]

        perplexity = self.check_perplexity(perplexity)
        k_neighbors = min(self.n_samples - 1, int(3 * perplexity))

        # Support shortcuts for built-in nearest neighbor methods
        methods = {'exact': BallTree, 'approx': NNDescent}
        if isinstance(method, KNNIndex):
            knn_index = method

        elif method not in methods:
            raise ValueError('Unrecognized nearest neighbor algorithm `%s`. '
                             'Please choose one of the supported methods or '
                             'provide a valid `KNNIndex` instance.' % method)
        else:
            if metric not in VALID_METRICS:
                raise ValueError('Unrecognized distance metric `%s`. Please '
                                 'choose one of the supported methods.' % metric)
            knn_index = methods[method](metric=metric, metric_params=metric_params,
                                        n_jobs=n_jobs, random_state=random_state)

        knn_index.build(data)
        neighbors, distances = knn_index.query_train(data, k=k_neighbors)

        # Store the results on the object
        self.perplexity = perplexity
        self.knn_index = knn_index
        self.P = joint_probabilities_nn(
            neighbors, distances, perplexity, symmetrize=symmetrize, n_jobs=n_jobs)

        self.n_jobs = n_jobs

    def to_new(self, data, perplexity=None, return_distances=False):
        perplexity = perplexity or self.perplexity
        perplexity = self.check_perplexity(perplexity)
        k_neighbors = min(self.n_samples - 1, int(3 * perplexity))

        neighbors, distances = self.knn_index.query(data, k_neighbors)

        P = joint_probabilities_nn(
            neighbors, distances, perplexity, symmetrize=False,
            n_reference_samples=self.n_samples, n_jobs=self.n_jobs,
        )

        if return_distances:
            return P, neighbors, distances

        return P

    def check_perplexity(self, perplexity):
        """Check for valid perplexity value."""
        if self.n_samples - 1 < 3 * perplexity:
            old_perplexity, perplexity = perplexity, (self.n_samples - 1) / 3
            log.warning('Perplexity value %d is too high. Using perplexity %.2f' %
                        (old_perplexity, perplexity))

        return perplexity


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
    conditional_P = np.asarray(conditional_P)

    P = csr_matrix((conditional_P.ravel(), neighbors.ravel(),
                    range(0, n_samples * k_neighbors + 1, k_neighbors)),
                   shape=(n_samples, n_reference_samples))

    # Symmetrize the probability matrix
    if symmetrize:
        P = (P + P.T) / 2

    # Convert weights to probabilities using pair-wise normalization scheme
    P /= np.sum(P)

    return P
