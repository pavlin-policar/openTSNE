import logging

import numpy as np
from scipy.sparse import csr_matrix

from fastTSNE import _tsne
from fastTSNE.nearest_neighbors import KDTree, NNDescent, KNNIndex

try:
    import networkx as nx
except ImportError:
    nx = None

log = logging.getLogger(__name__)


class Affinities:
    """Compute the affinities among some initial data and new data.

    tSNE takes as input an affinity matrix P, and does not really care about
    the space in which the original data points lie. This means we are not
    limited to problems with numeric matrices (although that is the most common
    use-case) but can also optimize graph layouts.

    We use perplexity, as defined by Van der Maaten in the original paper as a
    continuous analogue to the number of neighbor affinities we want to
    preserve during optimization.

    """
    def __init__(self, perplexity=30):
        self.perplexity = perplexity
        self.P = None

    def to_new(self, data, perplexity=None, return_distances=False):
        """Compute the affinities of new data points to the existing ones.

        This is especially useful for `transform` where we need the conditional
        probabilities from the existing to the new data.

        """


class NearestNeighborAffinities(Affinities):
    """Compute affinities using the nearest neighbors defined by perplexity."""
    def __init__(self, data, perplexity=30, method='approx', metric='euclidean',
                 symmetrize=True, n_jobs=1):
        self.n_samples = data.shape[0]

        perplexity = self.check_perplexity(perplexity)
        k_neighbors = min(self.n_samples - 1, int(3 * perplexity))

        # Support shortcuts for built-in nearest neighbor methods
        methods = {'exact': KDTree, 'approx': NNDescent}
        if isinstance(method, KNNIndex):
            knn_index = method

        elif method not in methods:
            raise ValueError('Unrecognized nearest neighbor algorithm `%s`. '
                             'Please choose one of the supported methods or '
                             'provide a valid `KNNIndex` instance.')
        else:
            knn_index = methods[method](metric=metric, n_jobs=n_jobs)

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


if nx is not None:
    class NxGraphAffinities(Affinities):
        def __init__(self, graph, use_directed=True, use_weights=True):
            super().__init__()

            xs, ys = list(zip(*graph.edges))
            xs = np.asarray(xs).astype(int)
            ys = np.asarray(ys).astype(int)
            self.P = csr_matrix((np.ones_like(xs), (xs, ys)), dtype=float)

            if not use_directed:
                self.P += self.P.T

            self.P /= self.P.sum()

        def to_new(self, data):
            pass
