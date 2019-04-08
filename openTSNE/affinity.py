import logging
import operator
from functools import reduce

import numpy as np
import scipy.sparse as sp

from . import _tsne
from . import nearest_neighbors

log = logging.getLogger(__name__)


class Affinities:
    """Compute the affinities between samples.

    t-SNE takes as input an affinity matrix :math:`P`, and does not really care
    about anything else from the data. This means we can use t-SNE for any data
    where we are able to express interactions between samples with an affinity
    matrix.

    Attributes
    ----------
    P: array_like
        The :math:`N \\times N` affinity matrix expressing interactions between
        :math:`N` initial data samples.

    """

    def __init__(self):
        self.P = None

    def to_new(self, data, return_distances=False):
        """Compute the affinities of new samples to the initial samples.

        This is necessary for embedding new data points into an existing
        embedding.

        Parameters
        ----------
        data: np.ndarray
            The data points to be added to the existing embedding.

        return_distances: bool
            If needed, the function can return the indices of the nearest
            neighbors and their corresponding distances.

        Returns
        -------
        P: array_like
            An :math:`N \\times M` affinity matrix expressing interactions
            between :math:`N` new data points the initial :math:`M` data
            samples.

        indices: np.ndarray
            Returned if ``return_distances=True``. The indices of the :math:`k`
            nearest neighbors in the existing embedding for every new data
            point.

        distances: np.ndarray
            Returned if ``return_distances=True``. The distances to the
            :math:`k` nearest neighbors in the existing embedding for every new
            data point.

        """


class PerplexityBasedNN(Affinities):
    """Compute affinities using nearest neighbors.

    Please see the :ref:`parameter-guide` for more information.

    Parameters
    ----------
    data: np.ndarray
        The data matrix.

    perplexity: float
        Perplexity can be thought of as the continuous :math:`k` number of
        nearest neighbors, for which t-SNE will attempt to preserve distances.

    method: str
        Specifies the nearest neighbor method to use. Can be either ``exact`` or
        ``approx``.

    metric: str
        The metric to be used to compute affinities between points in the
        original space.

    metric_params: dict
        Additional keyword arguments for the metric function.

    symmetrize: bool
        Symmetrize affinity matrix. Standard t-SNE symmetrizes the interactions
        but when embedding new data, symmetrization is not performed.

    n_jobs: int
        The number of threads to use while running t-SNE. This follows the
        scikit-learn convention, ``-1`` meaning all processors, ``-2`` meaning
        all but one, etc.

    random_state: Union[int, RandomState]
        If the value is an int, random_state is the seed used by the random
        number generator. If the value is a RandomState instance, then it will
        be used as the random number generator. If the value is None, the random
        number generator is the RandomState instance used by `np.random`.

    """

    def __init__(
        self,
        data,
        perplexity=30,
        method="approx",
        metric="euclidean",
        metric_params=None,
        symmetrize=True,
        n_jobs=1,
        random_state=None,
    ):
        self.n_samples = data.shape[0]
        self.perplexity = self.check_perplexity(perplexity)

        k_neighbors = min(self.n_samples - 1, int(3 * self.perplexity))
        self.knn_index, self.__neighbors, self.__distances = build_knn_index(
            data, method, k_neighbors, metric, metric_params, n_jobs, random_state
        )

        self.P = joint_probabilities_nn(
            self.__neighbors,
            self.__distances,
            [self.perplexity],
            symmetrize=symmetrize,
            n_jobs=n_jobs,
        )

        self.n_jobs = n_jobs

    def set_perplexity(self, new_perplexity):
        """Change the perplexity of the affinity matrix.

        Note that we only allow lowering the perplexity or restoring it to its
        original value. This restriction exists because setting a higher
        perplexity value requires recomputing all the nearest neighbors, which
        can take a long time. To avoid potential confusion as to why execution
        time is slow, this is not allowed. If you would like to increase the
        perplexity above the initial value, simply create a new instance.

        Parameters
        ----------
        new_perplexity: float
            The new perplexity.

        """
        # If the value hasn't changed, there's nothing to do
        if new_perplexity == self.perplexity:
            return
        # Verify that the perplexity isn't too large
        new_perplexity = self.check_perplexity(new_perplexity)
        # Recompute the affinity matrix
        k_neighbors = min(self.n_samples - 1, int(3 * new_perplexity))
        if k_neighbors > self.__neighbors.shape[1]:
            raise RuntimeError(
                "The desired perplexity `%.2f` is larger than the initial one "
                "used. This would need to recompute the nearest neighbors, "
                "which is not efficient. Please create a new `%s` instance "
                "with the increased perplexity."
                % (new_perplexity, self.__class__.__name__)
            )

        self.perplexity = new_perplexity
        self.P = joint_probabilities_nn(
            self.__neighbors[:, :k_neighbors],
            self.__distances[:, :k_neighbors],
            [self.perplexity],
            symmetrize=True,
            n_jobs=self.n_jobs,
        )

    def to_new(self, data, perplexity=None, return_distances=False):
        """Compute the affinities of new samples to the initial samples.

        This is necessary for embedding new data points into an existing
        embedding.

        Please see the :ref:`parameter-guide` for more information.

        Parameters
        ----------
        data: np.ndarray
            The data points to be added to the existing embedding.

        perplexity: float
            Perplexity can be thought of as the continuous :math:`k` number of
            nearest neighbors, for which t-SNE will attempt to preserve
            distances.

        return_distances: bool
            If needed, the function can return the indices of the nearest
            neighbors and their corresponding distances.

        Returns
        -------
        P: array_like
            An :math:`N \\times M` affinity matrix expressing interactions
            between :math:`N` new data points the initial :math:`M` data
            samples.

        indices: np.ndarray
            Returned if ``return_distances=True``. The indices of the :math:`k`
            nearest neighbors in the existing embedding for every new data
            point.

        distances: np.ndarray
            Returned if ``return_distances=True``. The distances to the
            :math:`k` nearest neighbors in the existing embedding for every new
            data point.

        """
        perplexity = perplexity if perplexity is not None else self.perplexity
        perplexity = self.check_perplexity(perplexity)
        k_neighbors = min(self.n_samples - 1, int(3 * perplexity))

        neighbors, distances = self.knn_index.query(data, k_neighbors)

        P = joint_probabilities_nn(
            neighbors,
            distances,
            [perplexity],
            symmetrize=False,
            normalization="point-wise",
            n_reference_samples=self.n_samples,
            n_jobs=self.n_jobs,
        )

        if return_distances:
            return P, neighbors, distances

        return P

    def check_perplexity(self, perplexity):
        if perplexity <= 0:
            raise ValueError("Perplexity must be >=0. %.2f given" % perplexity)

        if self.n_samples - 1 < 3 * perplexity:
            old_perplexity, perplexity = perplexity, (self.n_samples - 1) / 3
            log.warning(
                "Perplexity value %d is too high. Using perplexity %.2f "
                "instead" % (old_perplexity, perplexity)
            )

        return perplexity


def build_knn_index(
    data, method, k, metric, metric_params=None, n_jobs=1, random_state=None
):
    methods = {
        "exact": nearest_neighbors.BallTree,
        "approx": nearest_neighbors.NNDescent,
    }
    if isinstance(method, nearest_neighbors.KNNIndex):
        knn_index = method

    elif method not in methods:
        raise ValueError(
            "Unrecognized nearest neighbor algorithm `%s`. Please choose one "
            "of the supported methods or provide a valid `KNNIndex` instance."
            % method
        )
    else:
        knn_index = methods[method](
            metric=metric,
            metric_params=metric_params,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    neighbors, distances = knn_index.build(data, k=k)

    return knn_index, neighbors, distances


def joint_probabilities_nn(
    neighbors,
    distances,
    perplexities,
    symmetrize=True,
    normalization="pair-wise",
    n_reference_samples=None,
    n_jobs=1,
):
    """Compute the conditional probability matrix P_{j|i}.

    This method computes an approximation to P using the nearest neighbors.

    Parameters
    ----------
    neighbors: np.ndarray
        A `n_samples * k_neighbors` matrix containing the indices to each
        points" nearest neighbors in descending order.
    distances: np.ndarray
        A `n_samples * k_neighbors` matrix containing the distances to the
        neighbors at indices defined in the neighbors parameter.
    perplexities: double
        The desired perplexity of the probability distribution.
    symmetrize: bool
        Whether to symmetrize the probability matrix or not. Symmetrizing is
        used for typical t-SNE, but does not make sense when embedding new data
        into an existing embedding.
    normalization: str
        The normalization scheme to use for the affinities. Standard t-SNE
        considers interactions between all the data points, therefore the entire
        affinity matrix is regarded as a probability distribution, and must sum
        to 1. When embedding new points, we only consider interactions to
        existing points, and treat each point separately. In this case, we
        row-normalize the affinity matrix, meaning each point gets its own
        probability distribution.
    n_reference_samples: int
        The number of samples in the existing (reference) embedding. Needed to
        properly construct the sparse P matrix.
    n_jobs: int
        Number of threads.

    Returns
    -------
    csr_matrix
        A `n_samples * n_reference_samples` matrix containing the probabilities
        that a new sample would appear as a neighbor of a reference point.

    """
    assert normalization in (
        "pair-wise",
        "point-wise",
    ), f"Unrecognized normalization scheme `{normalization}`."

    n_samples, k_neighbors = distances.shape

    if n_reference_samples is None:
        n_reference_samples = n_samples

    # Compute asymmetric pairwise input similarities
    conditional_P = _tsne.compute_gaussian_perplexity(
        distances, np.array(perplexities, dtype=float), num_threads=n_jobs
    )
    conditional_P = np.asarray(conditional_P)

    P = sp.csr_matrix(
        (
            conditional_P.ravel(),
            neighbors.ravel(),
            range(0, n_samples * k_neighbors + 1, k_neighbors),
        ),
        shape=(n_samples, n_reference_samples),
    )

    # Symmetrize the probability matrix
    if symmetrize:
        P = (P + P.T) / 2

    if normalization == "pair-wise":
        P /= np.sum(P)
    elif normalization == "point-wise":
        P = sp.diags(np.asarray(1 / P.sum(axis=1)).ravel()) @ P

    return P


class FixedSigmaNN(Affinities):
    """Compute affinities using using nearest neighbors and a fixed bandwidth
    for the Gaussians in the ambient space.

    Using a fixed Gaussian bandwidth can enable us to find smaller clusters of
    data points than we might be able to using the automatically determined
    bandwidths using perplexity. Note however that this requires mostly trial
    and error.

    Parameters
    ----------
    data: np.ndarray
        The data matrix.

    sigma: float
        The bandwidth to use for the Gaussian kernels in the ambient space.

    k: int
        The number of nearest neighbors to consider for each kernel.

    method: str
        Specifies the nearest neighbor method to use. Can be either ``exact`` or
        ``approx``.

    metric: str
        The metric to be used to compute affinities between points in the
        original space.

    metric_params: dict
        Additional keyword arguments for the metric function.

    symmetrize: bool
        Symmetrize affinity matrix. Standard t-SNE symmetrizes the interactions
        but when embedding new data, symmetrization is not performed.

    n_jobs: int
        The number of threads to use while running t-SNE. This follows the
        scikit-learn convention, ``-1`` meaning all processors, ``-2`` meaning
        all but one, etc.

    random_state: Union[int, RandomState]
        If the value is an int, random_state is the seed used by the random
        number generator. If the value is a RandomState instance, then it will
        be used as the random number generator. If the value is None, the random
        number generator is the RandomState instance used by `np.random`.

    """

    def __init__(
        self,
        data,
        sigma,
        k=30,
        method="approx",
        metric="euclidean",
        metric_params=None,
        symmetrize=True,
        n_jobs=1,
        random_state=None,
    ):
        self.n_samples = n_samples = data.shape[0]

        if k >= self.n_samples:
            raise ValueError(
                "`k` (%d) cannot be larger than N-1 (%d)." % (k, self.n_samples)
            )

        knn_index, neighbors, distances = build_knn_index(
            data, method, k, metric, metric_params, n_jobs, random_state
        )

        self.knn_index = knn_index

        # Compute asymmetric pairwise input similarities
        conditional_P = np.exp(-distances ** 2 / (2 * sigma ** 2))
        conditional_P /= np.sum(conditional_P, axis=1)[:, np.newaxis]

        P = sp.csr_matrix(
            (conditional_P.ravel(), neighbors.ravel(), range(0, n_samples * k + 1, k)),
            shape=(n_samples, n_samples),
        )

        # Symmetrize the probability matrix
        if symmetrize:
            P = (P + P.T) / 2

        # Convert weights to probabilities
        P /= np.sum(P)

        self.sigma = sigma
        self.k = k
        self.P = P
        self.n_jobs = n_jobs

    def to_new(self, data, k=None, sigma=None, return_distances=False):
        """Compute the affinities of new samples to the initial samples.

        This is necessary for embedding new data points into an existing
        embedding.

        Parameters
        ----------
        data: np.ndarray
            The data points to be added to the existing embedding.

        k: int
            The number of nearest neighbors to consider for each kernel.

        sigma: float
            The bandwidth to use for the Gaussian kernels in the ambient space.

        return_distances: bool
            If needed, the function can return the indices of the nearest
            neighbors and their corresponding distances.

        Returns
        -------
        P: array_like
            An :math:`N \\times M` affinity matrix expressing interactions
            between :math:`N` new data points the initial :math:`M` data
            samples.

        indices: np.ndarray
            Returned if ``return_distances=True``. The indices of the :math:`k`
            nearest neighbors in the existing embedding for every new data
            point.

        distances: np.ndarray
            Returned if ``return_distances=True``. The distances to the
            :math:`k` nearest neighbors in the existing embedding for every new
            data point.

        """
        n_samples = data.shape[0]
        n_reference_samples = self.n_samples

        if k is None:
            k = self.k
        elif k >= n_reference_samples:
            raise ValueError(
                "`k` (%d) cannot be larger than the number of reference "
                "samples (%d)." % (k, self.n_samples)
            )

        if sigma is None:
            sigma = self.sigma

        # Find nearest neighbors and the distances to the new points
        neighbors, distances = self.knn_index.query(data, k)

        # Compute asymmetric pairwise input similarities
        conditional_P = np.exp(-distances ** 2 / (2 * sigma ** 2))

        # Convert weights to probabilities
        conditional_P /= np.sum(conditional_P, axis=1)[:, np.newaxis]

        P = sp.csr_matrix(
            (conditional_P.ravel(), neighbors.ravel(), range(0, n_samples * k + 1, k)),
            shape=(n_samples, n_reference_samples),
        )

        if return_distances:
            return P, neighbors, distances

        return P


class MultiscaleMixture(Affinities):
    """Calculate affinities using a Gaussian mixture kernel.

    Instead of using a single perplexity to compute the affinities between data
    points, we can use a multiscale Gaussian kernel instead. This allows us to
    incorporate long range interactions.

    Please see the :ref:`parameter-guide` for more information.

    Parameters
    ----------
    data: np.ndarray
        The data matrix.

    perplexities: List[float]
        A list of perplexity values, which will be used in the multiscale
        Gaussian kernel. Perplexity can be thought of as the continuous
        :math:`k` number of nearest neighbors, for which t-SNE will attempt to
        preserve distances.

    method: str
        Specifies the nearest neighbor method to use. Can be either ``exact`` or
        ``approx``.

    metric: str
        The metric to be used to compute affinities between points in the
        original space.

    metric_params: dict
        Additional keyword arguments for the metric function.

    symmetrize: bool
        Symmetrize affinity matrix. Standard t-SNE symmetrizes the interactions
        but when embedding new data, symmetrization is not performed.

    n_jobs: int
        The number of threads to use while running t-SNE. This follows the
        scikit-learn convention, ``-1`` meaning all processors, ``-2`` meaning
        all but one, etc.

    random_state: Union[int, RandomState]
        If the value is an int, random_state is the seed used by the random
        number generator. If the value is a RandomState instance, then it will
        be used as the random number generator. If the value is None, the random
        number generator is the RandomState instance used by `np.random`.

    """

    def __init__(
        self,
        data,
        perplexities,
        method="approx",
        metric="euclidean",
        metric_params=None,
        symmetrize=True,
        n_jobs=1,
        random_state=None,
    ):
        self.n_samples = data.shape[0]

        # We will compute the nearest neighbors to the max value of perplexity,
        # smaller values can just use indexing to truncate unneeded neighbors
        perplexities = self.check_perplexities(perplexities)
        max_perplexity = np.max(perplexities)
        k_neighbors = min(self.n_samples - 1, int(3 * max_perplexity))

        self.knn_index, self.__neighbors, self.__distances = build_knn_index(
            data, method, k_neighbors, metric, metric_params, n_jobs, random_state
        )

        self.P = self._calculate_P(
            self.__neighbors,
            self.__distances,
            perplexities,
            symmetrize=symmetrize,
            n_jobs=n_jobs,
        )

        self.perplexities = perplexities
        self.n_jobs = n_jobs

    @staticmethod
    def _calculate_P(
        neighbors,
        distances,
        perplexities,
        symmetrize=True,
        normalization="pair-wise",
        n_reference_samples=None,
        n_jobs=1,
    ):
        return joint_probabilities_nn(
            neighbors,
            distances,
            perplexities,
            symmetrize=symmetrize,
            normalization=normalization,
            n_reference_samples=n_reference_samples,
            n_jobs=n_jobs,
        )

    def set_perplexities(self, new_perplexities):
        """Change the perplexities of the affinity matrix.

        Note that we only allow lowering the perplexities or restoring them to
        their original maximum value. This restriction exists because setting a
        higher perplexity value requires recomputing all the nearest neighbors,
        which can take a long time. To avoid potential confusion as to why
        execution time is slow, this is not allowed. If you would like to
        increase the perplexity above the initial value, simply create a new
        instance.

        Parameters
        ----------
        new_perplexities: List[float]
            The new list of perplexities.

        """
        if np.array_equal(self.perplexities, new_perplexities):
            return

        new_perplexities = self.check_perplexities(new_perplexities)
        max_perplexity = np.max(new_perplexities)
        k_neighbors = min(self.n_samples - 1, int(3 * max_perplexity))

        if k_neighbors > self.__neighbors.shape[1]:
            raise RuntimeError(
                "The largest perplexity `%.2f` is larger than the initial one "
                "used. This would need to recompute the nearest neighbors, "
                "which is not efficient. Please create a new `%s` instance "
                "with the increased perplexity."
                % (max_perplexity, self.__class__.__name__)
            )

        self.perplexities = new_perplexities
        self.P = self._calculate_P(
            self.__neighbors[:, :k_neighbors],
            self.__distances[:, :k_neighbors],
            self.perplexities,
            symmetrize=True,
            n_jobs=self.n_jobs,
        )

    def to_new(self, data, perplexities=None, return_distances=False):
        """Compute the affinities of new samples to the initial samples.

        This is necessary for embedding new data points into an existing
        embedding.

        Please see the :ref:`parameter-guide` for more information.

        Parameters
        ----------
        data: np.ndarray
            The data points to be added to the existing embedding.

        perplexities: List[float]
            A list of perplexity values, which will be used in the multiscale
            Gaussian kernel. Perplexity can be thought of as the continuous
            :math:`k` number of nearest neighbors, for which t-SNE will attempt
            to preserve distances.

        return_distances: bool
            If needed, the function can return the indices of the nearest
            neighbors and their corresponding distances.

        Returns
        -------
        P: array_like
            An :math:`N \\times M` affinity matrix expressing interactions
            between :math:`N` new data points the initial :math:`M` data
            samples.

        indices: np.ndarray
            Returned if ``return_distances=True``. The indices of the :math:`k`
            nearest neighbors in the existing embedding for every new data
            point.

        distances: np.ndarray
            Returned if ``return_distances=True``. The distances to the
            :math:`k` nearest neighbors in the existing embedding for every new
            data point.

        """
        perplexities = perplexities if perplexities is not None else self.perplexities
        perplexities = self.check_perplexities(perplexities)

        max_perplexity = np.max(perplexities)
        k_neighbors = min(self.n_samples - 1, int(3 * max_perplexity))

        neighbors, distances = self.knn_index.query(data, k_neighbors)

        P = self._calculate_P(
            neighbors,
            distances,
            perplexities,
            symmetrize=False,
            normalization="point-wise",
            n_reference_samples=self.n_samples,
            n_jobs=self.n_jobs,
        )

        if return_distances:
            return P, neighbors, distances

        return P

    def check_perplexities(self, perplexities):
        """Check and correct/truncate perplexities.

        If a perplexity is too large, it is corrected to the largest allowed
        value. It is then inserted into the list of perplexities only if that
        value doesn't already exist in the list.

        """
        usable_perplexities = []
        for perplexity in sorted(perplexities):
            if 3 * perplexity > self.n_samples - 1:
                new_perplexity = (self.n_samples - 1) / 3

                if new_perplexity in usable_perplexities:
                    log.warning(
                        "Perplexity value %d is too high. Dropping "
                        "because the max perplexity is already in the "
                        "list." % perplexity
                    )
                else:
                    usable_perplexities.append(new_perplexity)
                    log.warning(
                        "Perplexity value %d is too high. Using "
                        "perplexity %.2f instead" % (perplexity, new_perplexity)
                    )
            else:
                usable_perplexities.append(perplexity)

        return usable_perplexities


class Multiscale(MultiscaleMixture):
    """Calculate affinities using averaged Gaussian perplexities.

    In contrast to :class:`MultiscaleMixture`, which uses a Gaussian mixture
    kernel, here, we first compute single scale Gaussian kernels, convert them
    to probability distributions, then average them out between scales.

    Please see the :ref:`parameter-guide` for more information.

    Parameters
    ----------
    data: np.ndarray
        The data matrix.

    perplexities: List[float]
        A list of perplexity values, which will be used in the multiscale
        Gaussian kernel. Perplexity can be thought of as the continuous
        :math:`k` number of nearest neighbors, for which t-SNE will attempt to
        preserve distances.

    method: str
        Specifies the nearest neighbor method to use. Can be either ``exact`` or
        ``approx``.

    metric: str
        The metric to be used to compute affinities between points in the
        original space.

    metric_params: dict
        Additional keyword arguments for the metric function.

    symmetrize: bool
        Symmetrize affinity matrix. Standard t-SNE symmetrizes the interactions
        but when embedding new data, symmetrization is not performed.

    n_jobs: int
        The number of threads to use while running t-SNE. This follows the
        scikit-learn convention, ``-1`` meaning all processors, ``-2`` meaning
        all but one, etc.

    random_state: Union[int, RandomState]
        If the value is an int, random_state is the seed used by the random
        number generator. If the value is a RandomState instance, then it will
        be used as the random number generator. If the value is None, the random
        number generator is the RandomState instance used by `np.random`.

    """

    @staticmethod
    def _calculate_P(
        neighbors,
        distances,
        perplexities,
        symmetrize=True,
        normalization="pair-wise",
        n_reference_samples=None,
        n_jobs=1,
    ):
        # Compute normalized probabilities for each perplexity
        partial_Ps = [
            joint_probabilities_nn(
                neighbors,
                distances,
                [perplexity],
                symmetrize=symmetrize,
                normalization=normalization,
                n_reference_samples=n_reference_samples,
                n_jobs=n_jobs,
            )
            for perplexity in perplexities
        ]
        # Sum them together, then normalize
        P = reduce(operator.add, partial_Ps, 0)

        # Take care to properly normalize the affinity matrix
        if normalization == "pair-wise":
            P /= np.sum(P)
        elif normalization == "point-wise":
            P = sp.diags(np.asarray(1 / P.sum(axis=1)).ravel()) @ P

        return P
