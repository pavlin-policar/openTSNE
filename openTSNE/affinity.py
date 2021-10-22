import logging
import operator
from functools import reduce

import numpy as np
import scipy.sparse as sp

from openTSNE import _tsne
from openTSNE import nearest_neighbors
from openTSNE import utils
from openTSNE.utils import is_package_installed

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

    verbose: bool

    """

    def __init__(self, verbose=False):
        self.P = None
        self.verbose = verbose
        self.knn_index: nearest_neighbors.KNNIndex = None

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

    @property
    def n_samples(self):
        if self.knn_index is None:
            raise RuntimeError("`knn_index` is not set!")
        return self.knn_index.n_samples


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
        Specifies the nearest neighbor method to use. Can be ``exact``, ``annoy``,
        ``pynndescent``, ``hnsw``, ``approx``, or ``auto`` (default). ``approx`` uses Annoy
        if the input data matrix is not a sparse object and if Annoy supports
        the given metric. Otherwise it uses Pynndescent. ``auto`` uses exact
        nearest neighbors for N<1000 and the same heuristic as ``approx`` for N>=1000.

    metric: Union[str, Callable]
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

    verbose: bool

    k_neighbors: int or ``auto``
        The number of neighbors to use in the kNN graph. If ``auto`` (default),
        it is set to three times the perplexity.

    knn_index: Optional[nearest_neighbors.KNNIndex]
        Optionally, a precomptued ``openTSNE.nearest_neighbors.KNNIndex`` object
        can be specified. This option will ignore any KNN-related parameters.
        When ``knn_index`` is specified, ``data`` must be set to None.

    """

    def __init__(
        self,
        data=None,
        perplexity=30,
        method="auto",
        metric="euclidean",
        metric_params=None,
        symmetrize=True,
        n_jobs=1,
        random_state=None,
        verbose=False,
        k_neighbors="auto",
        knn_index=None,
    ):
        # This can't work if neither data nor the knn index are specified
        if data is None and knn_index is None:
            raise ValueError(
                "At least one of the parameters `data` or `knn_index` must be specified!"
            )
        # This can't work if both data and the knn index are specified
        if data is not None and knn_index is not None:
            raise ValueError(
                "Both `data` or `knn_index` were specified! Please pass only one."
            )

        # Find the nearest neighbors
        if knn_index is None:
            n_samples = data.shape[0]

            if k_neighbors == "auto":
                _k_neighbors = min(n_samples - 1, int(3 * perplexity))
            else:
                _k_neighbors = k_neighbors

            self.perplexity = self.check_perplexity(perplexity, _k_neighbors)
            if _k_neighbors > int(3 * self.perplexity):
                log.warning(
                    "The k_neighbors value is over 3 times larger than the perplexity value. "
                    "This may result in an unnecessary slowdown."
                )

            self.knn_index = get_knn_index(
                data, method, _k_neighbors, metric, metric_params, n_jobs,
                random_state, verbose
            )

        else:
            self.knn_index = knn_index
            self.perplexity = self.check_perplexity(perplexity, self.knn_index.k)
            log.info("KNN index provided. Ignoring KNN-related parameters.")

        self.__neighbors, self.__distances = self.knn_index.build()

        with utils.Timer("Calculating affinity matrix...", verbose):
            self.P = joint_probabilities_nn(
                self.__neighbors,
                self.__distances,
                [self.perplexity],
                symmetrize=symmetrize,
                n_jobs=n_jobs,
            )

        self.symmetrize = symmetrize
        self.n_jobs = n_jobs
        self.verbose = verbose

    def set_perplexity(self, new_perplexity):
        """Change the perplexity of the affinity matrix.

        Note that we only allow setting the perplexity to a value not larger
        than the number of neighbors used for the original perplexity. This
        restriction exists because setting a higher perplexity value requires
        recomputing all the nearest neighbors, which can take a long time.
        To avoid potential confusion as to why execution time is slow, this
        is not allowed. If you would like to increase the perplexity above
        that value, simply create a new instance.

        Parameters
        ----------
        new_perplexity: float
            The new perplexity.

        """
        # If the value hasn't changed, there's nothing to do
        if new_perplexity == self.perplexity:
            return
        # Verify that the perplexity isn't negative
        new_perplexity = self.check_perplexity(new_perplexity, np.inf)
        # Verify that the perplexity isn't too large for the kNN graph
        if new_perplexity > self.__neighbors.shape[1]:
            raise RuntimeError(
                "The desired perplexity `%.2f` is larger than the kNN graph "
                "allows. This would need to recompute the nearest neighbors, "
                "which is not efficient. Please create a new `%s` instance "
                "with the increased perplexity."
                % (new_perplexity, self.__class__.__name__)
            )
        # Warn if the perplexity is larger than the heuristic
        if 3 * new_perplexity > self.__neighbors.shape[1]:
            log.warning(
                "The new perplexity is quite close to the computed number of "
                "nearest neighbors. The results may be unexpected. Consider "
                "creating a new `%s` instance with the increased perplexity."
                % (self.__class__.__name__)
            )

        # Recompute the affinity matrix
        self.perplexity = new_perplexity
        k_neighbors = int(3 * new_perplexity)

        with utils.Timer(
            "Perplexity changed. Recomputing affinity matrix...", self.verbose
        ):
            self.P = joint_probabilities_nn(
                self.__neighbors[:, :k_neighbors],
                self.__distances[:, :k_neighbors],
                [self.perplexity],
                symmetrize=self.symmetrize,
                n_jobs=self.n_jobs,
            )

    def to_new(
        self, data, perplexity=None, return_distances=False, k_neighbors="auto"
    ):
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

        k_neighbors: int or ``auto``
            The number of neighbors to query kNN graph for. If ``auto``
            (default), it is set to three times the perplexity.

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

        if k_neighbors == "auto":
            _k_neighbors = min(self.n_samples, int(3 * perplexity))
        else:
            _k_neighbors = k_neighbors

        perplexity = self.check_perplexity(perplexity, _k_neighbors)

        neighbors, distances = self.knn_index.query(data, _k_neighbors)

        with utils.Timer("Calculating affinity matrix...", self.verbose):
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

    def check_perplexity(self, perplexity, k_neighbors):
        if perplexity <= 0:
            raise ValueError("Perplexity must be >=0. %.2f given" % perplexity)

        if perplexity > k_neighbors:
            old_perplexity, perplexity = perplexity, k_neighbors / 3
            log.warning(
                "Perplexity value %d is too high. Using perplexity %.2f instead"
                % (old_perplexity, perplexity)
            )

        return perplexity


def get_knn_index(
    data, method, k, metric, metric_params=None, n_jobs=1, random_state=None, verbose=False
):
    # If we're dealing with a precomputed distance matrix, our job is very easy
    # so we can skip all the remaining checks
    if metric == "precomputed":
        return nearest_neighbors.PrecomputedDistanceMatrix(data, k=k)

    preferred_approx_method = nearest_neighbors.Annoy
    if is_package_installed("pynndescent") and (sp.issparse(data) or metric not in [
        "cosine",
        "euclidean",
        "manhattan",
        "hamming",
        "dot",
        "l1",
        "l2",
        "taxicab",
    ]):
        preferred_approx_method = nearest_neighbors.NNDescent

    if data.shape[0] < 1000:
        preferred_method = nearest_neighbors.Sklearn
    else:
        preferred_method = preferred_approx_method

    methods = {
        "exact": nearest_neighbors.Sklearn,
        "auto": preferred_method,
        "approx": preferred_approx_method,
        "annoy": nearest_neighbors.Annoy,
        "pynndescent": nearest_neighbors.NNDescent,
        "hnsw": nearest_neighbors.HNSW
    }
    if isinstance(method, nearest_neighbors.KNNIndex):
        knn_index = method

    elif method not in methods:
        raise ValueError(
            "Unrecognized nearest neighbor algorithm `%s`. Please choose one "
            "of the supported methods or provide a valid `KNNIndex` instance." % method
        )
    else:
        knn_index = methods[method](
            data=data,
            k=k,
            metric=metric,
            metric_params=metric_params,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )

    return knn_index


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
        np.array(distances, dtype=float),
        np.array(perplexities, dtype=float),
        num_threads=n_jobs,
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
        Specifies the nearest neighbor method to use. Can be ``exact``, ``annoy``,
        ``pynndescent``, ``hnsw``, ``approx``, or ``auto`` (default). ``approx`` uses Annoy
        if the input data matrix is not a sparse object and if Annoy supports
        the given metric. Otherwise it uses Pynndescent. ``auto`` uses exact
        nearest neighbors for N<1000 and the same heuristic as ``approx`` for N>=1000.


    metric: Union[str, Callable]
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

    verbose: bool

    knn_index: Optional[nearest_neighbors.KNNIndex]
        Optionally, a precomptued ``openTSNE.nearest_neighbors.KNNIndex`` object
        can be specified. This option will ignore any KNN-related parameters.
        When ``knn_index`` is specified, ``data`` must be set to None.

    """

    def __init__(
        self,
        data=None,
        sigma=None,
        k=30,
        method="auto",
        metric="euclidean",
        metric_params=None,
        symmetrize=True,
        n_jobs=1,
        random_state=None,
        verbose=False,
        knn_index=None,
    ):
        # Sigma must be specified, but has default set to none, so the parameter
        # order makes more sense
        if sigma is None:
            raise ValueError("`sigma` must be specified!")

        # This can't work if neither data nor the knn index are specified
        if data is None and knn_index is None:
            raise ValueError(
                "At least one of the parameters `data` or `knn_index` must be specified!"
            )
        # This can't work if both data and the knn index are specified
        if data is not None and knn_index is not None:
            raise ValueError(
                "Both `data` or `knn_index` were specified! Please pass only one."
            )

        # Find the nearest neighbors
        if knn_index is None:
            if k >= data.shape[0]:
                raise ValueError(
                    "`k` (%d) cannot be larger than N-1 (%d)." % (k, data.shape[0])
                )

            self.knn_index = get_knn_index(
                data, method, k, metric, metric_params, n_jobs, random_state, verbose
            )

        else:
            self.knn_index = knn_index
            log.info("KNN index provided. Ignoring KNN-related parameters.")

        neighbors, distances = self.knn_index.build()

        with utils.Timer("Calculating affinity matrix...", verbose):
            # Compute asymmetric pairwise input similarities
            conditional_P = np.exp(-(distances ** 2) / (2 * sigma ** 2))
            conditional_P /= np.sum(conditional_P, axis=1)[:, np.newaxis]

            n_samples = self.knn_index.n_samples
            P = sp.csr_matrix(
                (
                    conditional_P.ravel(),
                    neighbors.ravel(),
                    range(0, n_samples * k + 1, k),
                ),
                shape=(n_samples, n_samples),
            )

            # Symmetrize the probability matrix
            if symmetrize:
                P = (P + P.T) / 2

            # Convert weights to probabilities
            P /= np.sum(P)

        self.sigma = sigma
        self.P = P
        self.n_jobs = n_jobs
        self.verbose = verbose

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
            k = self.knn_index.k
        elif k >= n_reference_samples:
            raise ValueError(
                "`k` (%d) cannot be larger than the number of reference "
                "samples (%d)." % (k, self.n_samples)
            )

        if sigma is None:
            sigma = self.sigma

        # Find nearest neighbors and the distances to the new points
        neighbors, distances = self.knn_index.query(data, k)

        with utils.Timer("Calculating affinity matrix...", self.verbose):
            # Compute asymmetric pairwise input similarities
            conditional_P = np.exp(-(distances ** 2) / (2 * sigma ** 2))

            # Convert weights to probabilities
            conditional_P /= np.sum(conditional_P, axis=1)[:, np.newaxis]

            P = sp.csr_matrix(
                (
                    conditional_P.ravel(),
                    neighbors.ravel(),
                    range(0, n_samples * k + 1, k),
                ),
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
        Specifies the nearest neighbor method to use. Can be ``exact``, ``annoy``,
        ``pynndescent``, ``hnsw``, ``approx``, or ``auto`` (default). ``approx`` uses Annoy
        if the input data matrix is not a sparse object and if Annoy supports
        the given metric. Otherwise it uses Pynndescent. ``auto`` uses exact
        nearest neighbors for N<1000 and the same heuristic as ``approx`` for N>=1000.

    metric: Union[str, Callable]
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

    verbose: bool

    knn_index: Optional[nearest_neighbors.KNNIndex]
        Optionally, a precomptued ``openTSNE.nearest_neighbors.KNNIndex`` object
        can be specified. This option will ignore any KNN-related parameters.
        When ``knn_index`` is specified, ``data`` must be set to None.

    """

    def __init__(
        self,
        data=None,
        perplexities=None,
        method="auto",
        metric="euclidean",
        metric_params=None,
        symmetrize=True,
        n_jobs=1,
        random_state=None,
        verbose=False,
        knn_index=None,
    ):
        # Perplexities must be specified, but has default set to none, so the
        # parameter order makes more sense
        if perplexities is None:
            raise ValueError("`perplexities` must be specified!")

        # This can't work if neither data nor the knn index are specified
        if data is None and knn_index is None:
            raise ValueError(
                "At least one of the parameters `data` or `knn_index` must be specified!"
            )
        # This can't work if both data and the knn index are specified
        if data is not None and knn_index is not None:
            raise ValueError(
                "Both `data` or `knn_index` were specified! Please pass only one."
            )

        # Find the nearest neighbors
        if knn_index is None:
            # We will compute the nearest neighbors to the max value of perplexity,
            # smaller values can just use indexing to truncate unneeded neighbors
            n_samples = data.shape[0]
            perplexities = self.check_perplexities(perplexities, n_samples)
            max_perplexity = np.max(perplexities)
            k_neighbors = min(n_samples - 1, int(3 * max_perplexity))

            self.knn_index = get_knn_index(
                data, method, k_neighbors, metric, metric_params, n_jobs, random_state, verbose
            )

        else:
            self.knn_index = knn_index
            log.info("KNN index provided. Ignoring KNN-related parameters.")

        self.__neighbors, self.__distances = self.knn_index.build()

        with utils.Timer("Calculating affinity matrix...", verbose):
            self.P = self._calculate_P(
                self.__neighbors,
                self.__distances,
                perplexities,
                symmetrize=symmetrize,
                n_jobs=n_jobs,
            )

        self.perplexities = perplexities
        self.symmetrize = symmetrize
        self.n_jobs = n_jobs
        self.verbose = verbose

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

        new_perplexities = self.check_perplexities(new_perplexities, self.n_samples)
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
        with utils.Timer(
            "Perplexity changed. Recomputing affinity matrix...", self.verbose
        ):
            self.P = self._calculate_P(
                self.__neighbors[:, :k_neighbors],
                self.__distances[:, :k_neighbors],
                self.perplexities,
                symmetrize=self.symmetrize,
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
        perplexities = self.check_perplexities(perplexities, self.n_samples)

        max_perplexity = np.max(perplexities)
        k_neighbors = min(self.n_samples - 1, int(3 * max_perplexity))

        neighbors, distances = self.knn_index.query(data, k_neighbors)

        with utils.Timer("Calculating affinity matrix...", self.verbose):
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

    def check_perplexities(self, perplexities, n_samples):
        """Check and correct/truncate perplexities.

        If a perplexity is too large, it is corrected to the largest allowed
        value. It is then inserted into the list of perplexities only if that
        value doesn't already exist in the list.

        """
        usable_perplexities = []
        for perplexity in sorted(perplexities):
            if perplexity <= 0:
                raise ValueError("Perplexity must be >=0. %.2f given" % perplexity)

            if 3 * perplexity > n_samples - 1:
                new_perplexity = (n_samples - 1) / 3

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
        Specifies the nearest neighbor method to use. Can be ``exact``, ``annoy``,
        ``pynndescent``, ``hnsw``, ``approx``, or ``auto`` (default). ``approx`` uses Annoy
        if the input data matrix is not a sparse object and if Annoy supports
        the given metric. Otherwise it uses Pynndescent. ``auto`` uses exact
        nearest neighbors for N<1000 and the same heuristic as ``approx`` for N>=1000.

    metric: Union[str, Callable]
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

    verbose: bool

    knn_index: Optional[nearest_neighbors.KNNIndex]
        Optionally, a precomptued ``openTSNE.nearest_neighbors.KNNIndex`` object
        can be specified. This option will ignore any KNN-related parameters.
        When ``knn_index`` is specified, ``data`` must be set to None.

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


class Uniform(Affinities):
    """Compute affinities using using nearest neighbors and uniform kernel in
    the ambient space.

    Parameters
    ----------
    data: np.ndarray
        The data matrix.

    k_neighbors: int

    method: str
        Specifies the nearest neighbor method to use. Can be ``exact``, ``annoy``,
        ``pynndescent``, ``hnsw``, ``approx``, or ``auto`` (default). ``approx`` uses Annoy
        if the input data matrix is not a sparse object and if Annoy supports
        the given metric. Otherwise it uses Pynndescent. ``auto`` uses exact
        nearest neighbors for N<1000 and the same heuristic as ``approx`` for N>=1000.


    metric: Union[str, Callable]
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

    verbose: bool

    knn_index: Optional[nearest_neighbors.KNNIndex]
        Optionally, a precomptued ``openTSNE.nearest_neighbors.KNNIndex`` object
        can be specified. This option will ignore any KNN-related parameters.
        When ``knn_index`` is specified, ``data`` must be set to None.

    """

    def __init__(
        self,
        data=None,
        k_neighbors=30,
        method="auto",
        metric="euclidean",
        metric_params=None,
        symmetrize=True,
        n_jobs=1,
        random_state=None,
        verbose=False,
        knn_index=None,
    ):
        # This can't work if neither data nor the knn index are specified
        if data is None and knn_index is None:
            raise ValueError(
                "At least one of the parameters `data` or `knn_index` must be specified!"
            )
        # This can't work if both data and the knn index are specified
        if data is not None and knn_index is not None:
            raise ValueError(
                "Both `data` or `knn_index` were specified! Please pass only one."
            )

        if knn_index is None:
            if k_neighbors >= data.shape[0]:
                raise ValueError(
                    "`k_neighbors` (%d) cannot be larger than N-1 (%d)." %
                    (k_neighbors, data.shape[0])
                )

            self.knn_index = get_knn_index(
                data, method, k_neighbors, metric, metric_params, n_jobs, random_state, verbose
            )

        else:
            self.knn_index = knn_index
            log.info("KNN index provided. Ignoring KNN-related parameters.")

        neighbors, distances = self.knn_index.build()

        k_neighbors = self.knn_index.k
        n_samples = self.knn_index.n_samples
        P = sp.csr_matrix(
            (
                np.ones_like(distances).ravel(),
                neighbors.ravel(),
                range(0, n_samples * k_neighbors + 1, k_neighbors),
            ),
            shape=(n_samples, n_samples),
        )

        # Symmetrize the probability matrix
        if symmetrize:
            P = (P + P.T) / 2

        # Convert weights to probabilities
        P /= np.sum(P)

        self.P = P
        self.verbose = verbose
        self.n_jobs = n_jobs

    def to_new(self, data, k_neighbors=None, return_distances=False):
        """Compute the affinities of new samples to the initial samples.

        This is necessary for embedding new data points into an existing
        embedding.

        Parameters
        ----------
        data: np.ndarray
            The data points to be added to the existing embedding.

        k_neighbors: int
            The number of nearest neighbors to consider.

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

        if k_neighbors is None:
            k_neighbors = self.knn_index.k
        elif k_neighbors >= n_reference_samples:
            raise ValueError(
                "`k` (%d) cannot be larger than the number of reference "
                "samples (%d)." % (k_neighbors, self.n_samples)
            )

        # Find nearest neighbors and the distances to the new points
        neighbors, distances = self.knn_index.query(data, k_neighbors)

        values = np.ones_like(distances)
        values /= np.sum(values, axis=1)[:, np.newaxis]

        P = sp.csr_matrix(
            (
                values.ravel(),
                neighbors.ravel(),
                range(0, n_samples * k_neighbors + 1, k_neighbors),
            ),
            shape=(n_samples, n_reference_samples),
        )

        if return_distances:
            return P, neighbors, distances

        return P
