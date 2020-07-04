import warnings

import numpy as np
from scipy.spatial.distance import cdist
from sklearn import neighbors
from sklearn.utils import check_random_state

from openTSNE import utils


class KNNIndex:
    VALID_METRICS = []

    def __init__(
        self, metric, metric_params=None, n_jobs=1, random_state=None, verbose=False
    ):
        self.index = None
        self.metric = self.check_metric(metric)
        self.metric_params = metric_params
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    def build(self, data, k):
        """Build the nearest neighbor index on the training data.

        Builds an index on the training data and computes the nearest neighbors
        on the training data.

        Parameters
        ----------
        data: array_like
            Training data.
        k: int
            The number of nearest neighbors to compute on the training data.

        Returns
        -------
        indices: np.ndarray
        distances: np.ndarray

        """

    def query(self, query, k):
        """Query the index with new points.

        Finds k nearest neighbors from the training data to each row of the
        query data.

        Parameters
        ----------
        query: array_like
        k: int

        Returns
        -------
        indices: np.ndarray
        distances: np.ndarray

        """

    def check_metric(self, metric):
        """Check that the metric is supported by the KNNIndex instance."""
        if callable(metric):
            pass
        elif metric not in self.VALID_METRICS:
            raise ValueError(
                f"`{self.__class__.__name__}` does not support the `{metric}` "
                f"metric. Please choose one of the supported metrics: "
                f"{', '.join(self.VALID_METRICS)}."
            )

        return metric


class BallTree(KNNIndex):
    VALID_METRICS = neighbors.BallTree.valid_metrics + ["cosine"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__data = None

    def build(self, data, k):
        timer = utils.Timer(
            f"Finding {k} nearest neighbors using exact search using "
            f"{self.metric} distance...",
            verbose=self.verbose,
        )
        timer.__enter__()

        if self.metric == "cosine":
            # The nearest neighbor ranking for cosine distance is the same as
            # for euclidean distance on normalized data
            effective_metric = "euclidean"
            effective_data = data.copy()
            effective_data = (
                effective_data / np.linalg.norm(effective_data, axis=1)[:, None]
            )
            # In order to properly compute cosine distances when querying the
            # index, we need to store the original data
            self.__data = data
        else:
            effective_metric = self.metric
            effective_data = data

        self.index = neighbors.NearestNeighbors(
            algorithm="ball_tree",
            metric=effective_metric,
            metric_params=self.metric_params,
            n_jobs=self.n_jobs,
        )
        self.index.fit(effective_data)

        # Return the nearest neighbors in the training set
        distances, indices = self.index.kneighbors(n_neighbors=k)

        # If using cosine distance, the computed distances will be wrong and
        # need to be recomputed
        if self.metric == "cosine":
            distances = np.vstack(
                [
                    cdist(np.atleast_2d(x), data[idx], metric="cosine")
                    for x, idx in zip(data, indices)
                ]
            )

        timer.__exit__()

        return indices, distances

    def query(self, query, k):
        timer = utils.Timer(
            f"Finding {k} nearest neighbors in existing embedding using exact search...",
            self.verbose,
        )
        timer.__enter__()

        # The nearest neighbor ranking for cosine distance is the same as for
        # euclidean distance on normalized data
        if self.metric == "cosine":
            effective_data = query.copy()
            effective_data = (
                effective_data / np.linalg.norm(effective_data, axis=1)[:, None]
            )
        else:
            effective_data = query

        distances, indices = self.index.kneighbors(effective_data, n_neighbors=k)

        # If using cosine distance, the computed distances will be wrong and
        # need to be recomputed
        if self.metric == "cosine":
            if self.__data is None:
                raise RuntimeError(
                    "The original data was unavailable when querying cosine "
                    "distance. Did you change the distance metric after "
                    "building the index? Please rebuild the index using cosine "
                    "similarity."
                )
            distances = np.vstack(
                [
                    cdist(np.atleast_2d(x), self.__data[idx], metric="cosine")
                    for x, idx in zip(query, indices)
                ]
            )

        timer.__exit__()

        return indices, distances


class Annoy(KNNIndex):
    VALID_METRICS = [
        "cosine",
        "euclidean",
        "manhattan",
        "hamming",
        "dot",
        "l1",
        "l2",
        "taxicab",
    ]

    def build(self, data, k):
        timer = utils.Timer(
            f"Finding {k} nearest neighbors using Annoy approximate search using "
            f"{self.metric} distance...",
            verbose=self.verbose,
        )
        timer.__enter__()

        from openTSNE.dependencies.annoy import AnnoyIndex

        N = data.shape[0]

        annoy_metric = self.metric
        annoy_aliases = {
            "cosine": "angular",
            "l1": "manhattan",
            "l2": "euclidean",
            "taxicab": "manhattan",
        }
        if annoy_metric in annoy_aliases:
            annoy_metric = annoy_aliases[annoy_metric]

        self.index = AnnoyIndex(data.shape[1], annoy_metric)

        if self.random_state:
            self.index.set_seed(self.random_state)

        for i in range(N):
            self.index.add_item(i, data[i])

        # Number of trees. FIt-SNE uses 50 by default.
        self.index.build(50)

        # Return the nearest neighbors in the training set
        distances = np.zeros((N, k))
        indices = np.zeros((N, k)).astype(int)

        def getnns(i):
            # Annoy returns the query point itself as the first element
            indices_i, distances_i = self.index.get_nns_by_item(
                i, k + 1, include_distances=True
            )
            indices[i] = indices_i[1:]
            distances[i] = distances_i[1:]

        if self.n_jobs == 1:
            for i in range(N):
                getnns(i)
        else:
            from joblib import Parallel, delayed

            Parallel(n_jobs=self.n_jobs, require="sharedmem")(
                delayed(getnns)(i) for i in range(N)
            )

        timer.__exit__()

        return indices, distances

    def query(self, query, k):
        timer = utils.Timer(
            f"Finding {k} nearest neighbors in existing embedding using Annoy "
            f"approximate search...",
            self.verbose,
        )
        timer.__enter__()

        N = query.shape[0]
        distances = np.zeros((N, k))
        indices = np.zeros((N, k)).astype(int)

        def getnns(i):
            indices[i], distances[i] = self.index.get_nns_by_vector(
                query[i], k, include_distances=True
            )

        if self.n_jobs == 1:
            for i in range(N):
                getnns(i)
        else:
            from joblib import Parallel, delayed

            Parallel(n_jobs=self.n_jobs, require="sharedmem")(
                delayed(getnns)(i) for i in range(N)
            )

        timer.__exit__()

        return indices, distances


class NNDescent(KNNIndex):
    VALID_METRICS = [
        # general minkowski distances
        "euclidean",
        "l2",
        "manhattan",
        "taxicab",
        "l1",
        "chebyshev",
        "linfinity",
        "linfty",
        "linf",
        "minkowski",
        # Standardised/weighted distances
        "seuclidean",
        "standardised_euclidean",
        "wminkowski",
        "weighted_minkowski",
        "mahalanobis",
        # Other distances
        "canberra",
        "cosine",
        "correlation",
        "hellinger",
        "haversine",
        "braycurtis",
        "spearmanr",
        # Binary distances
        "hamming",
        "jaccard",
        "dice",
        "matching",
        "kulsinski",
        "rogerstanimoto",
        "russellrao",
        "sokalsneath",
        "sokalmichener",
        "yule",
    ]

    def __init__(self, *args, **kwargs):
        try:
            import pynndescent  # pylint: disable=unused-import,unused-variable
        except ImportError:
            raise ImportError(
                "Please install pynndescent: `conda install -c conda-forge "
                "pynndescent` or `pip install pynndescent`."
            )
        super().__init__(*args, **kwargs)

    def check_metric(self, metric):
        import pynndescent

        if not np.array_equal(
            list(pynndescent.distances.named_distances), self.VALID_METRICS
        ):
            warnings.warn(
                "`pynndescent` has recently changed which distance metrics are supported, "
                "and `openTSNE.nearest_neighbors` has not been updated. Please notify the "
                "developers of this change."
            )

        if callable(metric):
            from numba.core.registry import CPUDispatcher

            if not isinstance(metric, CPUDispatcher):
                warnings.warn(
                    f"`pynndescent` requires callable metrics to be "
                    f"compiled with `numba`, but `{metric.__name__}` is not compiled. "
                    f"`openTSNE.nearest_neighbors.NNDescent` "
                    f"will attempt to compile the function. "
                    f"If this results in an error, then the function may not be "
                    f"compatible with `numba.njit` and should be rewritten. "
                    f"Otherwise, set `neighbors`='exact' to use `scikit-learn` "
                    f"for calculating nearest neighbors."
                )
                from numba import njit

                metric = njit(fastmath=True)(metric)

        return super().check_metric(metric)

    def build(self, data, k):
        timer = utils.Timer(
            f"Finding {k} nearest neighbors using NN descent approximate search using "
            f"{self.metric} distance...",
            verbose=self.verbose,
        )
        timer.__enter__()

        # These values were taken from UMAP, which we assume to be sensible defaults
        n_trees = 5 + int(round((data.shape[0]) ** 0.5 / 20))
        n_iters = max(5, int(round(np.log2(data.shape[0]))))

        # Numba takes a while to load up, so there's little point in loading it
        # unless we're actually going to use it
        import pynndescent

        # Will use query() only for k>15
        if k <= 15:
            n_neighbors_build = k + 1
        else:
            n_neighbors_build = 15

        # Due to a bug, pynndescent currently does not support n_jobs>1
        # for sparse inputs. This should be removed once it's fixed.
        n_jobs_pynndescent = self.n_jobs
        import scipy.sparse as sp

        if sp.issparse(data) and self.n_jobs != 1:
            warnings.warn(
                f"Running `pynndescent` with n_jobs=1 because it does not "
                f"currently support n_jobs>1 with sparse inputs. See "
                f"https://github.com/lmcinnes/pynndescent/issues/94."
            )
            n_jobs_pynndescent = 1

        # UMAP uses the "alternative" algorithm, but that sometimes causes
        # memory corruption, so use the standard one, which seems to work fine
        self.index = pynndescent.NNDescent(
            data,
            n_neighbors=n_neighbors_build,
            metric=self.metric,
            metric_kwds=self.metric_params,
            random_state=self.random_state,
            n_trees=n_trees,
            n_iters=n_iters,
            algorithm="standard",
            max_candidates=60,
            n_jobs=n_jobs_pynndescent,
        )

        # -1 in indices means that pynndescent failed
        indices, distances = self.index.neighbor_graph
        mask = np.sum(indices == -1, axis=1) > 0

        if k > 15:
            indices, distances = self.index.query(data, k=k + 1)

        # As a workaround, we let the failed points group together
        if np.sum(mask) > 0:
            if self.verbose:
                opt = np.get_printoptions()
                np.set_printoptions(threshold=np.inf)
                warnings.warn(
                    f"`pynndescent` failed to find neighbors for some of the points. "
                    f"As a workaround, openTSNE considers all such points similar to "
                    f"each other, so they will likely form a cluster in the embedding."
                    f"The indices of the failed points are:\n{np.where(mask)[0]}"
                )
                np.set_printoptions(**opt)
            else:
                warnings.warn(
                    f"`pynndescent` failed to find neighbors for some of the points. "
                    f"As a workaround, openTSNE considers all such points similar to "
                    f"each other, so they will likely form a cluster in the embedding. "
                    f"Run with verbose=True, to see indices of the failed points."
                )
            distances[mask] = 1
            rs = check_random_state(self.random_state)
            fake_indices = rs.choice(
                np.sum(mask), size=np.sum(mask) * indices.shape[1], replace=True
            )
            fake_indices = np.where(mask)[0][fake_indices]
            indices[mask] = np.reshape(fake_indices, (np.sum(mask), indices.shape[1]))

        timer.__exit__()

        return indices[:, 1:], distances[:, 1:]

    def query(self, query, k):
        timer = utils.Timer(
            f"Finding {k} nearest neighbors in existing embedding using NN Descent "
            f"approxmimate search...",
            self.verbose,
        )
        timer.__enter__()

        indices, distances = self.index.query(query, k=k)

        timer.__exit__()

        return indices, distances

