import logging
import os
import warnings

import numpy as np
from scipy.spatial.distance import cdist
from sklearn import neighbors
from sklearn.utils import check_random_state

from openTSNE import utils

log = logging.getLogger(__name__)


class KNNIndex:
    VALID_METRICS = []

    def __init__(
        self,
        data,
        k,
        metric="euclidean",
        metric_params=None,
        n_jobs=1,
        random_state=None,
        verbose=False,
    ):
        self.data = data
        self.n_samples = data.shape[0]
        self.k = k
        self.metric = self.check_metric(metric)
        self.metric_params = metric_params
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

        self.index = None

    def build(self):
        """Build the nearest neighbor index on the training data.

        Builds an index on the training data and computes the nearest neighbors
        on the training data.

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


class Sklearn(KNNIndex):
    VALID_METRICS = [
        "braycurtis",
        "canberra",
        "chebyshev",
        "cityblock",
        "dice",
        "euclidean",
        "hamming",
        "haversine",
        "infinity",
        "jaccard",
        "kulsinski",
        "l1",
        "l2",
        "mahalanobis",
        "manhattan",
        "matching",
        "minkowski",
        "p",
        "pyfunc",
        "rogerstanimoto",
        "russellrao",
        "seuclidean",
        "sokalmichener",
        "sokalsneath",
        "wminkowski",
    ] + ["cosine"]  # our own workaround implementation

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__data = None

    def build(self):
        data, k = self.data, self.k

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
            algorithm="auto",
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
    """Annoy KNN Index.

    Notes
    -----
    Pickling:
        Annoy doesn't support pickling. As a workaround, we override the pickling
        process and save the annoy index file separately. Upon unpickling, this
        file will attempt to be reloaded.

        However, since we can't access the actual pickle file location from
        __getstate__, the annoy index is saved into the current working directory.
        And, it will also be loaded from cwd. This means that if we pickle an
        object into a specific directory, our files could end up in different
        places. And when sharing a pickle, they may need to be put into different
        directories.

        Alternatively, the use can set the ``.pickle_fname`` attribute to specify
        a file name and location for the save annoy index e.g.
        ``./pickle/my-index.ann``. This should make it at least somewhat easier
        to specify the pickle names.

        This is extremely messy, but it is better than not supporting pickling at
        all. If anyone has a better solution, I would welcome any and all help.

    """

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self):
        data, k = self.data, self.k

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

        random_state = check_random_state(self.random_state)
        self.index.set_seed(random_state.randint(np.iinfo(np.int32).max))

        for i in range(N):
            self.index.add_item(i, data[i])

        # Number of trees. FIt-SNE uses 50 by default.
        self.index.build(50, n_jobs=self.n_jobs)

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

    def __getstate__(self):
        import tempfile
        import base64
        from os import path

        d = dict(self.__dict__)
        # If the index is not None, we want to save the encoded index
        if self.index is not None:
            with tempfile.TemporaryDirectory() as dirname:
                self.index.save(path.join(dirname, "tmp.ann"))

                with open(path.join(dirname, "tmp.ann"), "rb") as f:
                    b64_index = base64.b64encode(f.read())

            d["b64_index"] = b64_index
            del d["index"]

        return d

    def __setstate__(self, state):
        import tempfile
        import base64
        from os import path

        from openTSNE.dependencies.annoy import AnnoyIndex

        # If a base64 index is given, we have to load the index
        if "b64_index" in state:
            assert "index" not in state
            b64_index = state["b64_index"]
            del state["b64_index"]

            annoy_metric = state["metric"]
            annoy_aliases = {
                "cosine": "angular",
                "l1": "manhattan",
                "l2": "euclidean",
                "taxicab": "manhattan",
            }
            if annoy_metric in annoy_aliases:
                annoy_metric = annoy_aliases[annoy_metric]

            self.index = AnnoyIndex(state["data"].shape[1], annoy_metric)
            with tempfile.TemporaryDirectory() as dirname:
                with open(path.join(dirname, "tmp.ann"), "wb") as f:
                    f.write(base64.b64decode(b64_index))
                self.index.load(path.join(dirname, "tmp.ann"))

        self.__dict__.update(state)


class NNDescent(KNNIndex):
    VALID_METRICS = [
        "euclidean",
        "l2",
        "sqeuclidean",
        "manhattan",
        "taxicab",
        "l1",
        "chebyshev",
        "linfinity",
        "linfty",
        "linf",
        "minkowski",
        "seuclidean",
        "standardised_euclidean",
        "wminkowski",
        "weighted_minkowski",
        "mahalanobis",
        "canberra",
        "cosine",
        "dot",
        "correlation",
        "haversine",
        "braycurtis",
        "spearmanr",
        "tsss",
        "true_angular",
        "hellinger",
        "kantorovich",
        "wasserstein",
        "sinkhorn",
        "jensen-shannon",
        "jensen_shannon",
        "symmetric-kl",
        "symmetric_kl",
        "symmetric_kullback_liebler",
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

    def build(self):
        data, k = self.data, self.k

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

        self.index = pynndescent.NNDescent(
            data,
            n_neighbors=n_neighbors_build,
            metric=self.metric,
            metric_kwds=self.metric_params,
            random_state=self.random_state,
            n_trees=n_trees,
            n_iters=n_iters,
            max_candidates=60,
            n_jobs=self.n_jobs,
            verbose=self.verbose > 1,
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


class HNSW(KNNIndex):
    VALID_METRICS = [
        "cosine",
        "euclidean",
        "dot",
        "l2",
        "ip",
    ]

    def __init__(self, *args, **kwargs):
        try:
            from hnswlib import Index  # pylint: disable=unused-import,unused-variable
        except ImportError:
            raise ImportError(
                "Please install hnswlib: `conda install -c conda-forge "
                "hnswlib` or `pip install hnswlib`."
            )
        super().__init__(*args, **kwargs)

    def build(self):
        data, k = self.data, self.k

        timer = utils.Timer(
            f"Finding {k} nearest neighbors using HNSWlib approximate search using "
            f"{self.metric} distance...",
            verbose=self.verbose,
        )
        timer.__enter__()

        from hnswlib import Index

        hnsw_space = {
            "cosine": "cosine",
            "dot": "ip",
            "euclidean": "l2",
            "ip": "ip",
            "l2": "l2",
        }[self.metric]

        random_state = check_random_state(self.random_state)
        random_seed = random_state.randint(np.iinfo(np.int32).max)

        self.index = Index(space=hnsw_space, dim=data.shape[1])

        # Initialize HNSW Index
        self.index.init_index(
            max_elements=data.shape[0],
            ef_construction=200,
            M=16,
            random_seed=random_seed,
        )

        # Build index tree from data
        self.index.add_items(data, num_threads=self.n_jobs)

        # Set ef parameter for (ideal) precision/recall
        self.index.set_ef(min(2 * k, self.index.get_current_count()))

        # Query for kNN
        indices, distances = self.index.knn_query(data, k=k + 1, num_threads=self.n_jobs)

        # Stop timer
        timer.__exit__()

        # return indices and distances, skip first entry, which is always the point itself
        return indices[:, 1:], distances[:, 1:]

    def query(self, query, k):
        timer = utils.Timer(
            f"Finding {k} nearest neighbors in existing embedding using HNSWlib "
            f"approximate search...",
            self.verbose,
        )
        timer.__enter__()

        # Set ef parameter for (ideal) precision/recall
        self.index.set_ef(min(2 * k, self.index.get_current_count()))

        # Query for kNN
        indices, distances = self.index.knn_query(query, k=k, num_threads=self.n_jobs)

        # Stop timer
        timer.__exit__()

        # return indices and distances
        return indices, distances

    def __getstate__(self):
        import tempfile
        import base64
        from os import path

        d = dict(self.__dict__)
        # If the index is not None, we want to save the encoded index
        if self.index is not None:
            with tempfile.TemporaryDirectory() as dirname:
                self.index.save_index(path.join(dirname, "tmp.bin"))

                with open(path.join(dirname, "tmp.bin"), "rb") as f:
                    b64_index = base64.b64encode(f.read())

            d["b64_index"] = b64_index
            del d["index"]

        return d

    def __setstate__(self, state):
        import tempfile
        import base64
        from os import path

        from hnswlib import Index

        # If a base64 index is given, we have to load the index
        if "b64_index" in state:
            assert "index" not in state
            b64_index = state["b64_index"]
            del state["b64_index"]

            hnsw_metric = state["metric"]
            hnsw_aliases = {
                "cosine": "cosine",
                "dot": "ip",
                "euclidean": "l2",
                "ip": "ip",
                "l2": "l2",
            }
            if hnsw_metric in hnsw_aliases:
                hnsw_metric = hnsw_aliases[hnsw_metric]

            self.index = Index(space=hnsw_metric, dim=state["data"].data.shape[1])
            with tempfile.TemporaryDirectory() as dirname:
                with open(path.join(dirname, "tmp.bin"), "wb") as f:
                    f.write(base64.b64decode(b64_index))
                self.index.load_index(path.join(dirname, "tmp.bin"))

        self.__dict__.update(state)


class PrecomputedDistanceMatrix(KNNIndex):
    """Use a precomputed distance matrix to construct the KNNG.

    Parameters
    ----------
    distance_matrix: np.ndarray
        A square, symmetric, and contain only poistive values.

    """

    def __init__(self, distance_matrix, k):
        nn = neighbors.NearestNeighbors(metric="precomputed")
        nn.fit(distance_matrix)
        self.distances, self.indices = nn.kneighbors(n_neighbors=k)
        self.n_samples = distance_matrix.shape[0]
        self.k = k

    def build(self):
        return self.indices, self.distances

    def query(self, *args, **kwargs):
        raise RuntimeError("Precomputed distance matrices cannot be queried")


class PrecomputedNeighbors(KNNIndex):
    """Use a precomputed distance matrix to construct the KNNG.

    Parameters
    ----------
    neighbors: np.ndarray
        A N x K matrix containing the indices of point i's k nearest neighbors.

    distances: np.ndarray
        A N x K matrix containing the distances to from data point i to its k
        nearest neighbors.

    """

    def __init__(self, neighbors, distances):
        self.distances, self.indices = distances, neighbors
        self.n_samples = neighbors.shape[0]
        self.k = neighbors.shape[1]

    def build(self):
        return self.indices, self.distances

    def query(self, *args, **kwargs):
        raise RuntimeError("Precomputed distance matrices cannot be queried")
