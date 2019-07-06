import warnings

import numpy as np
from scipy.spatial.distance import cdist
from sklearn import neighbors


class KNNIndex:
    VALID_METRICS = []

    def __init__(self, metric, metric_params=None, n_jobs=1, random_state=None):
        self.index = None
        self.metric = self.check_metric(metric)
        self.metric_params = metric_params
        self.n_jobs = n_jobs
        self.random_state = random_state

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
        if metric not in self.VALID_METRICS:
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
        if self.metric == "cosine":
            # The nearest neighbor ranking for cosine distance is the same as
            # for euclidean distance on normalized data
            effective_metric = "euclidean"
            effective_data = data.copy()
            effective_data = effective_data / np.linalg.norm(effective_data, axis=1)[:, None]
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
            distances = np.vstack([
                cdist(np.atleast_2d(x), data[idx], metric="cosine")
                for x, idx in zip(data, indices)
            ])

        return indices, distances

    def query(self, query, k):
        # The nearest neighbor ranking for cosine distance is the same as for
        # euclidean distance on normalized data
        if self.metric == "cosine":
            effective_data = query.copy()
            effective_data = effective_data / np.linalg.norm(effective_data, axis=1)[:, None]
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
            distances = np.vstack([
                cdist(np.atleast_2d(x), self.__data[idx], metric="cosine")
                for x, idx in zip(query, indices)
            ])

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
        "haversine",
        "braycurtis",
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

    def check_metric(self, *args, **kwargs):
        import pynndescent

        if not np.array_equal(pynndescent.distances.named_distances, self.VALID_METRICS):
            warnings.warn(
                "`pynndescent` has recently changed which distance metrics are supported, "
                "and `openTSNE.nearest_neighbors` has not been updated. Please notify the "
                "developers of this change."
            )

        return super().check_metric(*args, **kwargs)

    def build(self, data, k):
        # These values were taken from UMAP, which we assume to be sensible defaults
        n_trees = 5 + int(round((data.shape[0]) ** 0.5 / 20))
        n_iters = max(5, int(round(np.log2(data.shape[0]))))

        # Numba takes a while to load up, so there's little point in loading it
        # unless we're actually going to use it
        import pynndescent

        # UMAP uses the "alternative" algorithm, but that sometimes causes
        # memory corruption, so use the standard one, which seems to work fine
        self.index = pynndescent.NNDescent(
            data,
            n_neighbors=15,
            metric=self.metric,
            metric_kwds=self.metric_params,
            random_state=self.random_state,
            n_trees=n_trees,
            n_iters=n_iters,
            algorithm="standard",
            max_candidates=60,
            n_jobs=self.n_jobs,
        )

        indices, distances = self.index.query(data, k=k + 1, queue_size=1)
        return indices[:, 1:], distances[:, 1:]

    def query(self, query, k):
        return self.index.query(query, k=k, queue_size=1)
