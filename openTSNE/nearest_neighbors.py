import numpy as np
from sklearn import neighbors

import pynndescent


class KNNIndex:
    VALID_METRICS = []

    def __init__(self, metric, metric_params=None, n_jobs=1, random_state=None):
        self.index = None
        self.metric = metric
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


class BallTree(KNNIndex):
    VALID_METRICS = neighbors.BallTree.valid_metrics

    def build(self, data, k):
        self.check_metric(self.metric)
        self.index = neighbors.NearestNeighbors(
            algorithm="ball_tree",
            metric=self.metric,
            metric_params=self.metric_params,
            n_jobs=self.n_jobs,
        )
        self.index.fit(data)

        # Return the nearest neighbors in the training set
        distances, indices = self.index.kneighbors(n_neighbors=k)
        return indices, distances

    def query(self, query, k):
        distances, indices = self.index.kneighbors(query, n_neighbors=k)
        return indices, distances


class NNDescent(KNNIndex):
    VALID_METRICS = pynndescent.distances.named_distances

    def build(self, data, k):
        self.check_metric(self.metric)

        # These values were taken from UMAP, which we assume to be sensible defaults
        n_trees = 5 + int(round((data.shape[0]) ** 0.5 / 20))
        n_iters = max(5, int(round(np.log2(data.shape[0]))))

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
