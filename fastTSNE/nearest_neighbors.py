from sklearn.neighbors import NearestNeighbors
from pynndescent import NNDescent as LibNNDescent


class KNNIndex:
    def __init__(self, metric, n_jobs=1):
        self.index = None
        self.metric = metric
        self.n_jobs = n_jobs

    def build(self, data):
        """Build the index so we can query nearest neighbors."""

    def query_train(self, data, k):
        """Query the index for the points used to build index."""

    def query(self, query, k):
        """Query the index with new points."""


class KDTree(KNNIndex):
    def build(self, data):
        self.index = NearestNeighbors(
            algorithm='kd_tree', metric=self.metric, n_jobs=self.n_jobs)
        self.index.fit(data)

    def query_train(self, data, k):
        distances, neighbors = self.index.kneighbors(n_neighbors=k)
        return neighbors, distances

    def query(self, query, k):
        distances, neighbors = self.index.kneighbors(query, n_neighbors=k)
        return neighbors, distances


class NNDescent(KNNIndex):
    # TODO: Make mapping from sklearn metrics to lib metrics

    def build(self, data):
        self.index = LibNNDescent(data, metric=self.metric, n_neighbors=5)

    def query_train(self, data, k):
        search_neighbors = min(data.shape[0] - 1, k + 1)
        neighbors, distances = self.index.query(data, k=search_neighbors, queue_size=1)
        return neighbors[:, 1:], distances[:, 1:]

    def query(self, query, k):
        return self.index.query(query, k=k, queue_size=1)
