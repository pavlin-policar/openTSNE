import unittest

import numpy as np

from fastTSNE import nearest_neighbors


class KNNIndexTestMixin:
    knn_index = None

    def __init__(self, *args, **kwargs):
        self.x1 = np.random.normal(100, 50, (100, 50))
        self.x2 = np.random.normal(100, 50, (100, 50))
        super().__init__(*args, **kwargs)

    def test_returns_correct_number_neighbors_query_train(self):
        ks = [1, 5, 10]
        n_samples = self.x1.shape[0]
        index: nearest_neighbors.KNNIndex = self.knn_index("euclidean")
        index.build(self.x1)

        for k in ks:
            indices, neighbors = index.query_train(self.x1, k)
            self.assertEqual(indices.shape, (n_samples, k))
            self.assertEqual(neighbors.shape, (n_samples, k))

    def test_returns_correct_number_neighbors_query(self):
        ks = [1, 5, 10]
        n_samples = self.x1.shape[0]
        index: nearest_neighbors.KNNIndex = self.knn_index("euclidean")
        index.build(self.x1)

        for k in ks:
            indices, neighbors = index.query(self.x2, k)
            self.assertEqual(indices.shape, (n_samples, k))
            self.assertEqual(neighbors.shape, (n_samples, k))


class TestBallTree(KNNIndexTestMixin, unittest.TestCase):
    knn_index = nearest_neighbors.BallTree


class TestVPTree(KNNIndexTestMixin, unittest.TestCase):
    knn_index = nearest_neighbors.VPTree


class TestNNDescent(KNNIndexTestMixin, unittest.TestCase):
    knn_index = nearest_neighbors.NNDescent
