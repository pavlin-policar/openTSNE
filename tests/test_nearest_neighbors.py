import openTSNE
import unittest
from unittest.mock import patch

import numpy as np

from openTSNE import nearest_neighbors
from .test_tsne import check_mock_called_with_kwargs


class KNNIndexTestMixin:
    knn_index = None

    def __init__(self, *args, **kwargs):
        self.x1 = np.random.normal(100, 50, (100, 50))
        self.x2 = np.random.normal(100, 50, (100, 50))
        super().__init__(*args, **kwargs)

    def test_returns_correct_number_neighbors_query_train(self):
        ks = [1, 5, 10, 30, 50]
        n_samples = self.x1.shape[0]
        index: nearest_neighbors.KNNIndex = self.knn_index("euclidean")

        for k in ks:
            indices, neighbors = index.build(self.x1, k=k)
            self.assertEqual(indices.shape, (n_samples, k))
            self.assertEqual(neighbors.shape, (n_samples, k))

    def test_returns_correct_number_neighbors_query(self):
        ks = [1, 5, 10, 30, 50]
        n_samples = self.x1.shape[0]
        index: nearest_neighbors.KNNIndex = self.knn_index("euclidean")
        index.build(self.x1, k=30)

        for k in ks:
            indices, neighbors = index.query(self.x2, k)
            self.assertEqual(indices.shape, (n_samples, k))
            self.assertEqual(neighbors.shape, (n_samples, k))


class TestBallTree(KNNIndexTestMixin, unittest.TestCase):
    knn_index = nearest_neighbors.BallTree


class TestNNDescent(KNNIndexTestMixin, unittest.TestCase):
    knn_index = nearest_neighbors.NNDescent

    def test_query_train_same_result_with_fixed_random_state(self):
        knn_index1 = nearest_neighbors.NNDescent("euclidean", random_state=1)
        indices1, distances1 = knn_index1.build(self.x1, k=20)

        knn_index2 = nearest_neighbors.NNDescent("euclidean", random_state=1)
        indices2, distances2 = knn_index2.build(self.x1, k=20)

        np.testing.assert_equal(indices1, indices2)
        np.testing.assert_equal(distances1, distances2)

    def test_query_same_result_with_fixed_random_state(self):
        knn_index1 = nearest_neighbors.NNDescent("euclidean", random_state=1)
        indices1, distances1 = knn_index1.build(self.x1, k=30)

        knn_index2 = nearest_neighbors.NNDescent("euclidean", random_state=1)
        indices2, distances2 = knn_index2.build(self.x1, k=30)

        np.testing.assert_equal(indices1, indices2)
        np.testing.assert_equal(distances1, distances2)

    @patch("openTSNE.pynndescent.NNDescent", wraps=openTSNE.pynndescent.NNDescent)
    def test_random_state_being_passed_through(self, nndescent):
        random_state = 1
        knn_index = nearest_neighbors.NNDescent("euclidean", random_state=random_state)
        knn_index.build(self.x1, k=30)

        nndescent.assert_called_once()
        check_mock_called_with_kwargs(nndescent, {"random_state": random_state})
