import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform
import pynndescent
from sklearn import datasets

from numba import njit
from numba.targets.registry import CPUDispatcher

from openTSNE import nearest_neighbors
from .test_tsne import check_mock_called_with_kwargs


class KNNIndexTestMixin:
    knn_index = None

    def __init__(self, *args, **kwargs):
        self.x1 = np.random.normal(100, 50, (150, 50))
        self.x2 = np.random.normal(100, 50, (100, 50))
        self.iris = datasets.load_iris().data
        super().__init__(*args, **kwargs)

    def test_returns_correct_number_neighbors_query_train(self):
        ks = [1, 5, 10, 30, 50]
        n_samples = self.x1.shape[0]
        index: nearest_neighbors.KNNIndex = self.knn_index("euclidean")

        for k in ks:
            indices, distances = index.build(self.x1, k=k)
            self.assertEqual(indices.shape, (n_samples, k))
            self.assertEqual(distances.shape, (n_samples, k))

    def test_returns_proper_distances_query_train(self):
        index: nearest_neighbors.KNNIndex = self.knn_index("euclidean")
        indices, distances = index.build(self.iris, k=30)
        self.assertTrue(np.isfinite(distances).all())

    def test_returns_correct_number_neighbors_query(self):
        ks = [1, 5, 10, 30, 50]
        n_samples = self.x2.shape[0]
        index: nearest_neighbors.KNNIndex = self.knn_index("euclidean")
        index.build(self.x1, k=30)

        for k in ks:
            indices, distances = index.query(self.x2, k)
            self.assertEqual(indices.shape, (n_samples, k))
            self.assertEqual(distances.shape, (n_samples, k))

    def test_query_train_same_result_with_fixed_random_state(self):
        knn_index1 = self.knn_index("euclidean", random_state=1)
        indices1, distances1 = knn_index1.build(self.x1, k=20)

        knn_index2 = self.knn_index("euclidean", random_state=1)
        indices2, distances2 = knn_index2.build(self.x1, k=20)

        np.testing.assert_equal(indices1, indices2)
        np.testing.assert_equal(distances1, distances2)

    def test_query_same_result_with_fixed_random_state(self):
        knn_index1 = self.knn_index("euclidean", random_state=1)
        indices1, distances1 = knn_index1.build(self.x1, k=30)

        knn_index2 = self.knn_index("euclidean", random_state=1)
        indices2, distances2 = knn_index2.build(self.x1, k=30)

        np.testing.assert_equal(indices1, indices2)
        np.testing.assert_equal(distances1, distances2)

    def test_uncompiled_callable_metric_same_result(self):

        knn_index = self.knn_index("manhattan")
        knn_index.build(self.x1, k=k)
        true_indices, true_distances = knn_index.query(self.x2, k=k)

        def manhattan(x, y):
            return np.sum(np.abs(x - y))

        knn_index = self.knn_index(manhattan)
        knn_index.build(self.x1, k=k)
        indices, distances = knn_index.query(self.x2, k=k)
        np.testing.assert_array_equal(
            indices, true_indices_, err_msg="Nearest neighbors do not match"
        )
        np.testing.assert_allclose(
            distances, true_distances_, err_msg="Distances do not match"
        )

    def test_numba_compiled_callable_metric_same_result(self):

        knn_index = self.knn_index("manhattan")
        knn_index.build(self.x1, k=k)
        true_indices, true_distances = knn_index.query(self.x2, k=k)

        @njit()
        def manhattan(x, y):
            return np.sum(np.abs(x - y))

        knn_index = self.knn_index(manhattan)
        knn_index.build(self.x1, k=k)
        indices, distances = knn_index.query(self.x2, k=k)
        np.testing.assert_array_equal(
            indices, true_indices_, err_msg="Nearest neighbors do not match"
        )
        np.testing.assert_allclose(
            distances, true_distances_, err_msg="Distances do not match"
        )


class TestBallTree(KNNIndexTestMixin, unittest.TestCase):
    knn_index = nearest_neighbors.BallTree

    def test_cosine_distance(self):
        k = 15
        # Compute cosine distance nearest neighbors using ball tree
        knn_index = nearest_neighbors.BallTree("cosine")
        indices, distances = knn_index.build(self.x1, k=k)

        # Compute the exact nearest neighbors as a reference
        true_distances = squareform(pdist(self.x1, metric="cosine"))
        true_indices_ = np.argsort(true_distances, axis=1)[:, 1:k + 1]
        true_distances_ = np.vstack([d[i] for d, i in zip(true_distances, true_indices_)])

        np.testing.assert_array_equal(
            indices, true_indices_, err_msg="Nearest neighbors do not match"
        )
        np.testing.assert_array_equal(
            distances, true_distances_, err_msg="Distances do not match"
        )

    def test_cosine_distance_query(self):
        k = 15
        # Compute cosine distance nearest neighbors using ball tree
        knn_index = nearest_neighbors.BallTree("cosine")
        knn_index.build(self.x1, k=k)

        indices, distances = knn_index.query(self.x2, k=k)

        # Compute the exact nearest neighbors as a reference
        true_distances = cdist(self.x2, self.x1, metric="cosine")
        true_indices_ = np.argsort(true_distances, axis=1)[:, :k]
        true_distances_ = np.vstack([d[i] for d, i in zip(true_distances, true_indices_)])

        np.testing.assert_array_equal(
            indices, true_indices_, err_msg="Nearest neighbors do not match"
        )
        np.testing.assert_array_equal(
            distances, true_distances_, err_msg="Distances do not match"
        )


class TestNNDescent(KNNIndexTestMixin, unittest.TestCase):
    knn_index = nearest_neighbors.NNDescent

    @patch("pynndescent.NNDescent", wraps=pynndescent.NNDescent)
    def test_random_state_being_passed_through(self, nndescent):
        random_state = 1
        knn_index = nearest_neighbors.NNDescent("euclidean", random_state=random_state)
        knn_index.build(self.x1, k=30)

        nndescent.assert_called_once()
        check_mock_called_with_kwargs(nndescent, {"random_state": random_state})

    def test_uncompiled_callable_is_compiled(self):

        knn_index = nearest_neighbors.NNDescent("manhattan")

        def manhattan(x, y):
            return np.sum(np.abs(x - y))

        compiled_metric = knn_index.check_metric(manhattan)
        self.assertTrue(isinstance(compiled_metric, CPUDispatcher))
