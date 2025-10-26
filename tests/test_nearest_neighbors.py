import pickle
import platform
import tempfile
import unittest
from os import path
from unittest.mock import patch, MagicMock

import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn import datasets

from sklearn.utils import check_random_state

from openTSNE import nearest_neighbors
from openTSNE.utils import is_package_installed
from .test_tsne import check_mock_called_with_kwargs


class KNNIndexTestMixin:
    knn_index = NotImplemented

    def __init__(self, *args, **kwargs):
        self.x1 = np.random.normal(100, 50, (150, 50))
        self.x2 = np.random.normal(100, 50, (100, 50))
        self.iris = datasets.load_iris().data
        super().__init__(*args, **kwargs)

    def test_returns_correct_number_neighbors_query_train(self):
        ks = [1, 5, 10, 30, 50]
        n_samples = self.x1.shape[0]

        for k in ks:
            index: nearest_neighbors.KNNIndex = self.knn_index(self.x1, k, "euclidean")
            indices, distances = index.build()
            self.assertEqual(indices.shape, (n_samples, k))
            self.assertEqual(distances.shape, (n_samples, k))

    def test_returns_proper_distances_query_train(self):
        index: nearest_neighbors.KNNIndex = self.knn_index(self.iris, 30, "euclidean")
        indices, distances = index.build()
        self.assertTrue(np.isfinite(distances).all())

    def test_returns_correct_number_neighbors_query(self):
        ks = [1, 5, 10, 30, 50]
        n_samples = self.x2.shape[0]
        index: nearest_neighbors.KNNIndex = self.knn_index(self.x1, 30, "euclidean")
        index.build()

        for k in ks:
            indices, distances = index.query(self.x2, k)
            self.assertEqual(indices.shape, (n_samples, k))
            self.assertEqual(distances.shape, (n_samples, k))

    def test_query_train_same_result_with_fixed_random_state(self):
        knn_index1 = self.knn_index(self.x1, 20, "euclidean", random_state=1)
        indices1, distances1 = knn_index1.build()

        knn_index2 = self.knn_index(self.x1, 20, "euclidean", random_state=1)
        indices2, distances2 = knn_index2.build()

        np.testing.assert_equal(indices1, indices2)
        np.testing.assert_equal(distances1, distances2)

    def test_query_same_result_with_fixed_random_state(self):
        knn_index1 = self.knn_index(self.x1, 30, "euclidean", random_state=1)
        indices1, distances1 = knn_index1.build()

        knn_index2 = self.knn_index(self.x1, 30, "euclidean", random_state=1)
        indices2, distances2 = knn_index2.build()

        np.testing.assert_equal(indices1, indices2)
        np.testing.assert_equal(distances1, distances2)

    def test_query_same_result_with_fixed_random_state_instance(self):
        random_state = np.random.RandomState(42)
        knn_index1 = self.knn_index(self.x1, 30, "euclidean", random_state=random_state)
        indices1, distances1 = knn_index1.build()

        random_state = np.random.RandomState(42)
        knn_index2 = self.knn_index(self.x1, 30, "euclidean", random_state=random_state)
        indices2, distances2 = knn_index2.build()

        np.testing.assert_equal(indices1, indices2)
        np.testing.assert_equal(distances1, distances2)


class TestAnnoy(KNNIndexTestMixin, unittest.TestCase):
    knn_index = nearest_neighbors.Annoy

    @unittest.skipIf(platform.system() == "Windows", "Files locked on Windows")
    def test_pickle_without_built_index(self):
        knn_index = nearest_neighbors.Annoy(self.iris, k=30)
        self.assertIsNone(knn_index.index)

        with tempfile.TemporaryDirectory() as dirname:
            with open(path.join(dirname, "index.pkl"), "wb") as f:
                pickle.dump(knn_index, f)

            with open(path.join(dirname, "index.pkl"), "rb") as f:
                loaded_obj = pickle.load(f)

        self.assertIsNone(loaded_obj.index)

    @unittest.skipIf(platform.system() == "Windows", "Files locked on Windows")
    def test_pickle_without_built_index_cleans_up_fname(self):
        knn_index = nearest_neighbors.Annoy(self.iris, k=30)
        with tempfile.TemporaryDirectory() as dirname:
            with open(path.join(dirname, "index.pkl"), "wb") as f:
                pickle.dump(knn_index, f)

            with open(path.join(dirname, "index.pkl"), "rb") as f:
                loaded_obj = pickle.load(f)

        self.assertIsNone(loaded_obj.index)

    @unittest.skipIf(platform.system() == "Windows", "Files locked on Windows")
    def test_pickle_with_built_index(self):
        knn_index = nearest_neighbors.Annoy(self.iris, k=30)
        knn_index.build()
        self.assertIsNotNone(knn_index.index)

        with tempfile.TemporaryDirectory() as dirname:
            with open(path.join(dirname, "index.pkl"), "wb") as f:
                pickle.dump(knn_index, f)

            with open(path.join(dirname, "index.pkl"), "rb") as f:
                loaded_obj = pickle.load(f)

        load_idx, load_dist = loaded_obj.query(self.iris, 15)
        orig_idx, orig_dist = knn_index.query(self.iris, 15)

        np.testing.assert_array_equal(load_idx, orig_idx)
        np.testing.assert_array_almost_equal(load_dist, orig_dist)

    def test_knn_kwargs(self):
        with patch(
            "openTSNE.dependencies.annoy.AnnoyIndex",
            autospec=True,
        ) as mock:
            params = dict(n_trees=10)
            knn_index = nearest_neighbors.Annoy(
                self.x1, 5, knn_kwargs=params
            )
            try:
                knn_index.build()
            except ValueError as e:
                # The build fails once we try to query the index, but at that
                # point, build was already called, and we can tell if it was
                # called properly
                if "not enough values to unpack (expected 2, got 0)" not in str(e):
                    raise

            check_mock_called_with_kwargs(mock.return_value.build, params)


class TestSklearn(KNNIndexTestMixin, unittest.TestCase):
    knn_index = nearest_neighbors.Sklearn

    def test_cosine_distance(self):
        k = 15
        # Compute cosine distance nearest neighbors using ball tree
        knn_index = self.knn_index(self.x1, k, "cosine")
        indices, distances = knn_index.build()

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
        knn_index = self.knn_index(self.x1, k, "cosine")
        knn_index.build()

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

    def test_uncompiled_callable_metric_same_result(self):
        k = 15

        knn_index = self.knn_index(self.x1, k, "manhattan", random_state=1)
        knn_index.build()
        true_indices_, true_distances_ = knn_index.query(self.x2, k=k)

        def manhattan(x, y):
            result = 0.0
            for i in range(x.shape[0]):
                result += np.abs(x[i] - y[i])

            return result

        knn_index = self.knn_index(self.x1, k, manhattan, random_state=1)
        knn_index.build()
        indices, distances = knn_index.query(self.x2, k=k)
        np.testing.assert_array_equal(
            indices, true_indices_, err_msg="Nearest neighbors do not match"
        )
        np.testing.assert_allclose(
            distances, true_distances_, err_msg="Distances do not match"
        )

    def test_knn_kwargs(self):
        import sklearn

        with patch(
            "sklearn.neighbors.NearestNeighbors",
            wraps=sklearn.neighbors.NearestNeighbors,
        ) as mock:
            params = dict(algorithm="kd_tree", leaf_size=10)
            knn_index = nearest_neighbors.Sklearn(
                self.x1, 5, knn_kwargs=params
            )
            knn_index.build()
            check_mock_called_with_kwargs(mock, params)


@unittest.skipIf(not is_package_installed("hnswlib"), "`hnswlib`is not installed")
class TestHNSW(KNNIndexTestMixin, unittest.TestCase):
    knn_index = nearest_neighbors.HNSW

    @classmethod
    def setUpClass(cls):
        global hnswlib
        import hnswlib

    @unittest.skipIf(platform.system() == "Windows", "Files locked on Windows")
    def test_pickle_without_built_index(self):
        knn_index = nearest_neighbors.HNSW(self.iris, k=30)
        self.assertIsNone(knn_index.index)

        with tempfile.TemporaryDirectory() as dirname:
            with open(path.join(dirname, "index.pkl"), "wb") as f:
                pickle.dump(knn_index, f)

            with open(path.join(dirname, "index.pkl"), "rb") as f:
                loaded_obj = pickle.load(f)

        self.assertIsNone(loaded_obj.index)

    @unittest.skipIf(platform.system() == "Windows", "Files locked on Windows")
    def test_pickle_without_built_index_cleans_up_fname(self):
        knn_index = nearest_neighbors.HNSW(self.iris, k=30)
        with tempfile.TemporaryDirectory() as dirname:
            with open(path.join(dirname, "index.pkl"), "wb") as f:
                pickle.dump(knn_index, f)

            with open(path.join(dirname, "index.pkl"), "rb") as f:
                loaded_obj = pickle.load(f)

        self.assertIsNone(loaded_obj.index)

    @unittest.skipIf(platform.system() == "Windows", "Files locked on Windows")
    def test_pickle_with_built_index(self):
        knn_index = nearest_neighbors.HNSW(self.iris, k=30)
        knn_index.build()
        self.assertIsNotNone(knn_index.index)

        with tempfile.TemporaryDirectory() as dirname:
            with open(path.join(dirname, "index.pkl"), "wb") as f:
                pickle.dump(knn_index, f)

            with open(path.join(dirname, "index.pkl"), "rb") as f:
                loaded_obj = pickle.load(f)

        load_idx, load_dist = loaded_obj.query(self.iris, 15)
        orig_idx, orig_dist = knn_index.query(self.iris, 15)

        np.testing.assert_array_equal(load_idx, orig_idx)
        np.testing.assert_array_almost_equal(load_dist, orig_dist)

    def test_knn_kwargs(self):
        with patch(
            "hnswlib.Index",
            autospec=True,
        ) as mock:
            mock.init_index = MagicMock(wraps=mock.init_index)

            params = dict(ef_construction=100, M=32)
            knn_index = nearest_neighbors.HNSW(
                self.x1, 5, knn_kwargs=params
            )
            try:
                knn_index.build()
            except TypeError:
                pass

            check_mock_called_with_kwargs(mock.return_value.init_index, params)


@unittest.skipIf(not is_package_installed("pynndescent"), "`pynndescent`is not installed")
class TestNNDescent(KNNIndexTestMixin, unittest.TestCase):
    knn_index = nearest_neighbors.NNDescent

    @classmethod
    def setUpClass(cls):
        global pynndescent, njit, CPUDispatcher

        import pynndescent
        from numba import njit
        from numba.core.registry import CPUDispatcher

    def test_random_state_being_passed_through(self):
        random_state = 1
        with patch("pynndescent.NNDescent", wraps=pynndescent.NNDescent) as nndescent:
            knn_index = nearest_neighbors.NNDescent(
                self.x1, 30, "euclidean", random_state=random_state
            )
            knn_index.build()

            nndescent.assert_called_once()
            check_mock_called_with_kwargs(nndescent, {"random_state": random_state})

    def test_uncompiled_callable_is_compiled(self):
        knn_index = nearest_neighbors.NNDescent(self.x1, 30, "manhattan")

        def manhattan(x, y):
            result = 0.0
            for i in range(x.shape[0]):
                result += np.abs(x[i] - y[i])

            return result

        compiled_metric = knn_index.check_metric(manhattan)
        self.assertTrue(isinstance(compiled_metric, CPUDispatcher))

    def test_uncompiled_callable_metric_same_result(self):
        k = 15

        knn_index = self.knn_index(self.x1, k, "manhattan", random_state=1)
        knn_index.build()
        true_indices_, true_distances_ = knn_index.query(self.x2, k=k)

        def manhattan(x, y):
            result = 0.0
            for i in range(x.shape[0]):
                result += np.abs(x[i] - y[i])

            return result

        knn_index = self.knn_index(self.x1, k, manhattan, random_state=1)
        knn_index.build()
        indices, distances = knn_index.query(self.x2, k=k)
        np.testing.assert_array_equal(
            indices, true_indices_, err_msg="Nearest neighbors do not match"
        )
        np.testing.assert_allclose(
            distances, true_distances_, err_msg="Distances do not match"
        )

    def test_numba_compiled_callable_metric_same_result(self):
        k = 15

        knn_index = self.knn_index(self.x1, k, "manhattan", random_state=1)
        knn_index.build()
        true_indices_, true_distances_ = knn_index.query(self.x2, k=k)

        @njit(fastmath=True)
        def manhattan(x, y):
            result = 0.0
            for i in range(x.shape[0]):
                result += np.abs(x[i] - y[i])

            return result

        knn_index = self.knn_index(self.x1, k, manhattan, random_state=1)
        knn_index.build()
        indices, distances = knn_index.query(self.x2, k=k)
        np.testing.assert_array_equal(
            indices, true_indices_, err_msg="Nearest neighbors do not match"
        )
        np.testing.assert_allclose(
            distances, true_distances_, err_msg="Distances do not match"
        )

    def test_building_with_lt15_builds_proper_graph(self):
        with patch("pynndescent.NNDescent", wraps=pynndescent.NNDescent) as nndescent:
            knn_index = nearest_neighbors.NNDescent(self.x1, 10, "euclidean")
            indices, distances = knn_index.build()

            self.assertEqual(indices.shape, (self.x1.shape[0], 10))
            self.assertEqual(distances.shape, (self.x1.shape[0], 10))
            self.assertFalse(np.all(indices[:, 0] == np.arange(self.x1.shape[0])))

        # Should be called with 11 because nearest neighbor in pynndescent is itself
        check_mock_called_with_kwargs(nndescent, dict(n_neighbors=11))

    def test_runs_with_correct_njobs_if_dense_input(self):
        with patch("pynndescent.NNDescent", wraps=pynndescent.NNDescent) as nndescent:
            knn_index = nearest_neighbors.NNDescent(self.x1, 5, "euclidean", n_jobs=2)
            knn_index.build()
            check_mock_called_with_kwargs(nndescent, dict(n_jobs=2))

    def test_runs_with_correct_njobs_if_sparse_input(self):
        with patch("pynndescent.NNDescent", wraps=pynndescent.NNDescent) as nndescent:
            x_sparse = sp.csr_matrix(self.x1)
            knn_index = nearest_neighbors.NNDescent(x_sparse, 5, "euclidean", n_jobs=2)
            knn_index.build()
            check_mock_called_with_kwargs(nndescent, dict(n_jobs=2))

    def test_random_cluster_when_invalid_indices(self):
        class MockIndex:
            def __init__(self, data, n_neighbors, **_):
                n_samples = data.shape[0]

                rs = check_random_state(0)
                indices = rs.randint(0, n_samples, size=(n_samples, n_neighbors))
                distances = rs.exponential(5, (n_samples, n_neighbors))

                # Set some of the points to have invalid indices
                indices[:10] = -1
                distances[:10] = -1

                self.neighbor_graph = indices, distances

        with patch("pynndescent.NNDescent", wraps=MockIndex):
            knn_index = nearest_neighbors.NNDescent(self.x1, 5, "euclidean", n_jobs=2)
            indices, distances = knn_index.build()

            # Check that indices were replaced by something
            self.assertTrue(np.all(indices[:10] != -1))
            # Check that that "something" are all indices of failed points
            self.assertTrue(np.all(indices[:10] < 10))
            # And check that the distances were set to something positive
            self.assertTrue(np.all(distances[:10] > 0))

    def test_knn_kwargs(self):
        with patch("pynndescent.NNDescent", wraps=pynndescent.NNDescent) as mock:
            params = dict(n_trees=15, n_iters=3, max_candidates=30, leaf_size=10)
            knn_index = nearest_neighbors.NNDescent(
                self.x1, 5, knn_kwargs=params
            )
            knn_index.build()
            check_mock_called_with_kwargs(mock, params)
