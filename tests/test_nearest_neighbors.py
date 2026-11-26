import base64
import pickle
import platform
import sys
import tempfile
import types
import unittest
import warnings
from os import path
from unittest.mock import patch, MagicMock

import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn import datasets

from sklearn.utils import check_random_state

from openTSNE import affinity, nearest_neighbors
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

    def test_returns_euclidean_distances_query_train(self):
        index: nearest_neighbors.KNNIndex = self.knn_index(self.x1, 15, "euclidean")
        indices, distances = index.build()

        true_distances = np.vstack(
            [cdist(x[None, :], self.x1[idx]) for x, idx in zip(self.x1, indices)]
        )
        np.testing.assert_allclose(distances, true_distances, rtol=1e-4)

    def test_returns_euclidean_distances_query(self):
        index: nearest_neighbors.KNNIndex = self.knn_index(self.x1, 15, "euclidean")
        index.build()
        indices, distances = index.query(self.x2, 15)

        true_distances = np.vstack(
            [cdist(x[None, :], self.x1[idx]) for x, idx in zip(self.x2, indices)]
        )
        np.testing.assert_allclose(distances, true_distances, rtol=1e-4)

    def test_returns_consistent_dtypes(self):
        index: nearest_neighbors.KNNIndex = self.knn_index(self.x1, 15, "euclidean")
        build_indices, build_distances = index.build()
        query_indices, query_distances = index.query(self.x2, 15)

        for indices in (build_indices, query_indices):
            self.assertTrue(np.issubdtype(indices.dtype, np.signedinteger))
        for distances in (build_distances, query_distances):
            self.assertEqual(distances.dtype, np.float64)

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

    def test_build_excludes_point_itself_with_duplicate_points(self):
        # A point and its duplicate are both at distance zero, so the index may
        # return either one first. k = 5 and k = 30 cover the NNDescent paths
        # with and without a follow-up query.
        x = np.vstack([self.x1, self.x1])
        n = self.x1.shape[0]
        twins = np.concatenate([np.arange(n, 2 * n), np.arange(n)])

        for k in [5, 30]:
            with self.subTest(k=k):
                knn_index = self.knn_index(x, k, "euclidean", random_state=0)
                indices, distances = knn_index.build()

                self.assertEqual(indices.shape, (2 * n, k))
                self.assertFalse(np.any(indices == np.arange(2 * n)[:, None]))
                # Each point's duplicate is its nearest neighbor
                np.testing.assert_array_equal(indices[:, 0], twins)
                np.testing.assert_allclose(distances[:, 0], 0, atol=1e-3)


class TestExcludeSelf(unittest.TestCase):
    def test_removes_one_entry_per_row(self):
        indices = np.array([
            [0, 1, 2],  # self first
            [0, 1, 2],  # self in the middle
            [0, 1, 3],  # self absent: drop the furthest
            [3, 0, 3],  # self listed twice: drop the first
        ])
        distances = np.arange(12, dtype=float).reshape(4, 3)

        new_indices, new_distances = nearest_neighbors._exclude_self(
            indices, distances
        )

        np.testing.assert_array_equal(
            new_indices, [[1, 2], [0, 2], [0, 1], [0, 3]]
        )
        np.testing.assert_array_equal(
            new_distances, [[1, 2], [3, 5], [6, 7], [10, 11]]
        )


class TestKNNIndexCoverage(unittest.TestCase):
    # These take neighbors or distances from the user and are tested separately
    not_searching = {
        nearest_neighbors.PrecomputedDistanceMatrix,
        nearest_neighbors.PrecomputedNeighbors,
    }

    def test_every_knn_index_runs_shared_tests(self):
        def subclasses(cls):
            for sub in cls.__subclasses__():
                yield sub
                yield from subclasses(sub)

        indices = {
            cls for cls in subclasses(nearest_neighbors.KNNIndex)
            if cls.__module__ == nearest_neighbors.__name__
        } - self.not_searching
        tested = {cls.knn_index for cls in subclasses(KNNIndexTestMixin)}

        self.assertEqual(
            indices - tested, set(),
            "Every KNNIndex must have a test case that subclasses "
            "KNNIndexTestMixin",
        )


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

    def test_returns_cosine_distances(self):
        knn_index = nearest_neighbors.Annoy(self.x1, 15, "cosine")
        indices, distances = knn_index.build()
        query_indices, query_distances = knn_index.query(self.x2, 15)

        for x, idx, dist in [
            (self.x1, indices, distances),
            (self.x2, query_indices, query_distances),
        ]:
            true_distances = np.vstack(
                [cdist(xi[None, :], self.x1[i], "cosine") for xi, i in zip(x, idx)]
            )
            np.testing.assert_allclose(dist, true_distances, rtol=1e-4, atol=1e-6)

    def test_returns_one_minus_inner_product_for_dot(self):
        # Normalized data, for which one minus the inner product is a distance
        x_train = self.x1 / np.linalg.norm(self.x1, axis=1, keepdims=True)
        x_test = self.x2 / np.linalg.norm(self.x2, axis=1, keepdims=True)

        knn_index = nearest_neighbors.Annoy(x_train, 15, "dot")
        indices, distances = knn_index.build()
        query_indices, query_distances = knn_index.query(x_test, 15)

        for x, idx, dist in [
            (x_train, indices, distances),
            (x_test, query_indices, query_distances),
        ]:
            true_distances = np.vstack([1 - x_train[i] @ xi for xi, i in zip(x, idx)])
            np.testing.assert_allclose(dist, true_distances, rtol=1e-4, atol=1e-6)
            # The most similar neighbors come first
            self.assertTrue(np.all(np.diff(dist, axis=1) >= -1e-6))

    def test_build_excludes_point_itself_with_dot(self):
        knn_index = nearest_neighbors.Annoy(self.x1, 15, "dot")
        indices, distances = knn_index.build()

        self.assertEqual(indices.shape, (self.x1.shape[0], 15))
        self.assertFalse(np.any(indices == np.arange(self.x1.shape[0])[:, None]))

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


@unittest.skipIf(not is_package_installed("mlxvs"), "`mlxvs` is not installed")
class TestMLXVS(KNNIndexTestMixin, unittest.TestCase):
    knn_index = nearest_neighbors.MLXVS

    def test_approximate_neighbors_and_query(self):
        """Above `exact_below` points, the NN-Descent graph and the graph search
        find nearly all of the exact neighbors."""
        x = np.random.RandomState(0).normal(size=(5000, 10))
        exact = nearest_neighbors.Sklearn(x, 15)
        exact_build, _ = exact.build()
        exact_query, _ = exact.query(x[:500], 15)
        for metric in ["euclidean", "cosine"]:
            knn_index = nearest_neighbors.MLXVS(
                x, 15, metric=metric, knn_kwargs={"exact_below": 1000}
            )
            if metric == "cosine":
                exact = nearest_neighbors.Sklearn(x, 15, metric="cosine")
                exact_build, _ = exact.build()
                exact_query, _ = exact.query(x[:500], 15)
            indices, distances = knn_index.build()
            self.assertEqual(indices.shape, (5000, 15))
            self.assertFalse((indices == np.arange(5000)[:, None]).any())
            recall = np.mean([len(set(a) & set(b)) / 15 for a, b in zip(indices, exact_build)])
            self.assertGreater(recall, 0.95)
            indices, distances = knn_index.query(x[:500], 15)
            recall = np.mean([len(set(a) & set(b)) / 15 for a, b in zip(indices, exact_query)])
            self.assertGreater(recall, 0.95)
            self.assertTrue((np.diff(distances, axis=1) >= -1e-12).all())

    def test_pickle_without_built_index(self):
        knn_index = nearest_neighbors.MLXVS(self.iris, k=30)
        self.assertIsNone(knn_index.index)

        with tempfile.TemporaryDirectory() as dirname:
            with open(path.join(dirname, "index.pkl"), "wb") as f:
                pickle.dump(knn_index, f)

            with open(path.join(dirname, "index.pkl"), "rb") as f:
                loaded_obj = pickle.load(f)

        self.assertIsNone(loaded_obj.index)


def _numpy_mlxvs():
    """A stand-in for the mlxvs module that searches exactly with NumPy, so
    that the MLXVS index runs on platforms where mlxvs cannot."""
    def brute_force_knn(base, queries, k, metric):
        # mlxvs returns squared euclidean distances for "l2"
        d = cdist(queries, base, "cosine" if metric == "cosine" else "sqeuclidean")
        indices = np.argsort(d, axis=1, kind="stable")[:, :k]
        return indices.astype(np.uint32), np.take_along_axis(d, indices, axis=1)

    def all_neighbors(rows, k, metric):
        indices, distances = brute_force_knn(rows, rows, k + 1, metric)
        return nearest_neighbors._exclude_self(indices.astype(np.int64), distances)

    class CagraIndex:
        @classmethod
        def build(cls, rows, graph_degree, metric):
            index = cls()
            index.rows, index.graph_degree, index.metric = rows, graph_degree, metric
            return index

        def search(self, queries, k, itopk_size):
            self.itopk_size = itopk_size
            return brute_force_knn(self.rows, queries, k, self.metric)

    module = types.ModuleType("mlxvs")
    module.brute_force_knn = brute_force_knn
    module.all_neighbors = all_neighbors
    module.CagraIndex = CagraIndex
    return module


class TestMLXVSOnAnyPlatform(unittest.TestCase):
    """The MLXVS index outside of mlxvs: the errors on platforms mlxvs does not
    support, and the index with a NumPy stand-in for mlxvs, on every platform."""

    def setUp(self):
        random_state = np.random.RandomState(0)
        self.x_train = random_state.normal(0, 1, (200, 5))
        self.x_test = random_state.normal(0, 1, (50, 5))

    @staticmethod
    def _platform(name, machine):
        return [
            patch.object(sys, "platform", name),
            patch("platform.machine", return_value=machine),
        ]

    def test_unsupported_platform_raises_import_error(self):
        platforms = [
            ("linux", "x86_64"),
            ("linux", "aarch64"),
            ("win32", "AMD64"),
            ("darwin", "x86_64"),  # an Intel Mac, or Intel Python on Rosetta
        ]
        for name, machine in platforms:
            with self.subTest(platform=name, machine=machine):
                patches = self._platform(name, machine)
                with patches[0], patches[1], patch.dict(sys.modules, {"mlxvs": _numpy_mlxvs()}):
                    with self.assertRaisesRegex(ImportError, "Apple silicon"):
                        nearest_neighbors.MLXVS(self.x_train, 15)
                    with self.assertRaisesRegex(ImportError, "Apple silicon"):
                        affinity.PerplexityBasedNN(
                            self.x_train, perplexity=5, method="mlxvs"
                        )

    def test_missing_package_raises_import_error(self):
        patches = self._platform("darwin", "arm64")
        # `None` in `sys.modules` makes `import mlxvs` raise an ImportError
        with patches[0], patches[1], patch.dict(sys.modules, {"mlxvs": None}):
            with self.assertRaisesRegex(ImportError, "pip install mlxvs"):
                nearest_neighbors.MLXVS(self.x_train, 15)

    def test_same_result_as_exact_search(self):
        k = 15
        patches = self._platform("darwin", "arm64")
        with patches[0], patches[1], patch.dict(sys.modules, {"mlxvs": _numpy_mlxvs()}):
            for metric in ["euclidean", "l2", "cosine"]:
                exact_index = nearest_neighbors.Sklearn(self.x_train, k, metric)
                true_indices, true_distances = exact_index.build()
                true_query_indices, true_query_distances = exact_index.query(
                    self.x_test, k
                )
                # Below `exact_below` points, and above, through the graph paths
                for exact_below in [nearest_neighbors.MLXVS.EXACT_BELOW, 100]:
                    with self.subTest(metric=metric, exact_below=exact_below):
                        knn_index = nearest_neighbors.MLXVS(
                            self.x_train, k, metric,
                            knn_kwargs=dict(exact_below=exact_below),
                        )
                        indices, distances = knn_index.build()
                        query_indices, query_distances = knn_index.query(
                            self.x_test, k
                        )

                        self.assertEqual(indices.dtype, np.int64)
                        self.assertEqual(distances.dtype, np.float64)
                        np.testing.assert_array_equal(indices, true_indices)
                        np.testing.assert_allclose(distances, true_distances)
                        np.testing.assert_array_equal(
                            query_indices, true_query_indices
                        )
                        np.testing.assert_allclose(
                            query_distances, true_query_distances
                        )

    def test_graph_settings_from_knn_kwargs(self):
        patches = self._platform("darwin", "arm64")
        with patches[0], patches[1], patch.dict(sys.modules, {"mlxvs": _numpy_mlxvs()}):
            knn_index = nearest_neighbors.MLXVS(
                self.x_train, 5, knn_kwargs=dict(exact_below=100)
            )
            knn_index.query(self.x_test, 5)
            self.assertEqual(knn_index.index.graph_degree, 32)
            self.assertEqual(knn_index.index.itopk_size, 64)

            knn_index = nearest_neighbors.MLXVS(
                self.x_train, 5,
                knn_kwargs=dict(exact_below=100, graph_degree=16, itopk_size=8),
            )
            knn_index.query(self.x_test, 15)
            self.assertEqual(knn_index.index.graph_degree, 16)
            # The search keeps at least k candidates
            self.assertEqual(knn_index.index.itopk_size, 15)

    def test_distances_in_blocks(self):
        patches = self._platform("darwin", "arm64")
        with patches[0], patches[1], patch.dict(sys.modules, {"mlxvs": _numpy_mlxvs()}):
            knn_index = nearest_neighbors.MLXVS(self.x_train, 15)
            expected = knn_index.build()
            # A bound that fits only a few points per block
            with patch.object(nearest_neighbors.MLXVS, "DISTANCE_BLOCK_BYTES", 3000):
                indices, distances = knn_index.build()
            np.testing.assert_array_equal(indices, expected[0])
            np.testing.assert_allclose(distances, expected[1])


@unittest.skipIf(not is_package_installed("hnswlib"), "`hnswlib`is not installed")
class TestHNSW(KNNIndexTestMixin, unittest.TestCase):
    knn_index = nearest_neighbors.HNSW

    @classmethod
    def setUpClass(cls):
        global hnswlib
        import hnswlib

    def test_pickle_without_built_index(self):
        knn_index = nearest_neighbors.HNSW(self.iris, k=30)
        self.assertIsNone(knn_index.index)

        with tempfile.TemporaryDirectory() as dirname:
            with open(path.join(dirname, "index.pkl"), "wb") as f:
                pickle.dump(knn_index, f)

            with open(path.join(dirname, "index.pkl"), "rb") as f:
                loaded_obj = pickle.load(f)

        self.assertIsNone(loaded_obj.index)

    def test_pickle_without_built_index_cleans_up_fname(self):
        knn_index = nearest_neighbors.HNSW(self.iris, k=30)
        with tempfile.TemporaryDirectory() as dirname:
            with open(path.join(dirname, "index.pkl"), "wb") as f:
                pickle.dump(knn_index, f)

            with open(path.join(dirname, "index.pkl"), "rb") as f:
                loaded_obj = pickle.load(f)

        self.assertIsNone(loaded_obj.index)

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

    def test_pickle_with_built_index_preserves_metric(self):
        for metric in nearest_neighbors.HNSW.VALID_METRICS:
            with self.subTest(metric=metric):
                knn_index = nearest_neighbors.HNSW(self.iris, k=30, metric=metric)
                knn_index.build()

                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    loaded_obj = pickle.loads(pickle.dumps(knn_index))
                self.assertEqual(loaded_obj.metric, metric)

                load_idx, load_dist = loaded_obj.query(self.iris, 15)
                orig_idx, orig_dist = knn_index.query(self.iris, 15)

                np.testing.assert_array_equal(load_idx, orig_idx)
                np.testing.assert_array_almost_equal(load_dist, orig_dist)

    @unittest.skipIf(not is_package_installed("pandas"), "`pandas` is not installed")
    def test_pickle_with_built_index_on_dataframe(self):
        import pandas as pd

        knn_index = nearest_neighbors.HNSW(pd.DataFrame(self.iris), k=30)
        knn_index.build()

        loaded_obj = pickle.loads(pickle.dumps(knn_index))

        load_idx, load_dist = loaded_obj.query(self.iris, 15)
        orig_idx, orig_dist = knn_index.query(self.iris, 15)

        np.testing.assert_array_equal(load_idx, orig_idx)
        np.testing.assert_array_almost_equal(load_dist, orig_dist)

    @staticmethod
    def _base64_encoded_state(knn_index):
        """The pickled state written by openTSNE<=1.0.4, which stores the index
        as a base64-encoded hnswlib index file."""
        state = dict(knn_index.__dict__)
        with tempfile.TemporaryDirectory() as dirname:
            fname = path.join(dirname, "index.bin")
            state.pop("index").save_index(fname)
            with open(fname, "rb") as f:
                state["b64_index"] = base64.b64encode(f.read())
        return state

    def _assert_loads_base64_encoded_state(self, data):
        knn_index = nearest_neighbors.HNSW(data, k=30, metric="cosine")
        knn_index.build()

        # Unpickling creates the instance without calling `__init__`
        loaded_obj = nearest_neighbors.HNSW.__new__(nearest_neighbors.HNSW)
        with self.assertWarns(FutureWarning):
            loaded_obj.__setstate__(self._base64_encoded_state(knn_index))
        self.assertIsInstance(loaded_obj.index, hnswlib.Index)
        self.assertEqual(loaded_obj.index.space, "cosine")

        load_idx, load_dist = loaded_obj.query(self.iris, 15)
        orig_idx, orig_dist = knn_index.query(self.iris, 15)

        np.testing.assert_array_equal(load_idx, orig_idx)
        np.testing.assert_array_almost_equal(load_dist, orig_dist)

    def test_unpickle_base64_encoded_index(self):
        self._assert_loads_base64_encoded_state(self.iris)

    @unittest.skipIf(not is_package_installed("pandas"), "`pandas` is not installed")
    def test_unpickle_base64_encoded_index_on_dataframe(self):
        import pandas as pd

        self._assert_loads_base64_encoded_state(pd.DataFrame(self.iris))

    def test_returns_int64_indices_and_float64_distances(self):
        for metric in nearest_neighbors.HNSW.VALID_METRICS:
            with self.subTest(metric=metric):
                knn_index = nearest_neighbors.HNSW(self.x1, 15, metric)
                build_indices, build_distances = knn_index.build()
                query_indices, query_distances = knn_index.query(self.x2, 15)

                self.assertEqual(build_indices.dtype, np.int64)
                self.assertEqual(query_indices.dtype, np.int64)
                self.assertEqual(build_distances.dtype, np.float64)
                self.assertEqual(query_distances.dtype, np.float64)

    def test_same_result_as_exact_search(self):
        # Low-dimensional data, on which HNSW finds the exact neighbors
        random_state = np.random.RandomState(0)
        x_train = random_state.normal(0, 1, (200, 5))
        x_test = random_state.normal(0, 1, (50, 5))

        k = 15
        for metric in ["euclidean", "l2", "cosine"]:
            with self.subTest(metric=metric):
                exact_index = nearest_neighbors.Sklearn(x_train, k, metric)
                true_indices, true_distances = exact_index.build()
                true_query_indices, true_query_distances = exact_index.query(x_test, k)

                knn_index = nearest_neighbors.HNSW(x_train, k, metric, random_state=0)
                indices, distances = knn_index.build()
                query_indices, query_distances = knn_index.query(x_test, k)

                np.testing.assert_array_equal(indices, true_indices)
                np.testing.assert_allclose(distances, true_distances, rtol=1e-4)
                np.testing.assert_array_equal(query_indices, true_query_indices)
                np.testing.assert_allclose(
                    query_distances, true_query_distances, rtol=1e-4
                )

    def test_build_excludes_point_itself_with_inner_product(self):
        for metric in ["ip", "dot"]:
            with self.subTest(metric=metric):
                knn_index = nearest_neighbors.HNSW(self.x1, 15, metric)
                indices, distances = knn_index.build()

                self.assertEqual(indices.shape, (self.x1.shape[0], 15))
                self.assertFalse(
                    np.any(indices == np.arange(self.x1.shape[0])[:, None])
                )

    def test_default_ef(self):
        knn_index = nearest_neighbors.HNSW(self.x1, 5, "euclidean")
        knn_index.build()
        self.assertEqual(knn_index.index.ef, 100)

        knn_index.query(self.x2, 60)
        self.assertEqual(knn_index.index.ef, 120)

        # `ef` cannot exceed the number of indexed points
        knn_index.query(self.x2, 90)
        self.assertEqual(knn_index.index.ef, self.x1.shape[0])

    def test_ef_from_knn_kwargs(self):
        knn_index = nearest_neighbors.HNSW(
            self.x1, 5, "euclidean", knn_kwargs=dict(ef=50)
        )
        knn_index.build()
        self.assertEqual(knn_index.index.ef, 50)

        knn_index.query(self.x2, 15)
        self.assertEqual(knn_index.index.ef, 50)

    def test_negative_n_jobs_follows_sklearn_convention(self):
        import joblib

        n_cores = joblib.cpu_count()
        for n_jobs, num_threads in [(-1, n_cores), (-2, max(n_cores - 1, 1))]:
            with self.subTest(n_jobs=n_jobs):
                knn_index = nearest_neighbors.HNSW(self.x1, 5, n_jobs=n_jobs)
                knn_index.build()
                knn_index.index = MagicMock(wraps=knn_index.index)
                knn_index.query(self.x2, 5)
                check_mock_called_with_kwargs(
                    knn_index.index.knn_query, dict(num_threads=num_threads)
                )

    def test_warns_if_random_state_with_multiple_threads(self):
        knn_index = nearest_neighbors.HNSW(self.x1, 5, n_jobs=2, random_state=1)
        with self.assertWarnsRegex(UserWarning, "not deterministic"):
            knn_index.build()

    def test_no_warning_if_random_state_with_single_thread(self):
        for n_jobs, random_state in [(1, 1), (2, None)]:
            with self.subTest(n_jobs=n_jobs, random_state=random_state):
                knn_index = nearest_neighbors.HNSW(
                    self.x1, 5, n_jobs=n_jobs, random_state=random_state
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    knn_index.build()

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

    def test_check_metric_warns_only_when_metrics_are_removed(self):
        import pynndescent

        knn_index = nearest_neighbors.NNDescent(self.x1, 30, "euclidean")
        named = dict.fromkeys(knn_index.VALID_METRICS)

        # A newer pynndescent with extra metrics: nothing to warn about
        with patch.object(
            pynndescent.distances, "named_distances", {**named, "new_metric": None}
        ), warnings.catch_warnings():
            warnings.simplefilter("error")
            knn_index.check_metric("euclidean")

        # A metric openTSNE accepts is gone from pynndescent
        del named["yule"]
        with patch.object(pynndescent.distances, "named_distances", named):
            with self.assertWarnsRegex(UserWarning, "recently changed"):
                knn_index.check_metric("euclidean")

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
