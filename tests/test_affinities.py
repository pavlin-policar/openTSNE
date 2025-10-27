import logging
import unittest
from functools import partial
from unittest.mock import patch

import numpy as np
from scipy.spatial.distance import squareform, pdist
from sklearn import datasets
from sklearn.model_selection import train_test_split

from openTSNE import affinity, nearest_neighbors
from openTSNE.utils import is_package_installed
from tests.test_tsne import check_mock_called_with_kwargs

affinity.log.setLevel(logging.ERROR)

Multiscale = partial(affinity.Multiscale, method="exact")
MultiscaleMixture = partial(affinity.MultiscaleMixture, method="exact")
PerplexityBasedNN = partial(affinity.PerplexityBasedNN, method="exact")
FixedSigmaNN = partial(affinity.FixedSigmaNN, method="exact")
Uniform = partial(affinity.Uniform, method="exact")

AFFINITY_CLASSES = [
    ("PerplexityBasedNN", PerplexityBasedNN),
    ("FixedSigmaNN", partial(FixedSigmaNN, sigma=1)),
    ("MultiscaleMixture", partial(MultiscaleMixture, perplexities=[10, 20])),
    ("Multiscale", partial(Multiscale, perplexities=[10, 20])),
    ("UniformMean", partial(Uniform, k_neighbors=5, symmetrize="mean")),
    ("UniformMax", partial(Uniform, k_neighbors=5, symmetrize="max")),
]


class TestPerplexityBased(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x = np.random.normal(100, 50, (91, 4))

    def test_doesnt_change_perplexity_parameter_value(self):
        aff = PerplexityBasedNN(self.x, perplexity=140)
        self.assertEqual(aff.perplexity, 140)
        self.assertEqual(aff.effective_perplexity_, 30)

    def test_handles_reducing_perplexity_value(self):
        perplexity = 20  # initial perplexity values
        k_neighbors = perplexity * 3
        aff = PerplexityBasedNN(self.x, perplexity=perplexity)

        self.assertEqual(aff.perplexity, perplexity)
        self.assertEqual(aff.effective_perplexity_, perplexity)

        # Check that the initial `P` matrix is allright
        n_samples = self.x.shape[0]
        original_P = aff.P.copy()
        # Can't check for equality because the matrix is symmetrized therefore
        # each point may have non-zero values in more than just the k neighbors
        self.assertTrue(original_P.nnz >= n_samples * k_neighbors)

        # Check that lowering the perplexity properly changes affinity matrix
        perplexity = 10  # lower the perplexity value
        k_neighbors = perplexity * 3

        aff.set_perplexity(perplexity)
        self.assertEqual(aff.perplexity, perplexity)
        self.assertEqual(aff.effective_perplexity_, perplexity)

        reduced_P = aff.P.copy()
        self.assertTrue(reduced_P.nnz >= n_samples * k_neighbors)
        self.assertTrue(reduced_P.nnz < original_P.nnz,
                        "Lower perplexities should consider less neighbors, "
                        "resulting in a sparser affinity matrix")
                        
        # Check that increasing the perplexity works (with a warning)
        # Larger than the original perplexity, but still less than the number of
        # nearest neighbors
        perplexity = 40
        with self.assertLogs(affinity.log, level="WARNING"):
            aff.set_perplexity(perplexity)
            self.assertEqual(aff.perplexity, perplexity)
            self.assertEqual(aff.effective_perplexity_, perplexity)

        # Raising the perplexity above the number of neighbors in the kNN graph
        # would need to recompute the nearest neighbors, so it should raise an error
        with self.assertRaises(RuntimeError):
            aff.set_perplexity(70)  # more than the number of nearest neighbors

    def test_set_perplexity_respects_symmetrization(self):
        # Apply symmetrization
        symmetrize = True
        aff = PerplexityBasedNN(self.x, perplexity=30, symmetrize=symmetrize)
        self.assertAlmostEqual(np.sum(aff.P - aff.P.T), 0, delta=1e-16)

        aff.set_perplexity(20)
        self.assertAlmostEqual(np.sum(aff.P - aff.P.T), 0, delta=1e-16)

        # Skip symmetrization
        symmetrize = False
        aff = PerplexityBasedNN(self.x, perplexity=30, symmetrize=symmetrize)
        self.assertAlmostEqual(np.sum(aff.P - aff.P.T), 0, delta=1e-16)

        aff.set_perplexity(20)
        self.assertAlmostEqual(np.sum(aff.P - aff.P.T), 0, delta=1e-16)


class TestMultiscale(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x = np.random.normal(100, 50, (91, 4))
        cls.x2 = np.random.normal(100, 50, (30, 4))

    def test_handles_too_large_perplexities(self):
        # x has 91 samples, this means that the max perplexity that we allow is
        # (91 - 1) / 3 = 30. -1 because we don't consider ith point. Anything
        # above that should be ignored or corrected

        ms = Multiscale(self.x, perplexities=[20])
        np.testing.assert_array_equal(ms.perplexities, [20])
        np.testing.assert_array_equal(
            ms.effective_perplexities_, [20],
            "Incorrectly changed perplexity that was within a valid range",
        )

        ms = Multiscale(self.x, perplexities=[20, 40])
        np.testing.assert_array_equal(ms.perplexities, [20, 40])
        np.testing.assert_array_equal(
            ms.effective_perplexities_, [20, 30],
            "Did not lower large perplexity."
        )

        ms = Multiscale(self.x, perplexities=[20, 40, 60])
        np.testing.assert_array_equal(ms.perplexities, [20, 40, 60])
        np.testing.assert_array_equal(
            ms.effective_perplexities_, [20, 30],
            "Did not drop large perplexities when more than one was too large."
        )

        ms = Multiscale(self.x, perplexities=[20, 30, 40, 60])
        np.testing.assert_array_equal(ms.perplexities, [20, 30, 40, 60])
        np.testing.assert_array_equal(
            ms.effective_perplexities_, [20, 30],
            "Did not drop duplicate corrected perplexity."
        )

    def test_handles_changing_perplexities(self):
        perplexities = [15, 25]
        k_neighbors = perplexities[-1] * 3

        ms = Multiscale(self.x, perplexities=perplexities)
        np.testing.assert_equal(ms.perplexities, perplexities)
        np.testing.assert_equal(ms.effective_perplexities_, perplexities)

        # Check that the initial `P` matrix is allright
        n_samples = self.x.shape[0]
        original_P = ms.P.copy()
        # Can't check for equality because the matrix is symmetrized therefore
        # each point may have non-zero values in more than just the k neighbors
        self.assertTrue(original_P.nnz >= n_samples * k_neighbors)

        # Check that lowering the perplexity properly changes affinity matrix
        new_perplexities = [10, 20]
        k_neighbors = new_perplexities[-1] * 3
        ms.set_perplexities(new_perplexities)
        np.testing.assert_equal(ms.perplexities, new_perplexities)
        np.testing.assert_equal(ms.effective_perplexities_, new_perplexities)

        reduced_P = ms.P.copy()
        self.assertTrue(reduced_P.nnz >= n_samples * k_neighbors)
        self.assertTrue(reduced_P.nnz < original_P.nnz,
                        "Lower perplexities should consider less neighbors, "
                        "resulting in a sparser affinity matrix")

        # Raising the perplexity above the initial value would need to recompute
        # the nearest neighbors, so it should raise an error
        with self.assertRaises(RuntimeError):
            ms.set_perplexities([20, 30])

    def test_set_perplexities_respects_symmetrization(self):
        # Apply symmetrization
        symmetrize = True
        aff = Multiscale(self.x, perplexities=[10, 20], symmetrize=symmetrize)
        self.assertAlmostEqual(np.sum(aff.P - aff.P.T), 0, delta=1e-16)

        aff.set_perplexities([5, 10])
        self.assertAlmostEqual(np.sum(aff.P - aff.P.T), 0, delta=1e-16)

        # Skip symmetrization
        symmetrize = False
        aff = Multiscale(self.x, perplexities=[10, 20], symmetrize=symmetrize)
        self.assertAlmostEqual(np.sum(aff.P - aff.P.T), 0, delta=1e-16)

        aff.set_perplexities([5, 10])
        self.assertAlmostEqual(np.sum(aff.P - aff.P.T), 0, delta=1e-16)

    def test_single_perplexity_produces_same_result_as_perplexity_based_nn(self):
        for perplexity in [5, 30, 50, 100]:
            ms1 = MultiscaleMixture(self.x, perplexities=[perplexity])
            ms2 = Multiscale(self.x, perplexities=[perplexity])
            perp = PerplexityBasedNN(self.x, perplexity=perplexity)

            # 1e-14 instead of 1e-16: It looks like a machine precision issue
            # There are no noticeable differences between the matrices
            self.assertAlmostEqual(
                np.sum(ms1.P - perp.P), 0, delta=1e-14, msg=f"perplexity={perplexity}"
            )
            self.assertAlmostEqual(
                np.sum(ms2.P - perp.P), 0, delta=1e-14, msg=f"perplexity={perplexity}"
            )

    def test_single_perplexity_produces_same_result_as_perplexity_based_nn_to_new(self):
        for perplexity in [5, 30, 50, 100]:
            ms1 = MultiscaleMixture(self.x, perplexities=[perplexity])
            ms2 = Multiscale(self.x, perplexities=[perplexity])
            perp = PerplexityBasedNN(self.x, perplexity=perplexity)

            ms1_new = ms1.to_new(self.x2)
            ms2_new = ms2.to_new(self.x2)
            perp_new = perp.to_new(self.x2)

            # 1e-14 instead of 1e-16: It looks like a machine precision issue
            # There are no noticeable differences between the matrices
            self.assertAlmostEqual(
                np.sum(ms1_new - perp_new), 0, delta=1e-14, msg=f"perplexity={perplexity}"
            )
            self.assertAlmostEqual(
                np.sum(ms2_new - perp_new), 0, delta=1e-14, msg=f"perplexity={perplexity}"
            )

    def test_accepts_single_perplexity_value(self):
        perplexity = 30

        ms = MultiscaleMixture(self.x, perplexities=perplexity)
        self.assertEqual(ms.perplexities, 30)
        self.assertEqual(ms.effective_perplexities_, [30])

        ms = MultiscaleMixture(self.x, perplexities=perplexity)
        self.assertEqual(ms.perplexities, 30)
        self.assertEqual(ms.effective_perplexities_, [30])


class TestUniform(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x = np.random.normal(100, 50, (91, 4))
        cls.y = np.random.normal(100, 50, (31, 4))

    def test_all_unsymmetrized_values_the_same(self):
        aff = Uniform(self.x, k_neighbors=10, symmetrize=False)
        values = aff.P.data
        np.testing.assert_allclose(values, values[0])

    def test_to_new_all_equal(self):
        aff = Uniform(self.x, k_neighbors=10, symmetrize=False)
        new_p = aff.to_new(self.y)

        values = new_p.data
        np.testing.assert_allclose(values, values[0])

        new_p = new_p.toarray()
        np.testing.assert_allclose(new_p.sum(axis=1), np.ones(self.y.shape[0]))

    def test_symmetrize_mean_has_two_distinct_values(self):
        aff = Uniform(self.x, k_neighbors=10, symmetrize="mean")
        values = aff.P.data
        self.assertEqual(len(np.unique(values)), 2)

    def test_symmetrize_max_has_one_distinct_value(self):
        aff = Uniform(self.x, k_neighbors=10, symmetrize="max")
        values = aff.P.data
        self.assertEqual(len(np.unique(values)), 1)

    def test_symmetrize_with_invalid_parameter(self):
        with self.assertRaises(ValueError):
            Uniform(self.x, k_neighbors=10, symmetrize="invalid")


class TestAffinityMatrixCorrectness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.iris = datasets.load_iris().data

    def test_that_regular_matrix_sums_to_one(self):
        for method_name, cls in AFFINITY_CLASSES:
            aff: affinity.Affinities = cls(self.iris)
            self.assertAlmostEqual(np.sum(aff.P), 1, msg=method_name)

    def test_that_to_new_transform_matrix_treats_each_datapoint_separately(self):
        x_train, x_test = train_test_split(self.iris, test_size=0.33, random_state=42)

        for method_name, cls in AFFINITY_CLASSES:
            aff: affinity.Affinities = cls(x_train)
            P = aff.to_new(x_test)
            np.testing.assert_allclose(
                np.asarray(np.sum(P, axis=1)).ravel(),
                np.ones(len(x_test)),
                err_msg=method_name,
            )

    def test_handles_precomputed_distance_matrices(self):
        x = np.random.normal(0, 1, (200, 5))
        d = squareform(pdist(x))

        for method_name, cls in AFFINITY_CLASSES:
            aff = cls(d, metric="precomputed")
            self.assertIsInstance(
                aff.knn_index, nearest_neighbors.PrecomputedDistanceMatrix, msg=method_name
            )

    def test_affinity_matrix_matches_precomputed_distance_affinity_matrix_random(self):
        x = np.random.normal(0, 1, (200, 5))
        d = squareform(pdist(x))

        for method_name, cls in AFFINITY_CLASSES:
            aff1 = cls(d, metric="precomputed")
            aff2 = cls(x, metric="euclidean")

            np.testing.assert_almost_equal(
                aff1.P.toarray(), aff2.P.toarray(), err_msg=method_name
            )

    def test_affinity_matrix_matches_precomputed_distance_affinity_matrix_iris(self):
        x = self.iris + np.random.normal(0, 1e-3, self.iris.shape)  # iris contains duplicate rows
        d = squareform(pdist(x))

        for method_name, cls in AFFINITY_CLASSES:
            aff1 = cls(d, metric="precomputed")
            aff2 = cls(x, metric="euclidean")

            np.testing.assert_almost_equal(
                aff1.P.toarray(), aff2.P.toarray(), err_msg=method_name
            )


class TestAffinityAcceptsKnnIndexAsParameter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.iris = datasets.load_iris().data
        cls.iris += np.random.normal(0, 1e-3, cls.iris.shape)

    def test_fails_if_neither_data_nor_index_specified(self):
        for method_name, cls in AFFINITY_CLASSES:
            with self.assertRaises(ValueError, msg=method_name):
                cls(data=None, knn_index=None)

    def test_fails_if_both_data_and_index_specified(self):
        knn_index = nearest_neighbors.Sklearn(self.iris, k=30)
        for method_name, cls in AFFINITY_CLASSES:
            with self.assertRaises(ValueError, msg=method_name):
                cls(data=self.iris, knn_index=knn_index)

    def test_accepts_knn_index(self):
        knn_index = nearest_neighbors.Sklearn(self.iris, k=30)
        for method_name, cls in AFFINITY_CLASSES:
            aff = cls(knn_index=knn_index)
            self.assertIs(aff.knn_index, knn_index, msg=method_name)
            self.assertEqual(aff.n_samples, self.iris.shape[0])

    def test_to_new(self):
        knn_index = nearest_neighbors.Sklearn(self.iris, k=30)
        for method_name, cls in AFFINITY_CLASSES:
            aff = cls(knn_index=knn_index)
            aff.to_new(self.iris)


class TestPrecomputedAffinity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.iris = datasets.load_iris().data

    def test_precomputed_affinity_matches(self):
        aff1 = PerplexityBasedNN(self.iris)
        aff2 = affinity.PrecomputedAffinities(aff1.P)
        np.testing.assert_allclose(
            aff1.P.toarray(),
            aff2.P.toarray(),
            err_msg="Precomputed affinities are not passed in correctly.",
        )

    def test_precomputed_affinity_normalization(self):
        aff1 = PerplexityBasedNN(self.iris)
        aff2 = affinity.PrecomputedAffinities(aff1.P * 2)
        np.testing.assert_allclose(
            aff1.P.toarray(),
            aff2.P.toarray(),
            err_msg="Precomputed affinities are not normalized to sum to 1.",
        )


class TestGenericAffinity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x = datasets.load_iris().data

    def test_knn_kwargs(self):
        import sklearn

        for method_name, cls in AFFINITY_CLASSES:
            with patch(
                "sklearn.neighbors.NearestNeighbors",
                wraps=sklearn.neighbors.NearestNeighbors,
            ) as mock:
                params = dict(algorithm="kd_tree", leaf_size=10)
                cls(self.x, method="exact", knn_kwargs=params)
                check_mock_called_with_kwargs(mock, params)

    @unittest.skipUnless(is_package_installed("pynndescent"), "pynndescent not installed")
    def test_call_with_custom_metric(self):
        # Let's define some custom metric
        def metric(x, y):
            return np.sum(x + y ** 2)

        for method_name, cls in AFFINITY_CLASSES:
            for neighbor_method in ["exact", "approx"]:
                cls(self.x[:50], metric=metric, method=neighbor_method)

    def test_call_with_custom_metric_with_unsupported_knn(self):
        # Let's define some custom metric
        def metric(x, y):
            return np.sum(x + y ** 2)

        for method_name, cls in AFFINITY_CLASSES:
            for neighbor_method in ["annoy", "hnsw"]:
                with self.assertRaisesRegex(ValueError, "does not support callable metrics"):
                    cls(self.x[:50], metric=metric, method=neighbor_method)
