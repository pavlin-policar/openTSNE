import logging
import unittest
from functools import partial

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from openTSNE import affinity

affinity.log.setLevel(logging.ERROR)

Multiscale = partial(affinity.Multiscale, method="exact")
MultiscaleMixture = partial(affinity.MultiscaleMixture, method="exact")
PerplexityBasedNN = partial(affinity.PerplexityBasedNN, method="exact")
FixedSigmaNN = partial(affinity.FixedSigmaNN, method="exact")


class TestPerplexityBased(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x = np.random.normal(100, 50, (91, 4))

    def test_properly_reduces_large_perplexity(self):
        aff = PerplexityBasedNN(self.x, perplexity=40)
        self.assertEqual(aff.perplexity, 30)

    def test_handles_reducing_perplexity_value(self):
        perplexity = 20
        k_neighbors = perplexity * 3
        aff = PerplexityBasedNN(self.x, perplexity=perplexity)

        self.assertEqual(aff.perplexity, perplexity)

        # Check that the initial `P` matrix is allright
        n_samples = self.x.shape[0]
        original_P = aff.P.copy()
        # Can't check for equality because the matrix is symmetrized therefore
        # each point may have non-zero values in more than just the k neighbors
        self.assertTrue(original_P.nnz >= n_samples * k_neighbors)

        # Check that lowering the perplexity properly changes affinity matrix
        perplexity = 10
        k_neighbors = perplexity * 3

        aff.set_perplexity(perplexity)
        self.assertEqual(aff.perplexity, perplexity)
        reduced_P = aff.P.copy()
        self.assertTrue(reduced_P.nnz >= n_samples * k_neighbors)
        self.assertTrue(reduced_P.nnz < original_P.nnz,
                        "Lower perplexities should consider less neighbors, "
                        "resulting in a sparser affinity matrix")

        # Raising the perplexity above the initial value would need to recompute
        # the nearest neighbors, so it should raise an error
        with self.assertRaises(RuntimeError):
            aff.set_perplexity(30)


class TestMultiscale(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x = np.random.normal(100, 50, (91, 4))

    def test_handles_too_large_perplexities(self):
        # x has 91 samples, this means that the max perplexity that we allow is
        # (91 - 1) / 3 = 30. -1 because we don't consider ith point. Anything
        # above that should be ignored or corrected

        ms = Multiscale(self.x, perplexities=[20])
        np.testing.assert_array_equal(
            ms.perplexities, [20],
            "Incorrectly changed perplexity that was within a valid range",
        )

        ms = Multiscale(self.x, perplexities=[20, 40])
        np.testing.assert_array_equal(
            ms.perplexities, [20, 30],
            "Did not lower large perplexity."
        )

        ms = Multiscale(self.x, perplexities=[20, 40, 60])
        np.testing.assert_array_equal(
            ms.perplexities, [20, 30],
            "Did not drop large perplexities when more than one was too large."
        )

        ms = Multiscale(self.x, perplexities=[20, 30, 40, 60])
        np.testing.assert_array_equal(
            ms.perplexities, [20, 30],
            "Did not drop duplicate corrected perplexity."
        )

    def test_handles_changing_perplexities(self):
        perplexities = [15, 25]
        k_neighbors = perplexities[-1] * 3

        ms = Multiscale(self.x, perplexities=perplexities)
        np.testing.assert_equal(ms.perplexities, perplexities)

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

        reduced_P = ms.P.copy()
        self.assertTrue(reduced_P.nnz >= n_samples * k_neighbors)
        self.assertTrue(reduced_P.nnz < original_P.nnz,
                        "Lower perplexities should consider less neighbors, "
                        "resulting in a sparser affinity matrix")

        # Raising the perplexity above the initial value would need to recompute
        # the nearest neighbors, so it should raise an error
        with self.assertRaises(RuntimeError):
            ms.set_perplexities([20, 30])


class TestAffinityMatrixCorrectness(unittest.TestCase):
    affinity_classes = [
        ("PerplexityBasedNN", PerplexityBasedNN),
        ("FixedSigmaNN", partial(FixedSigmaNN, sigma=1)),
        ("MultiscaleMixture", partial(MultiscaleMixture, perplexities=[10, 20])),
        ("Multiscale", partial(Multiscale, perplexities=[10, 20])),
    ]

    @classmethod
    def setUpClass(cls):
        cls.iris = datasets.load_iris().data

    def test_that_regular_matrix_sums_to_one(self):
        for method_name, cls in self.affinity_classes:
            aff: affinity.Affinities = cls(self.iris)
            self.assertAlmostEqual(np.sum(aff.P), 1, msg=method_name)

    def test_that_to_new_transform_matrix_treats_each_datapoint_separately(self):
        x_train, x_test = train_test_split(self.iris, test_size=0.33, random_state=42)

        for method_name, cls in self.affinity_classes:
            aff: affinity.Affinities = cls(x_train)
            P = aff.to_new(x_test)
            np.testing.assert_allclose(
                np.asarray(np.sum(P, axis=1)).ravel(),
                np.ones(len(x_test)),
                err_msg=method_name,
            )

