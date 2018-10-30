import unittest

import numpy as np

from fastTSNE.affinity import Multiscale, PerplexityBasedNN


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
                        'Lower perplexities should consider less neighbors, '
                        'resulting in a sparser affinity matrix')

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
            'Incorrectly changed perplexity that was within a valid range',
        )

        ms = Multiscale(self.x, perplexities=[20, 40])
        np.testing.assert_array_equal(
            ms.perplexities, [20, 30],
            'Did not lower large perplexity.'
        )

        ms = Multiscale(self.x, perplexities=[20, 40, 60])
        np.testing.assert_array_equal(
            ms.perplexities, [20, 30],
            'Did not drop large perplexities when more than one was too large.'
        )

        ms = Multiscale(self.x, perplexities=[20, 30, 40, 60])
        np.testing.assert_array_equal(
            ms.perplexities, [20, 30],
            'Did not drop duplicate corrected perplexity.'
        )
