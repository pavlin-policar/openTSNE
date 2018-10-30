import unittest

import numpy as np

from fastTSNE.affinity import Multiscale


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
