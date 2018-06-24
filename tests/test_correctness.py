import unittest

import numpy as np

from tsne.callbacks import VerifyExaggerationError
from tsne.tsne import TSNE

np.random.seed(42)


class TestTSNECorrectness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tsne = TSNE()
        cls.x = np.random.randn(100, 4)

    def test_error_exaggeration_correction(self):
        embedding = self.tsne.prepare_initial(self.x)

        # The callback raises if the KL divergence does not match the true one
        embedding.optimize(
            50, exaggeration=5, callbacks=[VerifyExaggerationError(embedding)],
            callbacks_every_iters=1, inplace=True,
        )
