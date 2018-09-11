import unittest

import numpy as np

from fastTSNE.callbacks import VerifyExaggerationError
from fastTSNE.tsne import TSNE, TSNEEmbedding

np.random.seed(42)


class TestTSNECorrectness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tsne = TSNE(early_exaggeration_iter=20, n_iter=100)
        # Set up two modalities, if we want to viually inspect test results
        cls.x = np.vstack((
            np.random.normal(+1, 1, (100, 4)),
            np.random.normal(-1, 1, (100, 4)),
        ))
        cls.x_test = np.random.normal(0, 1, (25, 4))

    def test_basic_flow(self):
        """Verify that the basic flow does not crash."""
        embedding = self.tsne.fit(self.x)
        self.assertFalse(np.any(np.isnan(embedding)))

        partial_embedding = embedding.transform(
            self.x_test, early_exaggeration_iter=20, n_iter=20)
        self.assertFalse(np.any(np.isnan(partial_embedding)))

    def test_advanced_flow(self):
        """Verify that the advanced flow does not crash."""
        embedding = self.tsne.prepare_initial(self.x)
        embedding = embedding.optimize(20, exaggeration=12)
        embedding = embedding.optimize(20)  # type: TSNEEmbedding
        self.assertFalse(np.any(np.isnan(embedding)))

        partial_embedding = embedding.prepare_partial(self.x_test)
        partial_embedding = partial_embedding.optimize(20, exaggeration=2)
        partial_embedding = partial_embedding.optimize(20)
        self.assertFalse(np.any(np.isnan(partial_embedding)))

    def test_error_exaggeration_correction(self):
        embedding = self.tsne.prepare_initial(self.x)

        # The callback raises if the KL divergence does not match the true one
        embedding.optimize(
            50, exaggeration=5, callbacks=[VerifyExaggerationError(embedding)],
            callbacks_every_iters=1, inplace=True,
        )
