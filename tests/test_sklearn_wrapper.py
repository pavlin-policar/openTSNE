import unittest
import numpy as np

from openTSNE.sklearn import TSNE


class TestTSNECorrectness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tsne = TSNE(
            early_exaggeration_iter=20,
            n_iter=100,
            neighbors="exact",
            negative_gradient_method="bh",
        )
        # Set up two modalities, if we want to viually inspect test results
        random_state = np.random.RandomState(0)
        cls.x = np.vstack(
            (random_state.normal(+1, 1, (100, 4)), random_state.normal(-1, 1, (100, 4)))
        )
        cls.x_test = random_state.normal(0, 1, (25, 4))

    def test_fit(self):
        retval = self.tsne.fit(self.x)
        self.assertIs(type(retval), TSNE)

    def test_fit_transform(self):
        retval = self.tsne.fit_transform(self.x)
        self.assertIs(type(retval), np.ndarray)

    def test_transform(self):
        self.tsne.fit(self.x)
        retval = self.tsne.transform(self.x_test)
        self.assertIs(type(retval), np.ndarray)
