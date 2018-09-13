import unittest

import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

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

    def test_iris(self):
        iris = datasets.load_iris()
        x, y = iris['data'], iris['target']

        # Evaluate tSNE optimization using a KNN classifier
        knn = KNeighborsClassifier(n_neighbors=5)
        tsne = TSNE(perplexity=30, initialization='random')

        # Prepare a random initialization
        embedding = tsne.prepare_initial(x)

        # KNN should do poorly on a random initialization
        knn.fit(embedding, y)
        predictions = knn.predict(embedding)
        self.assertTrue(accuracy_score(predictions, y) < .5)

        # Optimize the embedding for a small number of steps so tests run fast
        embedding.optimize(50, inplace=True)

        # Similar points should be grouped together, therefore KNN should do well
        knn.fit(embedding, y)
        predictions = knn.predict(embedding)
        self.assertTrue(accuracy_score(predictions, y) > .95)
