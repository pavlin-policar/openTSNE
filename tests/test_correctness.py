import unittest
from functools import partial

import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import fastTSNE
import fastTSNE.affinity
import fastTSNE.initialization
from fastTSNE import tsne
from fastTSNE.callbacks import VerifyExaggerationError
from fastTSNE.tsne import TSNEEmbedding

TSNE = partial(tsne.TSNE, neighbors="exact", negative_gradient_method="bh")


class TestTSNECorrectness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tsne = TSNE(early_exaggeration_iter=20, n_iter=100)
        # Set up two modalities, if we want to viually inspect test results
        random_state = np.random.RandomState(0)
        cls.x = np.vstack((
            random_state.normal(+1, 1, (100, 4)),
            random_state.normal(-1, 1, (100, 4)),
        ))
        cls.x_test = random_state.normal(0, 1, (25, 4))
        cls.iris = datasets.load_iris()

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
        x, y = iris["data"], iris["target"]

        # Evaluate tSNE optimization using a KNN classifier
        knn = KNeighborsClassifier(n_neighbors=10)
        tsne = TSNE(perplexity=30, initialization="random", random_state=0)

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

    def test_bh_transform_with_point_subsets_using_perplexity_nn(self):
        x_train, x_test = train_test_split(self.iris.data, test_size=0.33, random_state=42)

        # Set up the initial embedding
        init = fastTSNE.initialization.pca(x_train)
        affinity = fastTSNE.affinity.PerplexityBasedNN(x_train, method="exact")
        embedding = fastTSNE.TSNEEmbedding(
            init, affinity, negative_gradient_method="bh", random_state=42
        )
        embedding.optimize(n_iter=50, inplace=True)

        # The test set contains 50 samples, so let's verify on half of those
        transform_params = dict(early_exaggeration_iter=0, n_iter=0)
        new_embedding_1 = embedding.transform(x_test, **transform_params)[:25]
        new_embedding_2 = embedding.transform(x_test[:25], **transform_params)

        np.testing.assert_equal(new_embedding_1, new_embedding_2)

    def test_fft_transform_with_point_subsets_using_perplexity_nn(self):
        x_train, x_test = train_test_split(self.iris.data, test_size=0.33, random_state=42)

        # Set up the initial embedding
        init = fastTSNE.initialization.pca(x_train)
        affinity = fastTSNE.affinity.PerplexityBasedNN(x_train, method="exact")
        embedding = fastTSNE.TSNEEmbedding(
            init, affinity, negative_gradient_method="fft", random_state=42
        )
        embedding.optimize(n_iter=50, inplace=True)

        # The test set contains 50 samples, so let's verify on half of those
        transform_params = dict(early_exaggeration_iter=0, n_iter=0)
        new_embedding_1 = embedding.transform(x_test, **transform_params)[:25]
        new_embedding_2 = embedding.transform(x_test[:25], **transform_params)

        np.testing.assert_equal(new_embedding_1, new_embedding_2)
