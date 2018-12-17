import unittest
from functools import partial

import openTSNE
import openTSNE.affinity
import openTSNE.initialization
import numpy as np
from openTSNE.callbacks import VerifyExaggerationError
from openTSNE.tsne import TSNEEmbedding
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TSNE = partial(openTSNE.TSNE, neighbors="exact", negative_gradient_method="bh")


class TestTSNECorrectness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tsne = TSNE(early_exaggeration_iter=20, n_iter=100)
        # Set up two modalities, if we want to viually inspect test results
        random_state = np.random.RandomState(0)
        cls.x = np.vstack(
            (random_state.normal(+1, 1, (100, 4)), random_state.normal(-1, 1, (100, 4)))
        )
        cls.x_test = random_state.normal(0, 1, (25, 4))
        cls.iris = datasets.load_iris()

    def test_basic_flow(self):
        """Verify that the basic flow does not crash."""
        embedding = self.tsne.fit(self.x)
        self.assertFalse(np.any(np.isnan(embedding)))

        partial_embedding = embedding.transform(self.x_test, n_iter=20)
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
            50,
            exaggeration=5,
            callbacks=[VerifyExaggerationError(embedding)],
            callbacks_every_iters=1,
            inplace=True,
        )

    def test_iris(self):
        x, y = self.iris.data, self.iris.target

        # Evaluate t-SNE optimization using a KNN classifier
        knn = KNeighborsClassifier(n_neighbors=10)
        tsne = TSNE(perplexity=30, initialization="random", random_state=0)

        # Prepare a random initialization
        embedding = tsne.prepare_initial(x)

        # KNN should do poorly on a random initialization
        knn.fit(embedding, y)
        predictions = knn.predict(embedding)
        self.assertLess(accuracy_score(predictions, y), 0.5)

        # Optimize the embedding for a small number of steps so tests run fast
        embedding.optimize(50, inplace=True)

        # Similar points should be grouped together, therefore KNN should do well
        knn.fit(embedding, y)
        predictions = knn.predict(embedding)
        self.assertGreater(accuracy_score(predictions, y), 0.95)

    def test_iris_bh_transform_equivalency_with_one_by_one(self):
        """Compare one by one embedding vs all at once using BH gradients."""
        x_train, x_test = train_test_split(
            self.iris.data, test_size=0.33, random_state=42
        )

        # Set up the initial embedding
        embedding = openTSNE.TSNE(
            early_exaggeration_iter=0,
            n_iter=50,
            neighbors="exact",
            negative_gradient_method="bh",
        ).fit(x_train)

        params = dict(n_iter=100, perplexity=5)
        # Build up an embedding by adding points one by one
        new_embedding_1 = np.vstack(
            [embedding.transform(np.atleast_2d(point), **params) for point in x_test]
        )
        # Add new points altogether
        new_embedding_2 = embedding.transform(x_test, **params)

        # Verify that the embedding has actually been optimized
        self.assertRaises(
            AssertionError,
            np.testing.assert_almost_equal,
            embedding.prepare_partial(x_test, perplexity=params["perplexity"]),
            new_embedding_1,
        )
        # Check that both methods produced the same embedding
        np.testing.assert_almost_equal(new_embedding_1, new_embedding_2)

    def test_iris_fft_transform_equivalency_with_one_by_one(self):
        """Compare one by one embedding vs all at once using FFT gradients.

        Note that this won't return the exact same embedding both times because
        the grid placed over the embedding will differ when placing points one
        by one vs. when placing them at once. The min/max coords will differ,
        thus changing the overall approximation. They should be quite similar
        though.

        """
        x_train, x_test = train_test_split(
            self.iris.data, test_size=0.33, random_state=42
        )

        # Set up the initial embedding
        embedding = openTSNE.TSNE(
            early_exaggeration_iter=0,
            n_iter=50,
            neighbors="exact",
            negative_gradient_method="fft",
        ).fit(x_train)

        params = dict(n_iter=100, perplexity=5)
        # Build up an embedding by adding points one by one
        new_embedding_1 = np.vstack(
            [embedding.transform(np.atleast_2d(point), **params) for point in x_test]
        )
        # Add new points altogether
        new_embedding_2 = embedding.transform(x_test, **params)

        # Verify that the embedding has actually been optimized
        self.assertRaises(
            AssertionError,
            np.testing.assert_almost_equal,
            embedding.prepare_partial(x_test, perplexity=params["perplexity"]),
            new_embedding_1,
        )
        # Check that both methods produced the same embedding
        np.testing.assert_almost_equal(new_embedding_1, new_embedding_2, decimal=1)

    def test_iris_bh_transform_correctness(self):
        x_train, x_test, y_train, y_test = train_test_split(
            self.iris.data, self.iris.target, test_size=0.33, random_state=42
        )

        # Set up the initial embedding
        embedding = openTSNE.TSNE(
            neighbors="exact",
            negative_gradient_method="bh",
            early_exaggeration_iter=0,
            n_iter=50,
        ).fit(x_train)

        # Evaluate t-SNE optimization using a KNN classifier
        knn = KNeighborsClassifier(n_neighbors=10)
        knn.fit(embedding, y_train)

        new_embedding = embedding.transform(x_test, n_iter=100, perplexity=100)
        predictions = knn.predict(new_embedding)
        self.assertGreater(accuracy_score(predictions, y_test), 0.99)

    def test_iris_fft_transform_correctness(self):
        x_train, x_test, y_train, y_test = train_test_split(
            self.iris.data, self.iris.target, test_size=0.33, random_state=42
        )

        # Set up the initial embedding
        embedding = openTSNE.TSNE(
            neighbors="exact",
            negative_gradient_method="fft",
            early_exaggeration_iter=0,
            n_iter=50,
        ).fit(x_train)

        # Evaluate t-SNE optimization using a KNN classifier
        knn = KNeighborsClassifier(n_neighbors=10)
        knn.fit(embedding, y_train)

        new_embedding = embedding.transform(x_test, n_iter=100, perplexity=100)
        predictions = knn.predict(new_embedding)
        self.assertGreater(accuracy_score(predictions, y_test), 0.99)

    def test_bh_transform_with_point_subsets_using_perplexity_nn(self):
        x_train, x_test = train_test_split(
            self.iris.data, test_size=0.33, random_state=42
        )

        # Set up the initial embedding
        init = openTSNE.initialization.pca(x_train)
        affinity = openTSNE.affinity.PerplexityBasedNN(x_train, method="exact")
        embedding = openTSNE.TSNEEmbedding(
            init, affinity, negative_gradient_method="bh", random_state=42
        )
        embedding.optimize(n_iter=50, inplace=True)

        # The test set contains 50 samples, so let's verify on half of those
        transform_params = dict(n_iter=0)
        new_embedding_1 = embedding.transform(x_test, **transform_params)[:25]
        new_embedding_2 = embedding.transform(x_test[:25], **transform_params)

        np.testing.assert_equal(new_embedding_1, new_embedding_2)

    def test_fft_transform_with_point_subsets_using_perplexity_nn(self):
        x_train, x_test = train_test_split(
            self.iris.data, test_size=0.33, random_state=42
        )

        # Set up the initial embedding
        init = openTSNE.initialization.pca(x_train)
        affinity = openTSNE.affinity.PerplexityBasedNN(x_train, method="exact")
        embedding = openTSNE.TSNEEmbedding(
            init, affinity, negative_gradient_method="fft", random_state=42
        )
        embedding.optimize(n_iter=50, inplace=True)

        # The test set contains 50 samples, so let's verify on half of those
        transform_params = dict(n_iter=0)
        new_embedding_1 = embedding.transform(x_test, **transform_params)[:25]
        new_embedding_2 = embedding.transform(x_test[:25], **transform_params)

        np.testing.assert_equal(new_embedding_1, new_embedding_2)
