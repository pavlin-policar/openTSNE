import unittest
from functools import partial

import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

import openTSNE
from openTSNE import affinity, initialization, nearest_neighbors, TSNEEmbedding

Multiscale = partial(affinity.Multiscale, method="exact")
MultiscaleMixture = partial(affinity.MultiscaleMixture, method="exact")
PerplexityBasedNN = partial(affinity.PerplexityBasedNN, method="exact")
FixedSigmaNN = partial(affinity.FixedSigmaNN, method="exact")
Uniform = partial(affinity.Uniform, method="exact")

tsne_params = dict(
    early_exaggeration_iter=25,
    n_iter=50,
    neighbors="exact",
    negative_gradient_method="bh",
)
TSNE = partial(openTSNE.TSNE, **tsne_params)


class TestUsage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.iris = datasets.load_iris()
        cls.x = cls.iris.data + np.random.normal(0, 1e-3, cls.iris.data.shape)
        cls.y = cls.iris.target

    def eval_embedding(self, embedding, y, method_name=None):
        knn = KNeighborsClassifier(n_neighbors=10)
        knn.fit(embedding, y)
        predictions = knn.predict(embedding)
        self.assertGreater(accuracy_score(predictions, y), 0.94, msg=method_name)


class TestUsageSimple(TestUsage):
    def test_simple(self):
        embedding = TSNE().fit(self.x)
        self.eval_embedding(embedding, self.y)
        new_embedding = embedding.transform(self.x)
        self.eval_embedding(new_embedding, self.y, "transform")

    def test_simple_multiscale(self):
        embedding = TSNE(perplexity=[10, 30]).fit(self.x)
        self.eval_embedding(embedding, self.y)
        new_embedding = embedding.transform(self.x, perplexity=[5, 10])
        self.eval_embedding(new_embedding, self.y, "transform")

    def test_with_precomputed_distances(self):
        d = squareform(pdist(self.x))
        embedding = TSNE(metric="precomputed").fit(d)
        self.eval_embedding(embedding, self.y)

        d_new = cdist(self.x[:20], self.x)
        new_embedding = embedding.transform(d_new)
        self.eval_embedding(new_embedding, self.y[:20], "transform")


class TestUsageLowestLevel(TestUsage):
    def test_1(self):
        init = initialization.pca(self.x)
        aff = affinity.PerplexityBasedNN(self.x, perplexity=30)
        embedding = openTSNE.TSNEEmbedding(init, aff)
        embedding.optimize(25, exaggeration=12, momentum=0.8, inplace=True)
        embedding.optimize(50, exaggeration=1, momentum=0.8, inplace=True)
        self.eval_embedding(embedding, self.y)
        new_embedding = embedding.transform(self.x)
        self.eval_embedding(new_embedding, self.y, "transform")

    def test_2(self):
        init = initialization.pca(self.x)
        aff = affinity.MultiscaleMixture(self.x, perplexities=[5, 30])
        embedding = openTSNE.TSNEEmbedding(init, aff)
        embedding.optimize(25, exaggeration=12, momentum=0.8, inplace=True)
        embedding.optimize(50, exaggeration=1, momentum=0.8, inplace=True)
        self.eval_embedding(embedding, self.y)
        new_embedding = embedding.transform(self.x)
        self.eval_embedding(new_embedding, self.y, "transform")


class TestUsageWithCustomAffinity(TestUsage):
    def test_affinity(self):
        init = initialization.random(self.x, random_state=0)

        for aff in [
            affinity.PerplexityBasedNN(self.x, perplexity=30),
            affinity.Uniform(self.x, k_neighbors=30),
            affinity.FixedSigmaNN(self.x, sigma=1, k=30),
            affinity.Multiscale(self.x, perplexities=[10, 30]),
            affinity.MultiscaleMixture(self.x, perplexities=[10, 30]),
        ]:
            # Without initilization
            embedding = TSNE().fit(affinities=aff)
            self.eval_embedding(embedding, self.y, aff.__class__.__name__)
            new_embedding = embedding.prepare_partial(self.x)
            new_embedding.optimize(10, learning_rate=0.1, inplace=True)
            self.eval_embedding(new_embedding, self.y, f"transform::{aff.__class__.__name__}")

            # With initilization
            embedding = TSNE().fit(affinities=aff, initialization=init)
            self.eval_embedding(embedding, self.y, aff.__class__.__name__)
            new_embedding = embedding.prepare_partial(self.x)
            new_embedding.optimize(10, learning_rate=0.1, inplace=True)
            self.eval_embedding(new_embedding, self.y, f"transform::{aff.__class__.__name__}")

    def test_precomputed_affinity(self):
        # Setup precomputed affinity
        aff = affinity.Uniform(self.x).P
        aff *= 10  # rescale
        precomputed_affinity = affinity.PrecomputedAffinities(aff, normalize=True)

        embedding = TSNE().fit(affinities=precomputed_affinity, initialization="spectral")
        self.eval_embedding(embedding, self.y, aff.__class__.__name__)


class TestUsageWithCustomAffinityAndCustomNeighbors(TestUsage):
    def test_affinity_with_queryable_knn_index(self):
        knn_index = nearest_neighbors.Sklearn(self.x, k=30)
        init = initialization.random(self.x, random_state=0)

        for aff in [
            affinity.PerplexityBasedNN(knn_index=knn_index, perplexity=30),
            affinity.Uniform(knn_index=knn_index, k_neighbors=30),
            affinity.FixedSigmaNN(knn_index=knn_index, sigma=1),
            affinity.Multiscale(knn_index=knn_index, perplexities=[10, 20]),
            affinity.MultiscaleMixture(knn_index=knn_index, perplexities=[10, 20]),
        ]:
            # Without initilization
            embedding = TSNE().fit(affinities=aff)
            self.eval_embedding(embedding, self.y, aff.__class__.__name__)
            new_embedding = embedding.prepare_partial(self.x)
            new_embedding.optimize(50, learning_rate=1, inplace=True)
            self.eval_embedding(new_embedding, self.y, f"transform::{aff.__class__.__name__}")

            # With initilization
            embedding = TSNE().fit(affinities=aff, initialization=init)
            self.eval_embedding(embedding, self.y, aff.__class__.__name__)
            new_embedding = embedding.prepare_partial(self.x)
            new_embedding.optimize(50, learning_rate=1, inplace=True)
            self.eval_embedding(new_embedding, self.y, f"transform::{aff.__class__.__name__}")

    def test_affinity_with_precomputed_distances(self):
        d = squareform(pdist(self.x))
        knn_index = nearest_neighbors.PrecomputedDistanceMatrix(d, k=30)
        init = initialization.random(self.x, random_state=0)

        for aff in [
            affinity.PerplexityBasedNN(knn_index=knn_index, perplexity=30),
            affinity.Uniform(knn_index=knn_index, k_neighbors=30),
            affinity.FixedSigmaNN(knn_index=knn_index, sigma=1),
            affinity.Multiscale(knn_index=knn_index, perplexities=[10, 20]),
            affinity.MultiscaleMixture(knn_index=knn_index, perplexities=[10, 20]),
        ]:
            # Without initilization
            embedding = TSNE().fit(affinities=aff)
            self.eval_embedding(embedding, self.y, aff.__class__.__name__)

            # With initialization
            embedding = TSNE().fit(affinities=aff, initialization=init)
            self.eval_embedding(embedding, self.y, aff.__class__.__name__)

    def test_affinity_with_precomputed_neighbors(self):
        nn = NearestNeighbors(n_neighbors=30)
        nn.fit(self.x)
        distances, neighbors = nn.kneighbors(n_neighbors=30)

        knn_index = nearest_neighbors.PrecomputedNeighbors(neighbors, distances)
        init = initialization.random(self.x, random_state=0)

        for aff in [
            affinity.PerplexityBasedNN(knn_index=knn_index, perplexity=30),
            affinity.Uniform(knn_index=knn_index, k_neighbors=30),
            affinity.FixedSigmaNN(knn_index=knn_index, sigma=1),
            affinity.Multiscale(knn_index=knn_index, perplexities=[10, 20]),
            affinity.MultiscaleMixture(knn_index=knn_index, perplexities=[10, 20]),
        ]:
            # Without initilization
            embedding = TSNE().fit(affinities=aff)
            self.eval_embedding(embedding, self.y, aff.__class__.__name__)

            # With initilization
            embedding = TSNE().fit(affinities=aff, initialization=init)
            self.eval_embedding(embedding, self.y, aff.__class__.__name__)


class TestUsageExplicitOptimizeCalls(TestUsage):
    def test_explicit_optimize_calls(self):
        embedding1 = TSNE(random_state=42).fit(self.x)
        
        A = affinity.PerplexityBasedNN(self.x)
        I = initialization.pca(self.x, random_state=42)
        embedding2 = TSNEEmbedding(I, A)
        embedding2 = embedding2.optimize(n_iter=25, exaggeration=12)
        embedding2 = embedding2.optimize(n_iter=50)
        
        np.testing.assert_array_equal(
            embedding1,
            embedding2,
            "Calling optimize twice with default parameters produced a different " \
            "result compared to the default fit() call"
        )

    def test_multiple_optimize_calls(self):
        A = affinity.PerplexityBasedNN(self.x)
        I = initialization.pca(self.x)
        embedding1 = TSNEEmbedding(I, A)
        embedding1.optimize(n_iter=50, inplace=True)
        embedding2 = TSNEEmbedding(I, A)
        for i in range(50):
            embedding2.optimize(n_iter=1, inplace=True)
        
        np.testing.assert_array_equal(
            embedding1,
            embedding2,
            "Calling optimize 50 times with n_iter=1 produced a different " \
            "result compared to calling it once with n_iter=50"
        )
