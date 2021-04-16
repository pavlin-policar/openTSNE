import unittest
from functools import partial

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

import openTSNE
from openTSNE import affinity, initialization, nearest_neighbors

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

    def eval_embedding(self, embedding, method_name=None):
        knn = KNeighborsClassifier(n_neighbors=10)
        knn.fit(embedding, self.y)
        predictions = knn.predict(embedding)
        self.assertGreater(accuracy_score(predictions, self.y), 0.95, msg=method_name)


class TestUsageSimple(TestUsage):
    def test_simple(self):
        embedding = TSNE().fit(self.x)
        self.eval_embedding(embedding)
        new_embedding = embedding.transform(self.x)
        self.eval_embedding(new_embedding, "transform")

    def test_with_precomputed_distances(self):
        d = squareform(pdist(self.x))
        embedding = TSNE(metric="precomputed").fit(d)
        self.eval_embedding(embedding)

        # No transform, precomputed distances can't be queried


class TestUsageLowestLevel(TestUsage):
    def test_1(self):
        init = initialization.pca(self.x)
        aff = affinity.PerplexityBasedNN(self.x, perplexity=30)
        embedding = openTSNE.TSNEEmbedding(init, aff)
        embedding.optimize(25, exaggeration=12, momentum=0.5, inplace=True)
        embedding.optimize(50, exaggeration=1, momentum=0.8, inplace=True)
        self.eval_embedding(embedding)
        new_embedding = embedding.transform(self.x)
        self.eval_embedding(new_embedding, f"transform")


class TestUsageWithCustomAffinity(TestUsage):
    def test_affinity(self):
        init = initialization.random(self.x, random_state=0)

        for aff in [
            affinity.PerplexityBasedNN(self.x, perplexity=30),
            affinity.Uniform(self.x, k_neighbors=30),
            affinity.FixedSigmaNN(self.x, sigma=1),
            affinity.Multiscale(self.x, perplexities=[10, 20]),
            affinity.MultiscaleMixture(self.x, perplexities=[10, 20]),
        ]:
            # Without initilization
            embedding = TSNE().fit(affinities=aff)
            self.eval_embedding(embedding, aff.__class__.__name__)
            new_embedding = embedding.prepare_partial(self.x)
            new_embedding.optimize(10, learning_rate=0.1, inplace=True)
            self.eval_embedding(new_embedding, f"transform::{aff.__class__.__name__}")

            # With initilization
            embedding = TSNE().fit(affinities=aff, initialization=init)
            self.eval_embedding(embedding, aff.__class__.__name__)
            new_embedding = embedding.prepare_partial(self.x)
            new_embedding.optimize(10, learning_rate=0.1, inplace=True)
            self.eval_embedding(new_embedding, f"transform::{aff.__class__.__name__}")


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
            self.eval_embedding(embedding, aff.__class__.__name__)
            new_embedding = embedding.prepare_partial(self.x)
            new_embedding.optimize(10, learning_rate=0.1, inplace=True)
            self.eval_embedding(new_embedding, f"transform::{aff.__class__.__name__}")

            # With initilization
            embedding = TSNE().fit(affinities=aff, initialization=init)
            self.eval_embedding(embedding, aff.__class__.__name__)
            new_embedding = embedding.prepare_partial(self.x)
            new_embedding.optimize(10, learning_rate=0.1, inplace=True)
            self.eval_embedding(new_embedding, f"transform::{aff.__class__.__name__}")

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
            self.eval_embedding(embedding, aff.__class__.__name__)

            # With initilization
            embedding = TSNE().fit(affinities=aff, initialization=init)
            self.eval_embedding(embedding, aff.__class__.__name__)

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
            self.eval_embedding(embedding, aff.__class__.__name__)

            # With initilization
            embedding = TSNE().fit(affinities=aff, initialization=init)
            self.eval_embedding(embedding, aff.__class__.__name__)
