import numpy as np

from tsne.tsne import TSNEEmbedding


def pBIC(embedding: TSNEEmbedding) -> float:
    n_samples = embedding.shape[0]

    return 2 * embedding.kl_divergence + np.log(n_samples) * \
        embedding.perplexity / n_samples
