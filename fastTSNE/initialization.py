import numpy as np
from sklearn.decomposition import PCA
from sklearn.utils import check_random_state


def random(n_samples, n_components, random_state=None):
    """Initialize an embedding using samples from an isotropic Gaussian.

    Parameters
    ----------
    n_samples: int
    n_components: int
    random_state:

    Returns
    -------
    np.ndarray

    """
    random_state = check_random_state(random_state)
    return random_state.normal(0, 1e-2, (n_samples, n_components))


def pca(X, n_components=2, scale_down=True, random_state=None):
    """Initialize am embedding using the top principal components.

    Parameters
    ----------
    X: np.ndarray
    n_components: int
    scale_down: bool
    random_state:

    Returns
    -------
    np.ndarray

    """
    pca_ = PCA(n_components=n_components, random_state=random_state)
    embedding = pca_.fit_transform(X)

    # The PCA embedding may have high variance, which leads to poor convergence
    if scale_down:
        normalization = np.std(embedding[:, 0]) * 100
        embedding /= normalization

    return embedding


def weighted_mean(X, embedding, neighbors, distances):
    """Initialize points onto an existing embedding by placing them in the
    weighted mean position of their nearest neighbors on the reference embedding.

    Parameters
    ----------
    X: np.ndarray
    embedding: TSNEEmbedding
    neighbors: np.ndarray
    distances: np.ndarray

    Returns
    -------
    np.ndarray

    """
    n_samples = X.shape[0]
    n_components = embedding.shape[1]

    partial_embedding = np.zeros((n_samples, n_components))
    for i in range(n_samples):
        partial_embedding[i] = np.average(
            embedding[neighbors[i]], axis=0, weights=distances[i],
        )

    return partial_embedding


def median(embedding, neighbors):
    """Initialize points onto an existing embedding by placing them in the
    median position of their nearest neighbors on the reference embedding.

    Parameters
    ----------
    embedding: TSNEEmbedding
    neighbors: np.ndarray

    Returns
    -------
    np.ndarray

    """
    return np.median(embedding[neighbors], axis=1)
