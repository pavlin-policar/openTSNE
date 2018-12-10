import numpy as np
from sklearn.decomposition import PCA
from sklearn.utils import check_random_state


def random(X, n_components=2, random_state=None):
    """Initialize an embedding using samples from an isotropic Gaussian.

    Parameters
    ----------
    X: np.ndarray
        The data matrix.

    n_components: int
        The dimension of the embedding space.

    random_state: Union[int, RandomState]
        If the value is an int, random_state is the seed used by the random
        number generator. If the value is a RandomState instance, then it will
        be used as the random number generator. If the value is None, the random
        number generator is the RandomState instance used by `np.random`.

    Returns
    -------
    initialization: np.ndarray

    """
    random_state = check_random_state(random_state)
    return random_state.normal(0, 1e-2, (X.shape[0], n_components))


def pca(X, n_components=2, random_state=None):
    """Initialize an embedding using the top principal components.

    Parameters
    ----------
    X: np.ndarray
        The data matrix.

    n_components: int
        The dimension of the embedding space.

    random_state: Union[int, RandomState]
        If the value is an int, random_state is the seed used by the random
        number generator. If the value is a RandomState instance, then it will
        be used as the random number generator. If the value is None, the random
        number generator is the RandomState instance used by `np.random`.

    Returns
    -------
    initialization: np.ndarray

    """
    pca_ = PCA(n_components=n_components, random_state=random_state)
    embedding = pca_.fit_transform(X)

    # The PCA embedding may have high variance, which leads to poor convergence
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
