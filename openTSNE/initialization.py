import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.utils import check_random_state

from openTSNE import utils


def rescale(x, inplace=False):
    """Rescale an embedding so optimization will not have convergence issues.

    Parameters
    ----------
    x: np.ndarray
    inplace: bool

    Returns
    -------
    np.ndarray
        A scaled-down version of ``x``.

    """
    if not inplace:
        x = np.array(x, copy=True)

    x /= np.std(x[:, 0]) * 10000

    return x


def random(X, n_components=2, random_state=None, verbose=False):
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

    verbose: bool

    Returns
    -------
    initialization: np.ndarray

    """
    random_state = check_random_state(random_state)
    embedding = random_state.normal(0, 1e-4, (X.shape[0], n_components))
    return np.ascontiguousarray(embedding)


def pca(X, n_components=2, svd_solver="auto", random_state=None, verbose=False):
    """Initialize an embedding using the top principal components.

    Parameters
    ----------
    X: np.ndarray
        The data matrix.

    n_components: int
        The dimension of the embedding space.

    svd_solver: str
        See sklearn.decomposition.PCA documentation.

    random_state: Union[int, RandomState]
        If the value is an int, random_state is the seed used by the random
        number generator. If the value is a RandomState instance, then it will
        be used as the random number generator. If the value is None, the random
        number generator is the RandomState instance used by `np.random`.

    verbose: bool

    Returns
    -------
    initialization: np.ndarray

    """
    timer = utils.Timer("Calculating PCA-based initialization...", verbose)
    timer.__enter__()

    pca_ = PCA(
        n_components=n_components, svd_solver=svd_solver, random_state=random_state
    )
    embedding = pca_.fit_transform(X)
    rescale(embedding, inplace=True)

    timer.__exit__()

    return np.ascontiguousarray(embedding)


def spectral(A, n_components=2, tol=1e-4, max_iter=None, random_state=None, verbose=False):
    """Initialize an embedding using the spectral embedding of the KNN graph.

    Specifically, we initialize data points by computing the diffusion map on
    the random walk transition matrix of the weighted graph given by the affiniy
    matrix.

    Parameters
    ----------
    A: Union[sp.csr_matrix, sp.csc_matrix, ...]
        The graph adjacency matrix.

    n_components: int
        The dimension of the embedding space.

    tol: float
        See scipy.sparse.linalg.eigsh documentation.

    max_iter: float
        See scipy.sparse.linalg.eigsh documentation.

    random_state: Any
        Unused, but kept for consistency between initialization schemes.

    verbose: bool

    Returns
    -------
    initialization: np.ndarray

    """
    if A.ndim != 2:
        raise ValueError("The graph adjacency matrix must be a 2-dimensional matrix.")
    if A.shape[0] != A.shape[1]:
        raise ValueError("The graph adjacency matrix must be a square matrix.")

    timer = utils.Timer("Calculating spectral initialization...", verbose)
    timer.__enter__()

    D = sp.diags(np.ravel(np.sum(A, axis=1)))

    # Find leading eigenvectors
    k = n_components + 1
    v0 = np.ones(A.shape[0]) / np.sqrt(A.shape[0])
    eigvals, eigvecs = sp.linalg.eigsh(
        A, M=D, k=k, tol=tol, maxiter=max_iter, which="LM", v0=v0
    )
    # Sort the eigenvalues in decreasing order
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]

    # In diffusion maps, we multiply the eigenvectors by their eigenvalues
    eigvecs *= eigvals

    # Drop the leading eigenvector
    embedding = eigvecs[:, 1:]

    rescale(embedding, inplace=True)

    timer.__exit__()

    return embedding


def weighted_mean(X, embedding, neighbors, distances, verbose=False):
    """Initialize points onto an existing embedding by placing them in the
    weighted mean position of their nearest neighbors on the reference embedding.

    Parameters
    ----------
    X: np.ndarray
    embedding: TSNEEmbedding
    neighbors: np.ndarray
    distances: np.ndarray
    verbose: bool

    Returns
    -------
    np.ndarray

    """
    n_samples = X.shape[0]
    n_components = embedding.shape[1]

    with utils.Timer("Calculating weighted-mean initialization...", verbose):
        partial_embedding = np.zeros((n_samples, n_components), order="C")
        for i in range(n_samples):
            partial_embedding[i] = np.average(
                embedding[neighbors[i]], axis=0, weights=distances[i]
            )

    return partial_embedding


def median(embedding, neighbors, verbose=False):
    """Initialize points onto an existing embedding by placing them in the
    median position of their nearest neighbors on the reference embedding.

    Parameters
    ----------
    embedding: TSNEEmbedding
    neighbors: np.ndarray
    verbose: bool

    Returns
    -------
    np.ndarray

    """
    with utils.Timer("Calculating meadian initialization...", verbose):
        embedding = np.median(embedding[neighbors], axis=1)
    return np.ascontiguousarray(embedding)
