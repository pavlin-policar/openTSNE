from dataclasses import dataclass
import numpy as np
import sklearn.datasets

from openTSNE import TSNE, TSNEEmbedding
from openTSNE.affinity import PerplexityBasedNN
from openTSNE.initialization import pca

import time
from sklearn.neighbors import NearestNeighbors
from plot_utils import plot_swiss_roll


@dataclass
class TSNEResult:
    """Dataclass to store the results of a single t-SNE algorithm run."""

    dataset_name: str
    n_samples: int
    n_iter: int
    embedding: np.ndarray
    optimization_mode: str | None
    kl_divergence: float
    initial_alpha: float
    alpha_lr: float | None
    im_embeddings: list
    im_KLs: np.ndarray
    im_alphas: np.ndarray
    im_alpha_grads: np.ndarray
    knn_recall: float | None


@dataclass
class SwissRoll:
    datapoints: np.ndarray
    labels: np.ndarray
    n_samples: int

    @classmethod
    def generate(cls, n_samples: int, noise: float) -> "SwissRoll":
        datapoints, labels = sklearn.datasets.make_swiss_roll(
            n_samples=n_samples, noise=noise
        )
        return SwissRoll(datapoints=datapoints, labels=labels, n_samples=n_samples)

    def plot(
        self, width: int = 600, height: int = 600, title: str | None = None
    ) -> None:
        plot_swiss_roll(
            self.datapoints,
            self.labels,
            n_samples=self.n_samples,
            width=width,
            height=height,
            title=title,
        )


def perepare_initial_embedding(
    data: np.ndarray,
    perplexity: float,
    initial_dof: float,
    n_components: int,
    random_state: int,
) -> TSNEEmbedding:
    """Prepare the initial t-SNE embedding using the OpenTSNE implementation.

    Parameters
    ----------
    data : np.ndarray
        The data to run t-SNE on.
    perplexity : float
        The perplexity parameter of the t-SNE algorithm
    initial_dof : float
        The initial degree-of-freedom (alpha) parameter of the t-SNE algorithm
    n_components : int
        The number of components to reduce the data to
    random_state : int
        The random state to use for the t-SNE algorithm

    Returns
    -------
    OpTSNEEmbedding
        The initial t-SNE embedding.

    """
    affinities_obj = PerplexityBasedNN(
        data=data,
        perplexity=perplexity,
        metric="euclidean",
        random_state=random_state,
    )

    pca_init = pca(data, random_state=random_state)
    return TSNE(
        n_components=n_components, dof=initial_dof, random_state=random_state
    ).prepare_initial(X=data, affinities=affinities_obj, initialization=pca_init)


def run_early_exaggeration_phase(
    initial_embedding: TSNEEmbedding,
    initial_alpha: float,
    n_jobs: int = 1,
    exagerration: int = 12,
    n_iter: int = 250,
) -> TSNEEmbedding:
    """Runs the standard early exaggeration phase of the t-SNE algorithm.

    Parameters
    ----------
    initial_embedding : OpTSNEEmbedding | TSNEEmbedding
        The initial t-SNE embedding.
    initial_alpha : float
        The initial degree-of-freedom parameter of the t-SNE algorithm.
    n_jobs : int, optional
        The number of jobs to use for the t-SNE algorithm, by default 1
    exagerration : int, optional
        The exaggeration factor for the early exaggeration phase, by default 12
    n_iter : int, optional
        The number of iterations to run the early exagerration phase, by default 250

    Returns
    -------
    OpTSNEEmbedding
        The t-SNE embedding after the early exaggeration phase.

    """
    n_samples = initial_embedding.shape[0]

    default_learning_rate = n_samples / exagerration

    print(  # noqa: T201
        f"Performing the early exaggeration fase with exaggeration = {exagerration} and learning rate = {default_learning_rate:.2f} for {n_iter} iterations..."
    )

    return initial_embedding.optimize(
        n_iter,
        exaggeration=exagerration,
        learning_rate=default_learning_rate,
        negative_gradient_method="bh",
        inplace=True,
        dof=initial_alpha,
        # optimize_for_alpha=False,
        verbose=True,
        n_jobs=n_jobs,
    )


def run_tsne(
    data: np.ndarray,
    *,
    perplexity: float,
    dof: float | str,
    n_iter: int,
    negative_gradient_method: str,
    dataset_name: str,
    initial_dof: float = 1.0,
    dof_lr: float | None = 0.5,
    dof_optimizer: str = "delta_bar_delta",
    n_jobs: int = 1,
    callbacks_every_iters: int = 1,
    eval_error_every_iter: int = 1,
    random_state: int = 42,
    n_components: int = 2,
) -> TSNEResult:
    """Runs the OpenTSNE implementation of the t-SNE algorithm on a dataset.

    Parameters
    ----------
    data : np.ndarray
        The data to run t-SNE on.
    perplexity : float
        The perplexity parameter of the t-SNE algorithm.
    initial_dof : float
        The initial degree-of-freedom (alpha) parameter of the t-SNE algorithm.
    optimize_for_dof : bool
        Whether to optimize for the degree-of-freedom parameter
    dof_lr : float | None
        The learning rate to use for the degree-of-freedom optimization, optional
    n_iter : int
        The number of iterations to run the t-SNE algorithm for.
    n_jobs : int, optional
        The number of jobs to use for the t-SNE algorithm, by default 1
    callbacks_every_iters : int, optional
        How many iterations should pass between each time the callbacks are invoked, by default 1
    eval_error_every_iter : int, optional
        How many iterations should pass between each time the error is evaluated, by default 1
    negative_gradient_method : str
        The method to use for computing the negative gradient
    dataset_name : str
        The name of the dataset
    random_state : int, optional
        The random state to use for the t-SNE algorithm, by default 42
    n_components : int, optional
        The number of components to reduce the data to, by default 2

    Returns
    -------
    TSNEResultsWithKNN
        The dataclass containing the results of the t-SNE algorithm and the KNN recall.

    """
    alpha_lr = 0.5 if dof_lr is None else dof_lr
    n_samples = data.shape[0]

    initial_embedding = perepare_initial_embedding(
        data, perplexity, initial_dof, n_components, random_state
    )

    embedding = run_early_exaggeration_phase(
        initial_embedding,
        initial_dof,
        n_jobs=n_jobs,
    )
    tic = time.time()

    optimized_embedding = embedding.optimize(
        n_iter=n_iter,
        negative_gradient_method=negative_gradient_method,
        inplace=True,
        verbose=True,
        dof=dof,
        dof_lr=dof_lr,
        dof_optimizer=dof_optimizer,
        n_jobs=n_jobs,
        use_callbacks=True,
        eval_error_every_iter=eval_error_every_iter,
        callbacks_every_iters=callbacks_every_iters,
    )
    toc = time.time()
    print(f"Optimization took {toc - tic:.2f} seconds")  # noqa: T201
    records = optimized_embedding.optimization_stats

    knn_recall = compute_knn_recall(data, optimized_embedding, 10)

    return TSNEResult(
        dataset_name=dataset_name,
        n_samples=n_samples,
        n_iter=n_iter,
        embedding=optimized_embedding,
        optimization_mode="BH",
        kl_divergence=optimized_embedding.kl_divergence,
        initial_alpha=initial_dof,
        alpha_lr=alpha_lr,
        im_embeddings=records.embeddings,
        im_KLs=records.KLs,
        im_alphas=records.alphas,
        im_alpha_grads=records.alpha_gradients,
        knn_recall=knn_recall,
    )


def compute_knn_recall(
    original_data: np.ndarray, tsne_data: np.ndarray, k: int = 10
) -> float:
    """Computes the recall of k-nearest neighbors between the original data and the t-SNE data.

    Parameters
    ----------
    original_data : np.ndarray
        The original multidimensional data.
    tsne_data : np.ndarray
        The t-SNE transformed data.
    k : int, optional
        The number of neighbors to consider, by default 7

    Returns
    -------
    float
        The average recall of k-nearest neighbors between the original data and the t-SNE data.

    Notes
    -----
    The formula is taken from: Gove et al. (2022)
    New guidance for using t-SNE: Alternative defaults, hyperparameter selection automation,
    and comparative evaluation,
    Visual Informatics, Volume 6, Issue 2, 2022,

    """
    # Fit kNN on original data
    knn_orig = NearestNeighbors(n_neighbors=k, metric="euclidean")
    knn_orig.fit(original_data)
    orig_neighbors = knn_orig.kneighbors(return_distance=False)

    # Fit kNN on t-SNE data
    knn_tsne = NearestNeighbors(n_neighbors=k, metric="euclidean")
    knn_tsne.fit(tsne_data)
    tsne_neighbors = knn_tsne.kneighbors(return_distance=False)

    # Calculate recall for each point
    recall_scores = np.zeros(len(original_data))
    for i in range(len(original_data)):
        shared_neighbors = np.intersect1d(orig_neighbors[i], tsne_neighbors[i])
        recall = len(shared_neighbors) / k
        recall_scores[i] = recall
    # Return average recall
    return np.mean(recall_scores)
