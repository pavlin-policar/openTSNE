import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
import sklearn.datasets
from plotly.subplots import make_subplots
from tensorflow.keras.datasets import mnist

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


def plot_tsne_result(
    data: TSNEResult,
    labels: np.ndarray,
    additional_title: str = "",
    marker_size: int = 3,
    *,
    black_template: bool = False,
    save_figure: bool = False,
) -> None:
    if black_template:
        template = "plotly_dark"
        text_color = "white"
    else:
        template = "plotly"
        text_color = "black"

    # Create subplot layout
    fig = sp.make_subplots(
        rows=2,
        cols=3,
        specs=[[{"colspan": 3}, None, None], [{}, {}, {}]],
        subplot_titles=["", "KL Divergence", "Alpha Values", "Alpha Gradient"],
        vertical_spacing=0.05,
        horizontal_spacing=0.1,
        row_heights=[0.6, 0.2],
        column_widths=[0.2, 0.2, 0.2],
    )

    # t-SNE Embedding Plot
    fig.add_trace(
        go.Scatter(
            x=data.embedding[:, 0],
            y=data.embedding[:, 1],
            mode="markers",
            marker={
                "color": labels,
                "size": marker_size,
                "showscale": False,
                "colorscale": "Jet",
            },
            name="t-SNE Embedding",
        ),
        row=1,
        col=1,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=1)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=1)

    # KL Divergence Plot
    min_kl = data.im_KLs[-1]
    im_kls_filtered = [kl for kl in data.im_KLs if kl != 0]
    fig.add_trace(
        go.Scatter(
            x=list(range(len(im_kls_filtered))),
            y=im_kls_filtered,
            mode="markers+lines",
            marker={"size": 2, "color": "blue"},
            line={"color": "blue"},
            name="KL Divergence",
        ),
        row=2,
        col=1,
    )
    fig.add_hline(
        y=min_kl,
        line_dash="dash",
        line_color="orange" if black_template else "black",
        annotation_text="min 𝓛",
        annotation_position="top right",
        opacity=0.8,
        row=2,
        col=1,
    )
    fig.update_yaxes(tickvals=[min_kl], ticktext=[f"{min_kl:.2f}"], row=2, col=1)

    # Alpha Values Plot
    last_alpha = data.im_alphas[-1]
    fig.add_trace(
        go.Scatter(
            x=list(range(len(data.im_alphas))),
            y=data.im_alphas,
            mode="lines",
            line={"color": "red"},
            name="Alpha Values",
        ),
        row=2,
        col=2,
    )
    fig.add_hline(
        y=last_alpha,
        line_dash="dash",
        line_color="green" if black_template else "black",
        annotation_text="Converged α",
        annotation_position="bottom right",
        opacity=0.8,
        row=2,
        col=2,
    )
    fig.update_yaxes(
        tickvals=[last_alpha], ticktext=[f"{last_alpha:.2f}"], row=2, col=2
    )

    # Alpha Gradient Plot
    fig.add_trace(
        go.Scatter(
            x=list(range(len(data.im_alpha_grads))),
            y=data.im_alpha_grads,
            mode="lines",
            line={"color": "green"},
            name="Alpha Gradient",
        ),
        row=2,
        col=3,
    )

    fig.update_yaxes(tickvals=[0], ticktext=["0"], row=2, col=3)

    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="red" if black_template else "black",
        annotation_text="Zero Gradient",
        annotation_position="bottom right",
        opacity=0.8,
        row=2,
        col=3,
    )

    for i in range(1, 4):
        fig.update_xaxes(
            tickvals=[0, len(data.im_KLs) // 2, len(data.im_KLs)],
            ticktext=["0", f"{len(data.im_KLs) // 2}", f"{len(data.im_KLs)}"],
            range=[0, len(data.im_KLs)],
            row=2,
            col=i,
        )

    # Titles and Layout
    fig.update_annotations(
        font={"size": 16, "color": text_color, "family": "Courier New, monospace"}
    )
    axis_label_style = {
        "size": 16,
        "color": text_color,
        "family": "Courier New, monospace",
    }
    fig.update_xaxes(
        title={"text": "Iterations", "font": axis_label_style}, row=2, col=2
    )
    knn_recall = getattr(data, "knn_recall", "N/A")
    title_param_1 = "adaptive" if data.optimization_mode else "fixed"
    title_param_2 = (
        f"Initial α = {data.initial_alpha}, α learning rate = {data.alpha_lr}"
        if data.optimization_mode
        else f"α = {data.initial_alpha}"
    )
    title_param_3 = f"Final α = {data.im_alphas[-1]:.2f}, Final 𝓛 = {data.kl_divergence:.2f}, kNN recall = {knn_recall:.2f}"

    fig_title = (
        f"t-SNE embedding with {title_param_1} α parameter<br>"
        f"{data.dataset_name} with {data.n_samples} samples<br>"
    )

    fig.update_layout(
        title={
            "text": (
                f"{fig_title}{title_param_2}<br>{title_param_3}<br>{additional_title}"
            ),
            "x": 0.5,
            "y": 0.95,
            "yanchor": "top",
            "font": {
                "size": 16,
                "color": text_color,
                "family": "Courier New, monospace",
            },
        },
        template=template,
        height=800,
        width=1000,
        showlegend=False,
    )

    # Sanitize the fig_title to remove invalid characters
    sanitized_fig_title = re.sub(r'[<>:"/\\|?*]', "_", fig_title)
    if save_figure:
        fig_path = Path(f"figures/{sanitized_fig_title} (plotly).pdf")
        fig_path.parent.mkdir(
            parents=True, exist_ok=True
        )  # Ensure the directory exists

        with fig_path.open("wb") as f:
            f.write(fig.to_image(format="pdf"))

    fig.show()


def plot_side_by_side(
    data1: TSNEResult,
    data2: TSNEResult,
    labels: np.ndarray,
    *,
    additional_title_1: str = "",
    additional_title_2: str = "",
    marker_size: int = 3,
    title: str = "",
    save_figure: bool = False,
) -> None:
    # Extract titles for the embeddings
    def get_titles(
        data: TSNEResult,
        knn_recall: float,
        additional_title: str = "",
    ) -> str:
        if data.optimization_mode:
            mode = "adaptive"
            params = f"Initial α = {data.initial_alpha:.2f}, α learning rate = {data.alpha_lr}"
            summary = (
                f"Final α = {data.im_alphas[-1]:.2f}, "
                f"Final Loss = {data.kl_divergence:.2f}, "
                f"kNN recall = {knn_recall:.2f}"
            )
        else:
            mode = "fixed"
            params = f"α = {data.initial_alpha:.2f}"
            summary = (
                f"Final Loss = {data.kl_divergence:.2f}, kNN recall = {knn_recall:.2f}"
            )

        return f"t-SNE embedding with {mode} α parameter<br>{params}<br>{summary}<br>{additional_title}"

    knn_recall_data1 = getattr(data1, "knn_recall", "N/A")
    knn_recall_data2 = getattr(data2, "knn_recall", "N/A")

    title1 = get_titles(data1, knn_recall_data1, additional_title_1)
    title2 = get_titles(data2, knn_recall_data2, additional_title_2)

    # Create subplot layout
    fig = sp.make_subplots(
        rows=2,
        cols=6,
        specs=[
            [{"colspan": 3}, None, None, {"colspan": 3}, None, None],
            [{}, {}, {}, {}, {}, {}],
        ],
        subplot_titles=[
            title1,
            title2,
            "",
            "",
            "",
            "",
        ],
        vertical_spacing=0.02,
        horizontal_spacing=0.05,
        row_heights=[0.7, 0.3],
        # lower the subtitles
    )

    # Final iteration for x-axis
    x_ticks = [0, len(data1.im_KLs) // 2, len(data1.im_KLs)]

    def add_embedding(data: TSNEResult, col: int, row: int, labels: np.ndarray) -> None:
        fig.add_trace(
            go.Scatter(
                x=data.embedding[:, 0],
                y=data.embedding[:, 1],
                mode="markers",
                marker={
                    "color": labels,
                    "size": marker_size,
                    "showscale": False,
                    "colorscale": "Rainbow",
                },
                name=f"t-SNE Embedding: {data.dataset_name}",
            ),
            row=row,
            col=col,
        )
        fig.update_xaxes(
            showgrid=False, zeroline=False, showticklabels=False, row=row, col=col
        )
        fig.update_yaxes(
            showgrid=False, zeroline=False, showticklabels=False, row=row, col=col
        )

    def add_metric_plot(
        data: TSNEResult,  # noqa: ARG001
        col: int,
        row: int,
        metric: list[float],
        name: str,
        line_color: str,
        hline: float | None = None,
        hline_color: str | None = None,
    ) -> None:
        # exclude 0 values from metric
        metric = [m for m in metric if m != 0]

        fig.add_trace(
            go.Scatter(
                x=list(range(len(metric))),
                y=metric,
                mode="markers+lines",
                marker={"size": 2, "color": line_color},
                line={"color": line_color},
                name=name,
            ),
            row=row,
            col=col,
        )
        if hline is not None:
            fig.add_hline(
                y=hline,
                line_dash="dash",
                line_color=hline_color if hline_color else line_color,
                annotation_text=name,
                annotation_position="bottom right" if hline == 0 else "top right",
                row=row,
                col=col,
            )
        fig.update_xaxes(
            tickvals=x_ticks,
            ticktext=[str(tick) for tick in x_ticks],
            range=[0, len(metric)],
            title={
                "text": "Iterations",
                "font": {"size": 24, "family": "Courier New, monospace"},
            }
            if col in (2, 5)
            else None,
            row=row,
            col=col,
        )

        fig.update_yaxes(row=row, col=col, tickvals=[hline], ticktext=[f"{hline:.2f}"])

    # Add first embedding and metrics
    add_embedding(data1, 1, 1, labels)
    add_metric_plot(
        data1, 1, 2, data1.im_KLs, "min 𝓛", "blue", data1.im_KLs[-1], "orange"
    )
    add_metric_plot(
        data1, 2, 2, data1.im_alphas, "Converged α", "red", data1.im_alphas[-1], "green"
    )
    add_metric_plot(
        data1, 3, 2, data1.im_alpha_grads, "Zero Gradient", "green", 0, "red"
    )

    # Add second embedding and metrics
    add_embedding(data2, 4, 1, labels)
    add_metric_plot(
        data2, 4, 2, data2.im_KLs, "min 𝓛", "blue", data2.im_KLs[-1], "orange"
    )
    add_metric_plot(
        data2,
        5,
        2,
        data2.im_alphas,
        "Converged α",
        "red",
        data2.im_alphas[-1],
        "green",
    )
    add_metric_plot(
        data2, 6, 2, data2.im_alpha_grads, "Zero Gradient", "green", 0, "red"
    )

    # Generate title if embeddings are the same
    if data1.dataset_name == data2.dataset_name and data1.n_samples == data2.n_samples:
        fig_title = f"Comparison of t-SNE embeddings for {data1.dataset_name} with {data1.n_samples} samples"
    else:
        fig_title = "Comparison of t-SNE embeddings"

    # Layout adjustments
    fig.update_annotations(
        font={"size": 20, "color": "white", "family": "Courier New, monospace"}
    )
    fig.update_layout(
        title={
            "text": f"{title}",
            "x": 0.5,
            "y": 0.99,
            "yanchor": "top",
            "font": {"size": 20, "color": "white", "family": "Courier New, monospace"},
        },
        template="plotly_dark",
        height=800,
        width=1600,
        showlegend=False,
    )
    if save_figure:
        sanitized_fig_title = re.sub(r'[<>:"/\\|?*]', "_", fig_title)
        fig_path = Path(f"figures/{sanitized_fig_title} (plotly).pdf")
        fig_path.parent.mkdir(
            parents=True, exist_ok=True
        )  # Ensure the directory exists

        with fig_path.open("wb") as f:
            f.write(fig.to_image(format="pdf"))

    fig.show()


# def plot_swiss_roll(
#     sr_points: np.ndarray,
#     sr_color: np.ndarray,
#     n_samples: int,
#     row: int = 1,
#     col: int = 1,
#     fig: go.Figure | None = None,
#     title: str | None = None,
#     width: int = 600,
#     height: int = 600,
#     marker_size: int = 3,
# ) -> go.Figure:
#     # If no existing figure is passed, create a new one
#     if fig is None:
#         fig = make_subplots(rows=row, cols=col, specs=[[{"type": "scene"}] * col] * row)

#     # Add the Swiss Roll 3D scatter plot
#     fig.add_trace(
#         go.Scatter3d(
#             x=sr_points[:, 0],
#             y=sr_points[:, 1],
#             z=sr_points[:, 2],
#             mode="markers",
#             marker={
#                 "size": marker_size,
#                 "color": sr_color,
#                 "colorscale": "Rainbow",  # Choose a colorscale
#                 "opacity": 0.8,
#             },
#         ),
#         row=row,
#         col=col,
#     )

#     # Update the layout for a dark background
#     fig.update_layout(
#         title="Swiss Roll in Ambient Space" if title is None else title,
#         template="plotly_dark",
#         scene={
#             "xaxis": {"title": "X"},
#             "yaxis": {"title": "Y"},
#             "zaxis": {"title": "Z"},
#             "camera": {
#                 "eye": {"x": -1, "y": 2, "z": 0.5},  # Camera perspective
#             },
#         },
#         width=width * col,
#         height=height * row,
#         font={"family": "Courier New, monospace", "size": 14},
#     )

#     fig.update_xaxes(
#         title_text="x",
#         title_font={"size": 12},  # Set the font size for the x-axis title
#         tickfont={"size": 10},  # Set the font size for the x-axis tick labels
#     )
#     fig.update_yaxes(
#         title_text="y",
#         title_font={"size": 12},  # Set the font size for the y-axis title
#         tickfont={"size": 10},  # Set the font size for the y-axis tick labels
#     )

#     # Add a text annotation for the number of samples
#     fig.add_annotation(
#         text=f"N samples={n_samples}",
#         xref="paper",
#         yref="paper",
#         x=1,
#         y=0.05,
#         showarrow=False,
#         font={"color": "white"},
#     )
#     fig.show()
#     return fig


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
