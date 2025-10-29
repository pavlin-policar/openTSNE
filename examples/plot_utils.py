import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import re


def plot_tsne_result(
    data,
    labels: np.ndarray,
    additional_title: str = "",
    marker_size: int = 3,
    *,
    black_template: bool = False,
    save_figure: bool = False,
) -> None:
    if black_template:
        plt.style.use("dark_background")
        text_color = "white"
        grid_color = "#2a2a2a"
        hline_colors = {"kl": "orange", "alpha": "lime", "grad": "red"}
    else:
        plt.style.use("default")
        text_color = "black"
        grid_color = "#cccccc"
        hline_colors = {"kl": "black", "alpha": "black", "grad": "black"}

    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(
        2,
        3,
        figure=fig,
        height_ratios=[0.6, 0.2],
        hspace=0.25,
        wspace=0.3,
        top=0.88,
        bottom=0.08,
    )

    # t-SNE Embedding Plot (spans all columns in first row)
    ax_main = fig.add_subplot(gs[0, :])
    scatter = ax_main.scatter(
        data.embedding[:, 0],
        data.embedding[:, 1],
        c=labels,
        s=marker_size,
        cmap="jet",
        alpha=0.8,
    )
    ax_main.set_aspect("equal", adjustable="box")
    ax_main.axis("off")

    # KL Divergence Plot
    ax_kl = fig.add_subplot(gs[1, 0])
    min_kl = data.im_KLs[-1]
    im_kls_filtered = [kl for kl in data.im_KLs if kl != 0]
    ax_kl.plot(
        range(len(im_kls_filtered)),
        im_kls_filtered,
        "o-",
        color="dodgerblue",
        markersize=2,
        linewidth=1,
    )
    ax_kl.axhline(
        y=min_kl, color=hline_colors["kl"], linestyle="--", alpha=0.8, linewidth=1.5
    )
    ax_kl.text(
        len(im_kls_filtered) * 0.98,
        min_kl * 1.02,
        "min L",
        ha="right",
        va="bottom",
        color=hline_colors["kl"],
        fontfamily="monospace",
        fontsize=10,
    )
    ax_kl.set_title(
        "KL Divergence", fontfamily="monospace", fontsize=14, color=text_color, pad=10
    )
    ax_kl.set_yticks([min_kl])
    ax_kl.set_yticklabels([f"{min_kl:.2f}"], fontfamily="monospace")
    ax_kl.set_xticks([0, len(data.im_KLs) // 2, len(data.im_KLs)])
    ax_kl.set_xticklabels(
        ["0", f"{len(data.im_KLs) // 2}", f"{len(data.im_KLs)}"], fontfamily="monospace"
    )
    ax_kl.set_xlim(0, len(data.im_KLs))
    ax_kl.grid(True, alpha=0.2, color=grid_color)

    # Alpha Values Plot
    ax_alpha = fig.add_subplot(gs[1, 1])
    last_alpha = data.im_alphas[-1]
    ax_alpha.plot(
        range(len(data.im_alphas)), data.im_alphas, color="red", linewidth=1.5
    )
    ax_alpha.axhline(
        y=last_alpha,
        color=hline_colors["alpha"],
        linestyle="--",
        alpha=0.8,
        linewidth=1.5,
    )
    ax_alpha.text(
        len(data.im_alphas) * 0.98,
        last_alpha * 0.98,
        "Converged α",
        ha="right",
        va="top",
        color=hline_colors["alpha"],
        fontfamily="monospace",
        fontsize=10,
    )
    ax_alpha.set_title(
        "Alpha Values", fontfamily="monospace", fontsize=14, color=text_color, pad=10
    )
    ax_alpha.set_yticks([last_alpha])
    ax_alpha.set_yticklabels([f"{last_alpha:.2f}"], fontfamily="monospace")
    ax_alpha.set_xlabel(
        "Iterations", fontfamily="monospace", fontsize=14, color=text_color
    )
    ax_alpha.set_xticks([0, len(data.im_KLs) // 2, len(data.im_KLs)])
    ax_alpha.set_xticklabels(
        ["0", f"{len(data.im_KLs) // 2}", f"{len(data.im_KLs)}"], fontfamily="monospace"
    )
    ax_alpha.set_xlim(0, len(data.im_KLs))
    ax_alpha.grid(True, alpha=0.2, color=grid_color)

    # Alpha Gradient Plot
    ax_grad = fig.add_subplot(gs[1, 2])
    ax_grad.plot(
        range(len(data.im_alpha_grads)),
        data.im_alpha_grads,
        color="lime",
        linewidth=1.5,
    )

    # Set ylim to center the zero line
    y_max = max(abs(min(data.im_alpha_grads)), abs(max(data.im_alpha_grads)))
    ax_grad.set_ylim(-y_max, y_max)

    ax_grad.axhline(
        y=0, color=hline_colors["grad"], linestyle="--", alpha=0.8, linewidth=1.5
    )
    ax_grad.text(
        len(data.im_alpha_grads) * 0.98,
        0,
        "Zero Gradient",
        ha="right",
        va="bottom",
        color=hline_colors["grad"],
        fontfamily="monospace",
        fontsize=10,
    )
    ax_grad.set_title(
        "Alpha Gradient", fontfamily="monospace", fontsize=14, color=text_color, pad=10
    )
    ax_grad.set_yticks([0])
    ax_grad.set_yticklabels(["0"], fontfamily="monospace")
    ax_grad.set_xticks([0, len(data.im_KLs) // 2, len(data.im_KLs)])
    ax_grad.set_xticklabels(
        ["0", f"{len(data.im_KLs) // 2}", f"{len(data.im_KLs)}"], fontfamily="monospace"
    )
    ax_grad.set_xlim(0, len(data.im_KLs))
    ax_grad.grid(True, alpha=0.2, color=grid_color)

    # Title
    knn_recall = getattr(data, "knn_recall", "N/A")
    title_param_1 = "adaptive" if data.optimization_mode else "fixed"
    title_param_2 = (
        f"Initial α = {data.initial_alpha}, α learning rate = {data.alpha_lr}"
        if data.optimization_mode
        else f"α = {data.initial_alpha}"
    )
    title_param_3 = f"Final α = {data.im_alphas[-1]:.2f}, Final L = {data.kl_divergence:.2f}, kNN recall = {knn_recall:.2f}"

    fig_title = (
        f"t-SNE embedding with {title_param_1} α parameter\n"
        f"{data.dataset_name} with {data.n_samples} samples\n"
        f"{title_param_2}\n{title_param_3}"
    )

    if additional_title:
        fig_title += f"\n{additional_title}"

    fig.suptitle(
        fig_title, fontfamily="monospace", fontsize=14, color=text_color, y=0.98
    )

    if save_figure:
        sanitized_fig_title = re.sub(r'[<>:"/\\|?*]', "_", fig_title.split("\n")[0])
        fig_path = Path(f"figures/{sanitized_fig_title} (matplotlib).pdf")
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            fig_path, bbox_inches="tight", dpi=300, facecolor=fig.get_facecolor()
        )

    plt.show()


def plot_tsne_comparison(
    data1,
    data2,
    labels: np.ndarray,
    title: str = "",
    marker_size: int = 3,
    save_figure: bool = False,
) -> None:
    plt.style.use("dark_background")
    text_color = "white"
    grid_color = "#2a2a2a"

    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(
        2,
        6,
        figure=fig,
        height_ratios=[0.6, 0.2],
        hspace=0.25,
        wspace=0.25,
        top=0.94,
        bottom=0.08,
    )

    x_ticks = [0, len(data1.im_KLs) // 2, len(data1.im_KLs)]

    def add_embedding(data, col_start, row, labels_array):
        ax = fig.add_subplot(gs[row, col_start : col_start + 3])
        ax.scatter(
            data.embedding[:, 0],
            data.embedding[:, 1],
            c=labels_array,
            s=marker_size,
            cmap="rainbow",
            alpha=0.8,
        )
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        return ax

    def add_metric_plot(
        data, col, row, metric, name, line_color, hline=None, hline_color=None
    ):
        ax = fig.add_subplot(gs[row, col])
        metric_filtered = [m for m in metric if m != 0]

        ax.plot(
            range(len(metric_filtered)),
            metric_filtered,
            "o-",
            color=line_color,
            markersize=2,
            linewidth=1,
        )

        # Center zero line for gradient plots
        if hline == 0:
            y_max = max(abs(min(metric_filtered)), abs(max(metric_filtered)))
            ax.set_ylim(-y_max, y_max)

        if hline is not None:
            h_color = hline_color if hline_color else line_color
            ax.axhline(y=hline, color=h_color, linestyle="--", alpha=0.8, linewidth=1.5)
            pos = "bottom" if hline == 0 else "top"
            va = "bottom" if hline == 0 else "top"
            offset = 1.02 if pos == "top" else 0.98
            ax.text(
                len(metric_filtered) * 0.98,
                hline * offset,
                name,
                ha="right",
                va=va,
                color=h_color,
                fontfamily="monospace",
                fontsize=10,
            )

        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(tick) for tick in x_ticks], fontfamily="monospace")
        ax.set_xlim(0, len(metric_filtered))

        if hline is not None:
            ax.set_yticks([hline])
            ax.set_yticklabels([f"{hline:.2f}"], fontfamily="monospace")

        if col in (1, 4):
            ax.set_xlabel(
                "Iterations", fontfamily="monospace", fontsize=16, color=text_color
            )

        ax.grid(True, alpha=0.2, color=grid_color)
        return ax

    # First embedding and metrics
    add_embedding(data1, 0, 0, labels)
    ax_kl1 = add_metric_plot(
        data1, 0, 1, data1.im_KLs, "min L", "dodgerblue", data1.im_KLs[-1], "orange"
    )
    ax_kl1.set_title(
        "KL Divergence", fontfamily="monospace", fontsize=16, color=text_color, pad=10
    )

    ax_alpha1 = add_metric_plot(
        data1, 1, 1, data1.im_alphas, "Converged α", "red", data1.im_alphas[-1], "lime"
    )
    ax_alpha1.set_title(
        "Alpha Values", fontfamily="monospace", fontsize=16, color=text_color, pad=10
    )

    ax_grad1 = add_metric_plot(
        data1, 2, 1, data1.im_alpha_grads, "Zero Gradient", "lime", 0, "red"
    )
    ax_grad1.set_title(
        "Alpha Gradient", fontfamily="monospace", fontsize=16, color=text_color, pad=10
    )

    # Second embedding and metrics
    add_embedding(data2, 3, 0, labels)
    ax_kl2 = add_metric_plot(
        data2, 3, 1, data2.im_KLs, "min L", "dodgerblue", data2.im_KLs[-1], "orange"
    )
    ax_kl2.set_title(
        "KL Divergence", fontfamily="monospace", fontsize=16, color=text_color, pad=10
    )

    ax_alpha2 = add_metric_plot(
        data2, 4, 1, data2.im_alphas, "Converged α", "red", data2.im_alphas[-1], "lime"
    )
    ax_alpha2.set_title(
        "Alpha Values", fontfamily="monospace", fontsize=16, color=text_color, pad=10
    )

    ax_grad2 = add_metric_plot(
        data2, 5, 1, data2.im_alpha_grads, "Zero Gradient", "lime", 0, "red"
    )
    ax_grad2.set_title(
        "Alpha Gradient", fontfamily="monospace", fontsize=16, color=text_color, pad=10
    )

    # Generate title if embeddings are the same
    if data1.dataset_name == data2.dataset_name and data1.n_samples == data2.n_samples:
        fig_title = f"Comparison of t-SNE embeddings for {data1.dataset_name} with {data1.n_samples} samples"
    else:
        fig_title = "Comparison of t-SNE embeddings"

    if not title:
        title = fig_title

    fig.suptitle(title, fontfamily="monospace", fontsize=16, color=text_color, y=0.98)

    if save_figure:
        sanitized_fig_title = re.sub(r'[<>:"/\\|?*]', "_", fig_title)
        fig_path = Path(f"figures/{sanitized_fig_title} (matplotlib).pdf")
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            fig_path, bbox_inches="tight", dpi=300, facecolor=fig.get_facecolor()
        )

    plt.show()


def plot_swiss_roll(
    sr_points: np.ndarray,
    sr_color: np.ndarray,
    n_samples: int,
    row: int = 1,
    col: int = 1,
    fig: plt.Figure | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
    width: int = 6,
    height: int = 6,
    marker_size: int = 3,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot Swiss Roll in 3D using matplotlib.

    Parameters:
    -----------
    sr_points : np.ndarray
        3D coordinates of points
    sr_color : np.ndarray
        Color values for each point
    n_samples : int
        Number of samples
    row : int
        Subplot row position (1-indexed)
    col : int
        Subplot column position (1-indexed)
    fig : plt.Figure or None
        Existing figure to add to, or None to create new
    ax : plt.Axes or None
        Existing 3D axis to plot on, or None to create new
    title : str or None
        Custom title, or None for default
    width : int
        Figure width in inches (per subplot)
    height : int
        Figure height in inches (per subplot)
    marker_size : int
        Size of markers

    Returns:
    --------
    tuple[plt.Figure, plt.Axes]
        The figure and axis objects
    """
    plt.style.use("dark_background")

    # If no existing figure is passed, create a new one
    if fig is None and ax is None:
        fig = plt.figure(figsize=(width, height))
        ax = fig.add_subplot(111, projection="3d")
    elif ax is None:
        # If figure exists but no axis, create axis at specified position
        ax = fig.add_subplot(row, col, (row - 1) * col + col, projection="3d")

    # Add the Swiss Roll 3D scatter plot
    scatter = ax.scatter(
        sr_points[:, 0],
        sr_points[:, 1],
        sr_points[:, 2],
        c=sr_color,
        s=marker_size,
        cmap="rainbow",
        alpha=0.8,
    )

    # Set labels
    ax.set_xlabel("X", fontfamily="monospace", fontsize=12)
    ax.set_ylabel("Y", fontfamily="monospace", fontsize=12)
    ax.set_zlabel("Z", fontfamily="monospace", fontsize=12)

    # Set camera perspective (elevation, azimuth)
    ax.view_init(elev=20, azim=45)

    # Set title
    if title is None:
        title = "Swiss Roll in Ambient Space"
    ax.set_title(title, fontfamily="monospace", fontsize=14, color="white", pad=20)

    # Adjust tick label sizes
    ax.tick_params(axis="both", which="major", labelsize=10)

    # Set background color for 3D panes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("gray")
    ax.yaxis.pane.set_edgecolor("gray")
    ax.zaxis.pane.set_edgecolor("gray")
    ax.grid(True, alpha=0.2)

    # Add text annotation for the number of samples
    ax.text2D(
        0.95,
        0.05,
        f"N samples={n_samples}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color="white",
        fontfamily="monospace",
        fontsize=12,
    )

    plt.tight_layout()

    return fig, ax
