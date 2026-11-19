import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
import umap

NOISE = -1


def umap_project(
    X: np.ndarray,
    max_points: int | None = 20_000,
    seed: int = 42,
    **umap_kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Project embeddings to 2-D with UMAP, on a reproducible subsample.

    UMAP is for visualization only. Clustering should happen in the original
    (or PCA-reduced) space, not in this 2-D projection.

    Args:
        X (np.ndarray): Embeddings, shape (n, d).
        max_points (int | None): Subsample size, so UMAP stays fast. None = all points.
        seed (int): Seed for both the subsample and UMAP.
        **umap_kwargs: Passed to `umap.UMAP`, overriding the defaults
            (n_neighbors=15, min_dist=0.1, metric='euclidean').

    Returns:
        tuple[np.ndarray, np.ndarray]: The (m, 2) projection and the indices of
            the rows of X it covers. Use the indices to select matching labels.
    """
    if max_points is not None and len(X) > max_points:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(len(X), size=max_points, replace=False))
    else:
        indices = np.arange(len(X))

    params = {"n_components": 2, "n_neighbors": 15, "min_dist": 0.1, "metric": "euclidean"}
    reducer = umap.UMAP(**(params | umap_kwargs), random_state=seed)
    return reducer.fit_transform(X[indices]), indices


def plot_clusters_2d(
    embedding_2d: np.ndarray,
    labels: np.ndarray,
    title: str = "UMAP of embeddings colored by cluster",
    ax: Axes | None = None,
) -> Axes:
    """
    Scatter a 2-D projection colored by integer cluster label.

    The number of clusters is read from the labels. Noise points (-1, from
    DBSCAN and similar) are drawn in light gray behind the clusters.

    Args:
        embedding_2d (np.ndarray): (m, 2) projection, e.g. from `umap_project`.
        labels (np.ndarray): Cluster label per point, aligned with embedding_2d.
        title (str): Plot title.
        ax (plt.Axes | None): Axes to draw on, so plots can be put side by side.
            None creates a new figure.

    Returns:
        plt.Axes: The axes drawn on.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 8))

    noise = labels == NOISE
    clusters = np.unique(labels[~noise])
    n_clusters = len(clusters)

    if noise.any():
        ax.scatter(*embedding_2d[noise].T, c="lightgray", s=3, alpha=0.4,
                   linewidths=0, label=f"noise ({noise.mean():.0%})")
        ax.legend(loc="upper right")

    points = ax.scatter(
        *embedding_2d[~noise].T,
        c=labels[~noise],
        cmap="tab10" if n_clusters <= 10 else "tab20",
        vmin=clusters.min() - 0.5,
        vmax=clusters.max() + 0.5,
        s=5,
        alpha=0.7,
        linewidths=0,
    )
    if n_clusters <= 20:
        ax.figure.colorbar(points, ax=ax, ticks=clusters, label="Cluster")

    ax.set(title=title, xlabel="UMAP 1", ylabel="UMAP 2")
    return ax


def plot_kmeans_sweep(
    sweep: pd.DataFrame,
    title: str | None = None,
) -> tuple[Axes, Axes]:
    """
    Plot inertia and silhouette against k, side by side.

    Look for the elbow in inertia (where adding clusters stops helping much)
    and the peak in silhouette. If inertia falls in a straight line and
    silhouette stays near 0, the data has no clear cluster structure.

    Args:
        sweep (pd.DataFrame): Output of `utils.kmeans_sweep`, with columns
            'k', 'inertia' and 'silhouette'.
        title (str | None): Optional overall title, e.g. the model name.

    Returns:
        tuple[Axes, Axes]: The inertia and silhouette axes.
    """
    fig, (ax_inertia, ax_silhouette) = plt.subplots(1, 2, figsize=(12, 4))

    ax_inertia.plot(sweep["k"], sweep["inertia"], marker="o")
    ax_inertia.set(title="Inertia (look for the elbow)", xlabel="k", ylabel="Inertia")

    ax_silhouette.plot(sweep["k"], sweep["silhouette"], marker="o")
    ax_silhouette.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    best_row = int(sweep["silhouette"].to_numpy().argmax())   # position, not index label
    best_k = int(sweep["k"].iloc[best_row])
    best_silhouette = float(sweep["silhouette"].iloc[best_row])
    ax_silhouette.scatter(best_k, best_silhouette, s=120, facecolors="none",
                          edgecolors="red", zorder=3, label=f"best k = {best_k}")
    ax_silhouette.set(title="Silhouette (higher is better)", xlabel="k", ylabel="Silhouette")
    ax_silhouette.legend()

    for ax in (ax_inertia, ax_silhouette):
        ax.set_xticks(sweep["k"])
        ax.grid(True, alpha=0.3)

    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    return ax_inertia, ax_silhouette


def plot_cluster_samples(
    dataset,
    labels: np.ndarray,
    n_per_cluster: int = 8,
    captions: pd.Series | None = None,
    seed: int = 42,
) -> None:
    """
    Show random paintings from each cluster: one row per cluster.

    Use it to see what a cluster actually groups (style, subject, brightness,
    amount of detail...). Rows are sorted by cluster size, largest first.

    Args:
        dataset: Indexable dataset whose rows have an 'image' (PIL image or
            (3, H, W) tensor), aligned with `labels` row by row, e.g.
            `ds_train` from `get_splits` with an unshuffled config.
        labels (np.ndarray): Cluster label per row of `dataset`. Noise (-1)
            is shown as its own row.
        n_per_cluster (int): Images per row. Smaller clusters show all members.
        captions (pd.Series | None): Optional text under each image, aligned
            with `labels`, e.g. `meta["style"]`.
        seed (int): Seed for picking the samples, so reruns show the same images.
    """
    rng = np.random.default_rng(seed)
    clusters, sizes = np.unique(labels, return_counts=True)
    order = np.argsort(-sizes)

    fig, axes = plt.subplots(
        len(clusters), n_per_cluster,
        figsize=(2 * n_per_cluster, 2.3 * len(clusters)),
        squeeze=False,
    )
    for row, cluster_index in enumerate(order):
        cluster, size = clusters[cluster_index], sizes[cluster_index]
        members = np.flatnonzero(labels == cluster)
        picked = rng.choice(members, size=min(n_per_cluster, size), replace=False)

        for col in range(n_per_cluster):
            ax = axes[row, col]
            ax.axis("off")
            if col >= len(picked):
                continue
            image = dataset[int(picked[col])]["image"]
            if hasattr(image, "permute"):                     # tensor (3, H, W) -> (H, W, 3)
                image = image.permute(1, 2, 0).clamp(0, 1)
            ax.imshow(image)
            if captions is not None:
                ax.set_title(str(captions.iloc[picked[col]])[:20], fontsize=7)

        name = "noise" if cluster == NOISE else f"cluster {cluster}"
        axes[row, 0].text(-0.1, 0.5, f"{name}\n(n={size})", transform=axes[row, 0].transAxes,
                          ha="right", va="center", fontsize=9)

    fig.tight_layout()
    plt.show()
