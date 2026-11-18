from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm.auto import tqdm


def kmeans_sweep(
    features: np.ndarray,
    ks: Iterable[int],
    seed: int = 42,
    n_init: int = 10,
    silhouette_sample_size: int | None = 10_000,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Fit K-means for each k and record inertia and silhouette.

    Use it to choose `n_clusters`: look for the elbow in inertia and the peak
    in silhouette. Pass features in the space you will cluster in (e.g. after
    StandardScaler / PCA), otherwise the numbers describe a different space.

    Args:
        features (np.ndarray): Points to cluster, shape (n, d).
        ks (Iterable[int]): Values of k to try. Each must be at least 2,
            because silhouette is undefined for a single cluster.
        seed (int): Seed for K-means and for the silhouette subsample.
        n_init (int): K-means restarts per k. Lower it to speed up large sweeps.
        silhouette_sample_size (int | None): Silhouette is O(n^2), so it is
            computed on a random subsample of this many points. None = all
            points (slow for large n).
        show_progress (bool): Show a tqdm progress bar over k.

    Returns:
        pd.DataFrame: One row per k, columns 'k', 'inertia' (lower is tighter,
            always decreases with k) and 'silhouette' (higher is better, in [-1, 1]).
    """
    ks = list(ks)
    if any(k < 2 for k in ks):
        raise ValueError("every k must be at least 2")

    use_sample = silhouette_sample_size is not None and len(features) > silhouette_sample_size

    rows = []
    for k in tqdm(ks, desc="K-means sweep", disable=not show_progress):
        model = KMeans(n_clusters=k, random_state=seed, n_init=n_init).fit(features)
        silhouette = silhouette_score(
            features,
            model.labels_,
            sample_size=silhouette_sample_size if use_sample else None,
            random_state=seed,
        )
        rows.append({"k": k, "inertia": float(model.inertia_), "silhouette": float(silhouette)})

    return pd.DataFrame(rows)
