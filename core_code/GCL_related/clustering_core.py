"""
Core HDBSCAN clustering used after GCL embedding extraction.

Only the preprocessing and final clustering steps are retained here.
Sensitivity grids, plotting, report generation, and experiment orchestration
are intentionally excluded.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from hdbscan import HDBSCAN
from sklearn.preprocessing import StandardScaler, normalize


DEFAULT_HDBSCAN = {
    "min_cluster_size": 300,
    "min_samples": 5,
    "metric": "euclidean",
    "cluster_selection_epsilon": 0.05,
    "cluster_selection_method": "eom",
}


def preprocess_embeddings(
    embeddings: np.ndarray,
) -> np.ndarray:
    """
    Apply the preprocessing used before formal HDBSCAN clustering:

        StandardScaler -> row-wise L2 normalization.
    """
    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    if (
        embeddings.ndim != 2
        or len(embeddings) < 2
        or not np.isfinite(embeddings).all()
    ):
        raise ValueError(
            "embeddings must be a finite 2-D array "
            "with at least two samples."
        )

    scaled = StandardScaler().fit_transform(
        embeddings
    )

    return normalize(
        scaled,
        norm="l2",
        axis=1,
    ).astype(
        np.float32,
        copy=False,
    )


def cluster_embeddings(
    embeddings: np.ndarray,
    *,
    min_cluster_size: int = 300,
    min_samples: int = 5,
    metric: str = "euclidean",
    cluster_selection_epsilon: float = 0.05,
    cluster_selection_method: str = "eom",
) -> Tuple[
    np.ndarray,
    HDBSCAN,
    np.ndarray,
    Dict[str, Any],
]:
    """
    Cluster GCL graph embeddings using the formal Stage-1 HDBSCAN procedure.

    Returns
    -------
    labels:
        HDBSCAN labels. Noise samples are labeled -1.
    clusterer:
        Fitted HDBSCAN estimator.
    processed_embeddings:
        Standardized and L2-normalized embeddings used for clustering.
    summary:
        Basic cluster-count and noise statistics.
    """
    processed_embeddings = preprocess_embeddings(
        embeddings
    )

    if min_cluster_size > len(processed_embeddings):
        raise ValueError(
            "min_cluster_size cannot exceed "
            "the number of samples."
        )

    clusterer = HDBSCAN(
        min_cluster_size=int(min_cluster_size),
        min_samples=int(min_samples),
        metric=metric,
        cluster_selection_epsilon=float(
            cluster_selection_epsilon
        ),
        cluster_selection_method=(
            cluster_selection_method
        ),
        prediction_data=True,
        gen_min_span_tree=True,
    )

    labels = clusterer.fit_predict(
        processed_embeddings
    )

    n_clusters = len(
        set(labels) - {-1}
    )
    noise_count = int(
        np.sum(labels == -1)
    )

    summary = {
        "n_samples": int(len(labels)),
        "n_clusters": int(n_clusters),
        "noise_count": noise_count,
        "noise_ratio": float(
            noise_count / len(labels)
        ),
        "min_cluster_size": int(
            min_cluster_size
        ),
        "min_samples": int(min_samples),
        "metric": metric,
        "cluster_selection_epsilon": float(
            cluster_selection_epsilon
        ),
        "cluster_selection_method": (
            cluster_selection_method
        ),
    }

    return (
        labels,
        clusterer,
        processed_embeddings,
        summary,
    )
