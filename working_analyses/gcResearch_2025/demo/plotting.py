"""Dimensionality-reduction plots for the gcResearch demonstration results."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def plot_pca(
    data: pd.DataFrame,
    *,
    standardize: bool = True,
    figsize: tuple[float, float] = (14.0, 6.0),
    point_size: float = 36.0,
    alpha: float = 0.8,
) -> tuple[Figure, np.ndarray, pd.DataFrame]:
    """Plot a two-dimensional PCA embedding by modality and domain.

    Parameters
    ----------
    data : pandas.DataFrame
        Numeric feature matrix with one dataset per row. Its index must contain
        three values per row: dataset name, data modality, and data domain.
    standardize : bool, default=True
        If ``True``, centre each feature and scale it to unit variance before
        dimensionality reduction.
    figsize : tuple of float, default=(14.0, 6.0)
        Width and height of the complete two-panel figure in inches.
    point_size : float, default=36.0
        Marker area passed to :meth:`matplotlib.axes.Axes.scatter`.
    alpha : float, default=0.8
        Marker opacity in the closed interval ``[0, 1]``.

    Returns
    -------
    figure : matplotlib.figure.Figure
        Figure containing the modality and domain panels.
    axes : numpy.ndarray
        One-dimensional array containing the two subplot axes.
    embedding : pandas.DataFrame
        Two PCA coordinates per input row, retaining the original index.

    Raises
    ------
    TypeError
        If ``data`` is not a pandas DataFrame.
    ValueError
        If the index, feature values, or plotting parameters are invalid, or
        fewer than two PCA dimensions can be calculated.
    """
    features, metadata = _prepare_input(data, standardize)
    _validate_plot_parameters(point_size, alpha)

    if min(features.shape) < 2:
        raise ValueError("PCA requires at least two rows and two numeric features.")

    reducer = PCA(n_components=2)
    coordinates = reducer.fit_transform(features)
    axis_labels = tuple(
        f"PC{i + 1} ({ratio:.1%} variance)"
        for i, ratio in enumerate(reducer.explained_variance_ratio_)
    )
    embedding = _make_embedding(data.index, coordinates, ("PC1", "PC2"))
    figure, axes = _plot_two_panels(
        coordinates,
        metadata,
        method_name="PCA",
        axis_labels=axis_labels,
        figsize=figsize,
        point_size=point_size,
        alpha=alpha,
    )
    return figure, axes, embedding


def plot_pca_tsne(
    data: pd.DataFrame,
    *,
    pca_components: int = 50,
    perplexity: float = 30.0,
    standardize: bool = True,
    random_seed: int | None = 0,
    figsize: tuple[float, float] = (14.0, 6.0),
    point_size: float = 36.0,
    alpha: float = 0.8,
    **tsne_kwargs: Any,
) -> tuple[Figure, np.ndarray, pd.DataFrame]:
    """Plot a PCA-preprocessed t-SNE embedding by modality and domain.

    PCA first reduces the feature matrix to at most ``pca_components``
    dimensions. t-SNE then constructs the final nonlinear two-dimensional
    embedding from those PCA scores.

    Parameters
    ----------
    data : pandas.DataFrame
        Numeric feature matrix indexed by dataset name, modality, and domain.
    pca_components : int, default=50
        Requested PCA dimensionality. It is capped at the smaller of the
        number of rows and features.
    perplexity : float, default=30.0
        Effective neighbourhood size used by t-SNE. Must be positive and less
        than the number of rows.
    standardize : bool, default=True
        If ``True``, standardize features before PCA.
    random_seed : int or None, default=0
        Random state supplied to t-SNE. Use ``None`` for nondeterministic runs.
    figsize : tuple of float, default=(14.0, 6.0)
        Complete figure size in inches.
    point_size : float, default=36.0
        Scatter-marker area.
    alpha : float, default=0.8
        Marker opacity in the closed interval ``[0, 1]``.
    **tsne_kwargs
        Additional keyword arguments forwarded to
        :class:`sklearn.manifold.TSNE`. The dimensionality and random state are
        controlled by this function.

    Returns
    -------
    figure : matplotlib.figure.Figure
        Figure containing the modality and domain panels.
    axes : numpy.ndarray
        One-dimensional array containing the two subplot axes.
    embedding : pandas.DataFrame
        Two t-SNE coordinates per input row, retaining the original index.

    Raises
    ------
    TypeError
        If ``data`` is not a pandas DataFrame or ``pca_components`` is not an
        integer.
    ValueError
        If input data or reduction parameters are invalid.
    """
    features, metadata = _prepare_input(data, standardize)
    _validate_plot_parameters(point_size, alpha)

    if isinstance(pca_components, bool) or not isinstance(pca_components, int):
        raise TypeError("pca_components must be an integer.")
    if pca_components < 1:
        raise ValueError("pca_components must be at least one.")
    if features.shape[0] < 2:
        raise ValueError("t-SNE requires at least two rows.")
    if not 0 < perplexity < features.shape[0]:
        raise ValueError(
            f"perplexity must be positive and less than the number of rows "
            f"({features.shape[0]})."
        )

    n_pca_components = min(pca_components, *features.shape)
    pca_coordinates = PCA(n_components=n_pca_components).fit_transform(features)

    conflicting_arguments = {"n_components", "random_state"}.intersection(tsne_kwargs)
    if conflicting_arguments:
        arguments = ", ".join(sorted(conflicting_arguments))
        raise ValueError(f"The following t-SNE arguments are controlled internally: {arguments}.")

    coordinates = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_seed,
        **tsne_kwargs,
    ).fit_transform(pca_coordinates)
    embedding = _make_embedding(data.index, coordinates, ("TSNE1", "TSNE2"))
    figure, axes = _plot_two_panels(
        coordinates,
        metadata,
        method_name=f"PCA ({n_pca_components} components) → t-SNE",
        axis_labels=("t-SNE 1", "t-SNE 2"),
        figsize=figsize,
        point_size=point_size,
        alpha=alpha,
    )
    return figure, axes, embedding


def plot_umap(
    data: pd.DataFrame,
    *,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    standardize: bool = True,
    random_seed: int | None = 0,
    figsize: tuple[float, float] = (14.0, 6.0),
    point_size: float = 36.0,
    alpha: float = 0.8,
    **umap_kwargs: Any,
) -> tuple[Figure, np.ndarray, pd.DataFrame]:
    """Plot a two-dimensional UMAP embedding by modality and domain.

    Parameters
    ----------
    data : pandas.DataFrame
        Numeric feature matrix indexed by dataset name, modality, and domain.
    n_neighbors : int, default=15
        Size of the local neighbourhood used to construct the UMAP graph.
        Must be smaller than the number of rows.
    min_dist : float, default=0.1
        Minimum distance allowed between embedded points. Must be nonnegative.
    metric : str, default="euclidean"
        Distance metric understood by :class:`umap.UMAP`.
    standardize : bool, default=True
        If ``True``, standardize features before UMAP.
    random_seed : int or None, default=0
        Random state supplied to UMAP.
    figsize : tuple of float, default=(14.0, 6.0)
        Complete figure size in inches.
    point_size : float, default=36.0
        Scatter-marker area.
    alpha : float, default=0.8
        Marker opacity in the closed interval ``[0, 1]``.
    **umap_kwargs
        Additional keyword arguments forwarded to :class:`umap.UMAP`.

    Returns
    -------
    figure : matplotlib.figure.Figure
        Figure containing the modality and domain panels.
    axes : numpy.ndarray
        One-dimensional array containing the two subplot axes.
    embedding : pandas.DataFrame
        Two UMAP coordinates per input row, retaining the original index.

    Raises
    ------
    ImportError
        If the optional ``umap-learn`` package is unavailable.
    TypeError
        If ``data`` is not a pandas DataFrame or ``n_neighbors`` is not an
        integer.
    ValueError
        If input data or reduction parameters are invalid.
    """
    try:
        from umap import UMAP
    except ImportError as error:
        raise ImportError(
            "plot_umap requires the optional 'umap-learn' package. Install it "
            "with 'python -m pip install umap-learn'."
        ) from error

    features, metadata = _prepare_input(data, standardize)
    _validate_plot_parameters(point_size, alpha)

    if isinstance(n_neighbors, bool) or not isinstance(n_neighbors, int):
        raise TypeError("n_neighbors must be an integer.")
    if features.shape[0] < 3:
        raise ValueError("UMAP requires at least three rows.")
    if not 2 <= n_neighbors < features.shape[0]:
        raise ValueError(
            f"n_neighbors must be at least two and less than the number of "
            f"rows ({features.shape[0]})."
        )
    if min_dist < 0:
        raise ValueError("min_dist must be nonnegative.")

    controlled_arguments = {
        "n_components",
        "n_neighbors",
        "min_dist",
        "metric",
        "random_state",
    }
    conflicting_arguments = controlled_arguments.intersection(umap_kwargs)
    if conflicting_arguments:
        arguments = ", ".join(sorted(conflicting_arguments))
        raise ValueError(f"The following UMAP arguments are controlled internally: {arguments}.")

    coordinates = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_seed,
        **umap_kwargs,
    ).fit_transform(features)
    embedding = _make_embedding(data.index, coordinates, ("UMAP1", "UMAP2"))
    figure, axes = _plot_two_panels(
        coordinates,
        metadata,
        method_name="UMAP",
        axis_labels=("UMAP 1", "UMAP 2"),
        figsize=figsize,
        point_size=point_size,
        alpha=alpha,
    )
    return figure, axes, embedding


def _prepare_input(
    data: pd.DataFrame,
    standardize: bool,
) -> tuple[np.ndarray, tuple[Sequence[Any], Sequence[Any]]]:
    """Validate a result table and extract its features and plot metadata."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    if data.empty:
        raise ValueError("data must contain at least one row and one feature.")

    index_values = data.index.to_list()
    if isinstance(data.index, pd.MultiIndex):
        valid_index = data.index.nlevels == 3
    else:
        valid_index = all(
            isinstance(value, tuple) and len(value) == 3
            for value in index_values
        )

    if not valid_index:
        raise ValueError(
            "data.index must contain exactly three values per row: dataset "
            "name, data modality, and data domain."
        )

    try:
        features = data.to_numpy(dtype=float, copy=True)
    except (TypeError, ValueError) as error:
        raise ValueError("All DataFrame feature columns must be numeric.") from error

    if not np.isfinite(features).all():
        raise ValueError("DataFrame features must not contain NaN or infinite values.")

    if standardize:
        features = StandardScaler().fit_transform(features)

    modalities = tuple(value[1] for value in index_values)
    domains = tuple(value[2] for value in index_values)
    return features, (modalities, domains)


def _validate_plot_parameters(point_size: float, alpha: float) -> None:
    """Validate parameters shared by all scatter plots."""
    if point_size <= 0:
        raise ValueError("point_size must be positive.")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must lie in the closed interval [0, 1].")


def _make_embedding(
    index: pd.Index,
    coordinates: np.ndarray,
    columns: tuple[str, str],
) -> pd.DataFrame:
    """Create an indexed DataFrame from two-dimensional coordinates."""
    return pd.DataFrame(coordinates, index=index.copy(), columns=columns)


def _plot_two_panels(
    coordinates: np.ndarray,
    metadata: tuple[Sequence[Any], Sequence[Any]],
    *,
    method_name: str,
    axis_labels: tuple[str, str],
    figsize: tuple[float, float],
    point_size: float,
    alpha: float,
) -> tuple[Figure, np.ndarray]:
    """Plot one embedding coloured independently by modality and domain."""
    figure, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    panel_specs = (
        (metadata[0], "Data modality"),
        (metadata[1], "Data domain"),
    )

    for axis, (labels, grouping_name) in zip(axes, panel_specs, strict=True):
        categories = list(dict.fromkeys(labels))
        colour_map = plt.get_cmap("tab20", max(len(categories), 1))

        for category_index, category in enumerate(categories):
            mask = np.fromiter(
                (label == category for label in labels),
                dtype=bool,
                count=len(labels),
            )
            axis.scatter(
                coordinates[mask, 0],
                coordinates[mask, 1],
                color=colour_map(category_index),
                label=str(category),
                s=point_size,
                alpha=alpha,
                edgecolors="none",
            )

        axis.set_title(f"{method_name} by {grouping_name.lower()}")
        axis.set_xlabel(axis_labels[0])
        axis.set_ylabel(axis_labels[1])
        axis.grid(alpha=0.2)
        axis.legend(title=grouping_name, bbox_to_anchor=(1.02, 1), loc="upper left")

    return figure, axes


__all__ = ["plot_pca", "plot_pca_tsne", "plot_umap"]
