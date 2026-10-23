"""Dimensionality-reduction plots for the gcResearch demonstration results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


RowIndexLevel = Literal["name", "modality", "domain"]
NanPolicy = Literal["ignore", "report", "row-omit", "col-omit"]
GaussianOverlapMode = Literal["blend", "dominant"]


def plot_pca(
    data: pd.DataFrame,
    *,
    standardize: bool = True,
    nan_policy: NanPolicy = "report",
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
    nan_policy : {"ignore", "report", "row-omit", "col-omit"}, default="report"
        How to handle missing, NaN, and infinite feature values.
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
    features, metadata, retained_index = _prepare_input(data, standardize, nan_policy)
    _validate_plot_parameters(point_size, alpha)

    if min(features.shape) < 2:
        raise ValueError("PCA requires at least two rows and two numeric features.")

    reducer = PCA(n_components=2)
    coordinates = reducer.fit_transform(features)
    axis_labels = tuple(
        f"PC{i + 1} ({ratio:.1%} variance)"
        for i, ratio in enumerate(reducer.explained_variance_ratio_)
    )
    embedding = _make_embedding(retained_index, coordinates, ("PC1", "PC2"))
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
    n_pca_components: int = 50,
    perplexity: float = 30.0,
    standardize: bool = True,
    nan_policy: NanPolicy = "report",
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
    nan_policy : {"ignore", "report", "row-omit", "col-omit"}, default="report"
        How to handle missing, NaN, and infinite feature values.
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
    _validate_plot_parameters(point_size, alpha)

    if isinstance(n_pca_components, bool) or not isinstance(n_pca_components, int):
        raise TypeError("pca_components must be an integer.")
    if n_pca_components < 1:
        raise ValueError("pca_components must be at least one.")

    features, metadata, retained_index = _prepare_input(data, standardize, nan_policy)
    

    if not 0 < perplexity < features.shape[0]:
        raise ValueError(
            f"perplexity must be positive and less than the number of rows "
            f"({features.shape[0]})."
        )

    pca_coordinates = _compute_pca(features, n_pca_components)
    tsne_coordinates = _compute_tsne(pca_coordinates, 2, perplexity, random_seed, **tsne_kwargs)
    embedding = _make_embedding(retained_index, tsne_coordinates, ("TSNE1", "TSNE2"))
    n_pca_components = pca_coordinates.shape[1]
    figure, axes = _plot_two_panels(
        tsne_coordinates,
        metadata,
        method_name=f"PCA ({n_pca_components} components) → t-SNE",
        axis_labels=("t-SNE 1", "t-SNE 2"),
        figsize=figsize,
        point_size=point_size,
        alpha=alpha,
    )
    return figure, axes, embedding


def compute_pca_tsne(
    data: pd.DataFrame,
    *,
    n_components: int = 2,
    n_pca_components: int = 50,
    perplexity: float = 30.0,
    standardize: bool = True,
    nan_policy: NanPolicy = "report",
    random_seed: int | None = 0,
    **tsne_kwargs):

    if isinstance(n_components, bool) or not isinstance(n_components, int):
        raise TypeError("pca_components must be an integer.")
    if n_components < 1:
        raise ValueError("pca_components must be at least one.")

    features, metadata, retained_index = _prepare_input(data, standardize, nan_policy)
    n_components = min(n_components, *features.shape)
    n_pca_components = min(n_pca_components, *features.shape)
    pca_coordinates = _compute_pca(features, n_pca_components)
    tsne_coordinates = _compute_tsne(
        pca_coordinates,
        n_components,
        perplexity,
        random_seed,
        **tsne_kwargs
    )
    embedding = _make_embedding(
        retained_index, tsne_coordinates, (f"TSNE{i+1}" for i in range(n_components)))
    return embedding

def compute_pca(
        data: pd.DataFrame,
        *,
        n_components: int,
        standardize: bool,
        nan_policy: NanPolicy = "report"
) -> pd.DataFrame:

    if isinstance(n_components, bool) or not isinstance(n_components, int):
        raise TypeError("pca_components must be an integer.")
    if n_components < 1:
        raise ValueError("pca_components must be at least one.")

    features, metadata, retained_index = _prepare_input(data, standardize, nan_policy)
    pca_coordinates = _compute_pca(features, n_components)
    embedding = _make_embedding(
        retained_index, pca_coordinates, (f"PCA{i+1}" for i in range(n_components)))
    return embedding


def _compute_pca(features: np.ndarray,
                 n_components: int,
) -> np.ndarray:

    if features.shape[0] < 2:
        raise ValueError("t-SNE requires at least two rows.")
        
    n_pca_components = min(n_components, *features.shape)
    pca_coordinates = PCA(n_components=n_pca_components).fit_transform(features)
    return pca_coordinates

def compute_tsne(
        pca_data: np.ndarray,
        *,
        n_components: int,
        perplexity: int,
        standardize: bool = True,
        random_seed: int | None = 0,
        nan_policy: NanPolicy = "report",
        **tsne_kwargs: Any
) -> pd.DataFrame:
    
    features, metadata, retained_index = _prepare_input(pca_data, standardize, nan_policy)
    n_components = min(n_components, *features.shape)
    tsne_coordinates = _compute_tsne(features, n_components, perplexity, random_seed, **tsne_kwargs)
    embedding = _make_embedding(
        retained_index, tsne_coordinates, (f"TSNE{i+1}" for i in range(n_components)))
    return embedding

def _compute_tsne(
        features: np.ndarray,
        n_components: int,
        perplexity: int,
        random_seed: int,
        **tsne_kwargs: Any
) -> np.ndarray:

    conflicting_arguments = {"n_components", "random_state"}.intersection(tsne_kwargs)

    if conflicting_arguments:
        arguments = ", ".join(sorted(conflicting_arguments))
        raise ValueError(f"The following t-SNE arguments are controlled internally: {arguments}.")
    
    method = "barnes_hut" if n_components < 4 else "exact"
    coordinates = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        method=method,
        random_state=random_seed,
        **tsne_kwargs,
    ).fit_transform(features)
    
    return coordinates

def plot_umap(
    data: pd.DataFrame,
    *,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    standardize: bool = True,
    nan_policy: NanPolicy = "report",
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
    nan_policy : {"ignore", "report", "row-omit", "col-omit"}, default="report"
        How to handle missing, NaN, and infinite feature values.
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

    features, metadata, retained_index = _prepare_input(data, standardize, nan_policy)
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
    embedding = _make_embedding(retained_index, coordinates, ("UMAP1", "UMAP2"))
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


def plot_heatmap(
    data: pd.DataFrame,
    *,
    standardize: bool = True,
    nan_policy: NanPolicy = "report",
    figsize: tuple[float, float] = (16.0, 10.0),
    cmap: str = "viridis",
    add_horiz_dividers: bool = True,
    add_vert_dividers: bool = True,
) -> tuple[Figure, Axes, pd.DataFrame]:
    """Plot a normalized result table as a grouped heatmap.

    Rows are stably ordered by the selected row-index level. Columns retain
    their existing order, with darker separators marking boundaries between
    adjacent Statistics. Repeated row labels are similarly grouped and
    separated horizontally.

    Parameters
    ----------
    data : pandas.DataFrame
        Numeric result table with a three-level row MultiIndex containing
        dataset name, data modality, and data domain, and a two-level column
        MultiIndex containing Statistic and Reducer names.
    y_axis : {"name", "modality", "domain"}, default="name"
        Row-index level used to order the table and label the heatmap y-axis.
        Matching is by these semantic positions rather than the existing
        pandas index-level names.
    standardize : bool, default=True
        If ``True``, centre each result column and scale it to unit variance,
        matching the normalization used by the dimensionality-reduction plots.
    nan_policy : {"ignore", "report", "row-omit", "col-omit"}, default="report"
        How to handle missing, NaN, and infinite values before normalization.
    figsize : tuple of float, default=(16.0, 10.0)
        Figure width and height in inches.
    cmap : str, default="viridis"
        Matplotlib colour-map name used for the heatmap.
    add_horiz_dividers : bool, default=True
        If ``True``, draw horizontal lines between distinct y-axis groups.
    add_vert_dividers : bool, default=True
        If ``True``, draw vertical lines between adjacent Statistic groups.

    Returns
    -------
    figure : matplotlib.figure.Figure
        Figure containing the heatmap and colour bar.
    axis : matplotlib.axes.Axes
        Axis containing the heatmap.
    plotted_data : pandas.DataFrame
        Ordered values displayed by the heatmap after optional normalization.

    Raises
    ------
    TypeError
        If ``data`` is not a pandas DataFrame.
    ValueError
        If either axis does not have the required MultiIndex structure, the
        selected row level is invalid, or values are empty or non-finite.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    if data.empty:
        raise ValueError("data must contain at least one row and one column.")

    index_name = data.index.names[0]

    if not isinstance(data.columns, pd.MultiIndex) or data.columns.nlevels != 2:
        raise ValueError(
            "data.columns must be a two-level MultiIndex containing Statistic "
            "and Reducer names."
        )

    handled_data = _handle_nonfinite_values(data, nan_policy)

    if handled_data.empty:
        raise ValueError("No data remain after applying nan_policy.")

    values = _to_float_array(handled_data, value_description="DataFrame values")

    if standardize:
        values = StandardScaler().fit_transform(values)

    normalized = pd.DataFrame(
        values,
        index=handled_data.index.copy(),
        columns=handled_data.columns.copy(),
    )

    # Mergesort is stable, so rows sharing the selected label retain their
    # original relative ordering while becoming one contiguous visual group.
    order = np.argsort(
        normalized.index.get_level_values(0).astype(str),
        kind="stable",
    )
    plotted_data = normalized.iloc[order]
    row_labels = plotted_data.index.get_level_values(0)
    statistic_labels = plotted_data.columns.get_level_values(0)
    reducer_labels = plotted_data.columns.get_level_values(1)

    figure, axis = plt.subplots(figsize=figsize, constrained_layout=True)
    image = axis.imshow(plotted_data.to_numpy(), aspect="auto", cmap=cmap)
    figure.colorbar(image, ax=axis, label="Standardized value" if standardize else "Value")

    # Reducer names belong to individual heatmap columns and therefore appear
    # on the ordinary lower x-axis.
    axis.set_xticks(
        np.arange(plotted_data.shape[1]),
        labels=[str(label) for label in reducer_labels],
    )
    axis.set_yticks(
        np.arange(plotted_data.shape[0]),
        labels=[str(label) for label in row_labels],
    )
    axis.tick_params(axis="x", labelrotation=90, labelsize=7)
    axis.tick_params(axis="y", labelsize=8)
    axis.set_xlabel("Reducer")
    axis.set_ylabel(index_name.capitalize())

    for label in axis.get_xticklabels():
        label.set_ha("right")
        label.set_va("top")
        label.set_rotation_mode("default")


    # The upper axis shares the heatmap's x coordinates. Place one Statistic
    # label at the midpoint of each contiguous group of Reducer columns.
    statistic_group_centres: list[float] = []
    statistic_group_names: list[str] = []
    group_start = 0

    for column_index in range(1, len(statistic_labels) + 1):
        group_finished = (
            column_index == len(statistic_labels)
            or statistic_labels[column_index] != statistic_labels[group_start]
        )

        if group_finished:
            group_end = column_index - 1
            statistic_group_centres.append((group_start + group_end) / 2)
            statistic_group_names.append(str(statistic_labels[group_start]))
            group_start = column_index

    statistic_axis = axis.secondary_xaxis("top")
    statistic_axis.set_xticks(
        statistic_group_centres,
        labels=statistic_group_names,
    )

    statistic_axis.tick_params(axis="x", length=0, labelrotation=45, labelsize=8)

    for label in statistic_axis.get_xticklabels():
        label.set_ha("left")
        label.set_va("bottom")
        label.set_rotation_mode("default")

    statistic_axis.set_xlabel("Statistic")

    # Draw boundaries only where the adjacent Statistic changes. Positioning
    # at i - 0.5 places each line between heatmap cells rather than over them.
    if add_vert_dividers:
        for column_index in range(1, len(statistic_labels)):
            if statistic_labels[column_index] != statistic_labels[column_index - 1]:
                axis.axvline(column_index - 0.5, color="black", linewidth=1.5)

    # Sorting makes equal row labels contiguous. Each label transition can
    # therefore be represented by one horizontal group boundary.
    if add_horiz_dividers:
        for row_index in range(1, len(row_labels)):
            if row_labels[row_index] != row_labels[row_index - 1]:
                axis.axhline(row_index - 0.5, color="black", linewidth=1.5)

    return figure, axis, plotted_data


def _prepare_input(
    data: pd.DataFrame,
    standardize: bool,
    nan_policy: NanPolicy = "report",
) -> tuple[np.ndarray, tuple[Sequence[Any], Sequence[Any]], pd.Index]:
    """Validate a result table and extract its features and plot metadata."""

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    if data.empty:
        raise ValueError("data must contain at least one row and one feature.")

    index_values = data.index.to_list()
    
    if isinstance(data.index, pd.MultiIndex):
        valid_index = data.index.nlevels >= 3
    else:
        valid_index = all(
            isinstance(value, tuple) and len(value) >= 3
            for value in index_values
        )

    if not valid_index:
        raise ValueError(
            "data.index must contain exactly three values per row: dataset "
            "name, data modality, and data domain."
        )

    handled_data = _handle_nonfinite_values(data, nan_policy)

    if handled_data.empty:
        raise ValueError("No data remain after applying nan_policy.")

    features = _to_float_array(
        handled_data,
        value_description="All DataFrame feature columns",
    )

    if standardize:
        features = StandardScaler().fit_transform(features)

    index_names = handled_data.index.names
    modality_idx = index_names.index("Modality")
    domain_idx = index_names.index("Domain")
    retained_index_values = handled_data.index.to_list()

    modalities = tuple(value[modality_idx] for value in retained_index_values)
    domains = tuple(value[domain_idx] for value in retained_index_values)
    return features, (modalities, domains), handled_data.index.copy()


def _handle_nonfinite_values(data: pd.DataFrame, nan_policy: NanPolicy) -> pd.DataFrame:
    """Report or omit missing and infinite values according to ``nan_policy``."""
    valid_policies = {"ignore", "report", "row-omit", "col-omit"}

    if nan_policy not in valid_policies:
        options = ", ".join(repr(option) for option in sorted(valid_policies))
        raise ValueError(f"nan_policy must be one of {options}, but got {nan_policy!r}.")
    if nan_policy == "ignore":
        return data

    values = data.to_numpy(dtype=object, copy=False)
    offences: list[tuple[int, int, str]] = []

    for row_position in range(values.shape[0]):
        for column_position in range(values.shape[1]):
            kind = _classify_nonfinite(values[row_position, column_position])

            if kind is not None:
                offences.append((row_position, column_position, kind))

    if nan_policy == "report":
        for row_position, column_position, kind in offences:
            print(
                f"Found {kind} at row {data.index[row_position]!r}, "
                f"column {data.columns[column_position]!r}."
            )

        return data

    if nan_policy == "row-omit":
        counts: dict[int, int] = {}

        for row_position, _, _ in offences:
            counts[row_position] = counts.get(row_position, 0) + 1

        for row_position, count in counts.items():
            print(
                f"Removing row {data.index[row_position]!r}: "
                f"{count} offending element(s)."
            )

        keep = np.ones(len(data.index), dtype=bool)
        keep[list(counts)] = False
        return data.iloc[keep]

    counts = {}

    for _, column_position, _ in offences:
        counts[column_position] = counts.get(column_position, 0) + 1

    for column_position, count in counts.items():
        print(
            f"Removing column {data.columns[column_position]!r}: "
            f"{count} offending element(s)."
        )

    keep = np.ones(len(data.columns), dtype=bool)
    keep[list(counts)] = False
    return data.iloc[:, keep]


def _classify_nonfinite(value: object) -> str | None:
    """Classify a scalar as pandas-missing, NaN, infinite, or finite."""
    if value is pd.NA or value is None or value is pd.NaT:
        return "n/a"

    try:
        if bool(np.isnan(value)):  # pyright: ignore[reportCallIssue]
            return "nan"
        if bool(np.isinf(value)):  # pyright: ignore[reportCallIssue]
            return "inf"
    except (TypeError, ValueError):
        return None

    return None


def _to_float_array(data: pd.DataFrame, *, value_description: str) -> np.ndarray:
    """Convert numeric frame values while normalizing pandas missing sentinels."""
    values = data.to_numpy(dtype=object, copy=True)

    # Object-backed pandas blocks may retain pd.NA even when ``na_value`` is
    # supplied to DataFrame.to_numpy(). Replace those sentinels explicitly so
    # the requested ignore/report policies can proceed with ordinary np.nan.
    missing = np.fromiter(
        (value is pd.NA or value is None or value is pd.NaT for value in values.flat),
        dtype=bool,
        count=values.size,
    ).reshape(values.shape)
    values[missing] = np.nan

    try:
        return np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{value_description} must be numeric.") from error


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


def plot_gaussian_density_blend(
    density_grids: Sequence[np.ndarray | pd.DataFrame],
    colours: Sequence[Any],
    *,
    original_points: Sequence[np.ndarray | pd.DataFrame] | None = None,
    label_name: str | None = None,
    labels: Sequence[str] | None = None,
    title: str | None = None,
    overlap_mode: GaussianOverlapMode = "blend",
    relative_density_cutoff: float = 0.0,
    alpha_power: float = 0.5,
    gaussian_alpha: float = 1.0,
    point_alpha: float = 0.8,
    point_size: float = 20.0,
    figsize: tuple[float, float] = (10.0, 8.0),
    axis: Axes | None = None,
) -> tuple[Figure, Axes, np.ndarray]:
    r"""Plot Gaussian supports using density-weighted colour blending.

    Every input grid may cover a different rectangular region, but all grids
    must use the same coordinate spacing and align to the same discretization
    lattice. They are embedded into a common union grid before their colours
    are blended. At each coordinate, the displayed RGB value is

    .. math::

        C(x, y) =
        \frac{\sum_i f_i(x, y)c_i}{\sum_i f_i(x, y)},

    where ``f_i`` is group ``i``'s density and ``c_i`` is its RGB colour.
    Total density controls opacity, preserving the distinction between strong
    and negligible support after the relative colour weights are normalized.

    Parameters
    ----------
    density_grids : Sequence of numpy.ndarray
        Gaussian grids with shape ``(n_x, n_y, 3)``. The final dimension must
        contain x coordinate, y coordinate, and density respectively.
    colours : Sequence
        One Matplotlib-compatible colour for each density grid.
    original_points : Sequence of numpy.ndarray or None, optional
        Original observations associated positionally with the Gaussian grids.
        Each array must have shape ``(n_observations, 2)``. Points are drawn
        over the density image using their group's original, unblended colour.
    labels : Sequence of str or None, optional
        Optional group names used to construct a legend.
    overlap_mode : {"blend", "dominant"}, default="blend"
        How colours are assigned where Gaussian supports overlap. ``"blend"``
        forms a density-weighted mixture of every contributing group.
        ``"dominant"`` assigns the point entirely to the group with the
        greatest density there. Ties are resolved by input order.
    relative_density_cutoff : float, default=0.0
        Per-group densities below this fraction of that group's maximum are
        treated as zero. This can suppress mathematically nonzero Gaussian
        tails. Must lie in the closed interval ``[0, 1]``.
    alpha_power : float, default=0.5
        Exponent applied to normalized total density to obtain opacity. Values
        below one make low-density support more visible; one gives linear
        opacity.
    gaussian_alpha : float, default=1.0
        Overall opacity multiplier for the blended Gaussian supports. Must lie
        in the closed interval ``[0, 1]``.
    point_alpha : float, default=0.8
        Opacity of the original observations, independently of
        ``gaussian_alpha``. Must lie in the closed interval ``[0, 1]``.
    point_size : float, default=20.0
        Marker area used to draw the original observations.
    figsize : tuple of float, default=(10.0, 8.0)
        Figure size used when ``axis`` is not supplied.
    axis : matplotlib.axes.Axes or None, optional
        Existing axis on which to draw. A new figure and axis are created when
        omitted.

    Returns
    -------
    figure : matplotlib.figure.Figure
        Figure containing the blended Gaussian image.
    axis : matplotlib.axes.Axes
        Axis containing the blended Gaussian image.
    blended_grid : numpy.ndarray
        Array with shape ``(n_x, n_y, 6)`` containing x coordinates, y
        coordinates, red, green, blue, and alpha respectively.

    Raises
    ------
    TypeError
        If a grid is not a NumPy array or plotting arguments have incompatible
        types.
    ValueError
        If collections differ in length, grids are malformed or non-finite,
        densities are negative, or coordinate lattices are incompatible.

    Notes
    -----
    Grid coordinates are cell centres. For example, grids discretized in
    multiples of ``0.05`` may have different bounds, provided their bounds
    remain aligned to that common lattice.
    """
    grids = tuple(np.asarray(x) for x in density_grids)
    colour_values = tuple(colours)
    point_groups = (None
        if original_points is None
        else tuple(np.asarray(x) for x in original_points))

    if not grids:
        raise ValueError("density_grids must contain at least one grid.")
    if len(colour_values) != len(grids):
        raise ValueError("colours must contain one colour per density grid.")
    if point_groups is not None and len(point_groups) != len(grids):
        raise ValueError(
            "original_points must contain one array per density grid.")
    if labels is not None and len(labels) != len(grids):
        raise ValueError("labels must contain one label per density grid.")
    if overlap_mode not in ("blend", "dominant"):
        raise ValueError("overlap_mode must be either 'blend' or 'dominant'.")
    if not 0 <= relative_density_cutoff <= 1:
        raise ValueError("relative_density_cutoff must lie within [0, 1].")
    if not np.isfinite(alpha_power) or alpha_power <= 0:
        raise ValueError("alpha_power must be finite and positive.")
    if not np.isfinite(gaussian_alpha) or not 0 <= gaussian_alpha <= 1:
        raise ValueError("gaussian_alpha must lie within [0, 1].")
    if not np.isfinite(point_alpha) or not 0 <= point_alpha <= 1:
        raise ValueError("point_alpha must lie within [0, 1].")
    if not np.isfinite(point_size) or point_size <= 0:
        raise ValueError("point_size must be finite and positive.")

    if point_groups is not None:
        for group_index, points in enumerate(point_groups):
            if not isinstance(points, np.ndarray):
                raise TypeError(
                    f"original_points[{group_index}] must be a NumPy array."
                )
            if points.ndim != 2 or points.shape[1] != 2:
                raise ValueError(
                    f"original_points[{group_index}] must have shape "
                    "(n_observations, 2)."
                )
            if not np.issubdtype(points.dtype, np.number):
                raise TypeError("Every original point array must be numeric.")
            if not np.isfinite(points).all():
                raise ValueError(
                    "Every original point array must contain only finite values."
                )

    x_grid, y_grid, densities = _merge_gaussian_density_grids(grids)

    if relative_density_cutoff > 0:
        group_maxima = densities.max(axis=(1, 2), keepdims=True)
        relative_densities = np.divide(
            densities,
            group_maxima,
            out=np.zeros_like(densities),
            where=group_maxima > 0,
        )
        densities = np.where(
            relative_densities >= relative_density_cutoff,
            densities,
            0.0,
        )

    try:
        rgb_colours = np.asarray(
            [to_rgb(colour) for colour in colour_values],
            dtype=float,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("Every colour must be understood by Matplotlib.") from error

    if overlap_mode == "blend":
        displayed_density = densities.sum(axis=0)
        weighted_rgb = np.einsum("gxy,gc->xyc", densities, rgb_colours)
        rgb = np.divide(
            weighted_rgb,
            displayed_density[..., np.newaxis],
            out=np.zeros_like(weighted_rgb),
            where=displayed_density[..., np.newaxis] > 0,
        )
    else:
        # Each point belongs to the group with greatest local support. Pixels
        # without any support are reset after indexing because argmax returns
        # the first group when every candidate is zero.
        dominant_groups = np.argmax(densities, axis=0)
        displayed_density = np.max(densities, axis=0)
        rgb = rgb_colours[dominant_groups]
        rgb[displayed_density == 0] = 0.0

    maximum_displayed_density = displayed_density.max()
    if maximum_displayed_density > 0:
        alpha = (
            (displayed_density / maximum_displayed_density) ** alpha_power
            * gaussian_alpha
        )
    else:
        alpha = np.zeros_like(displayed_density)

    rgba = np.concatenate((rgb, alpha[..., np.newaxis]), axis=-1)
    blended_grid = np.concatenate(
        (x_grid[..., np.newaxis], y_grid[..., np.newaxis], rgba),
        axis=-1,
    )

    if axis is None:
        figure, axis = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        figure = axis.figure

    x_step = x_grid[1, 0] - x_grid[0, 0]
    y_step = y_grid[0, 1] - y_grid[0, 0]
    extent = (
        x_grid[0, 0] - x_step / 2,
        x_grid[-1, 0] + x_step / 2,
        y_grid[0, 0] - y_step / 2,
        y_grid[0, -1] + y_step / 2,
    )

    # Matplotlib treats the first image dimension as y. The internal arrays
    # intentionally retain x on axis zero, so transpose only for rendering.
    axis.imshow(
        np.swapaxes(rgba, 0, 1),
        origin="lower",
        extent=extent,
        interpolation="nearest",
        aspect="equal",
    )

    if point_groups is not None:
        for points, colour in zip(point_groups, colour_values, strict=True):
            axis.scatter(
                points[:, 0],
                points[:, 1],
                color=colour,
                alpha=point_alpha,
                s=point_size,
                edgecolors="grey",
                linewidths=1,
                zorder=2,
            )

    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_title(title or "Density-weighted Gaussian group supports")

    if labels is not None:
        handles = [
            Patch(facecolor=colour, label=str(label))
            for colour, label in zip(colour_values, labels, strict=True)
        ]
        axis.legend(handles=handles, title=label_name or "Group")

    return figure, axis, blended_grid


def _merge_gaussian_density_grids(
    density_grids: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Embed aligned density grids into one common coordinate lattice."""
    coordinates: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    x_steps: list[float] = []
    y_steps: list[float] = []

    for grid_index, grid in enumerate(density_grids):
        if not isinstance(grid, np.ndarray):
            raise TypeError(f"density_grids[{grid_index}] must be a NumPy array.")
        if grid.ndim != 3 or grid.shape[2] != 3:
            raise ValueError(
                f"density_grids[{grid_index}] must have shape (n_x, n_y, 3)."
            )
        if grid.shape[0] < 2 or grid.shape[1] < 2:
            raise ValueError("Every density grid must have at least two x and y points.")
        if not np.issubdtype(grid.dtype, np.number):
            raise TypeError("Every density grid must contain numeric values.")
        if not np.isfinite(grid).all():
            raise ValueError("Every density grid must contain only finite values.")
        if np.any(grid[..., 2] < 0):
            raise ValueError("Gaussian density values cannot be negative.")

        x_points = np.asarray(grid[:, 0, 0], dtype=float)
        y_points = np.asarray(grid[0, :, 1], dtype=float)

        if not np.allclose(grid[..., 0], x_points[:, np.newaxis]):
            raise ValueError("Each x coordinate must be constant along grid axis one.")
        if not np.allclose(grid[..., 1], y_points[np.newaxis, :]):
            raise ValueError("Each y coordinate must be constant along grid axis zero.")

        x_differences = np.diff(x_points)
        y_differences = np.diff(y_points)

        if np.any(x_differences <= 0) or not np.allclose(
                x_differences, x_differences[0]):
            raise ValueError("Grid x coordinates must be regular and increasing.")
        if np.any(y_differences <= 0) or not np.allclose(
                y_differences, y_differences[0]):
            raise ValueError("Grid y coordinates must be regular and increasing.")

        x_steps.append(float(x_differences[0]))
        y_steps.append(float(y_differences[0]))
        coordinates.append((x_points, y_points, np.asarray(grid[..., 2], dtype=float)))

    x_step = x_steps[0]
    y_step = y_steps[0]

    if not np.allclose(x_steps, x_step) or not np.allclose(y_steps, y_step):
        raise ValueError("All Gaussian grids must use the same x and y spacing.")

    x_min = min(points[0] for points, _, _ in coordinates)
    x_max = max(points[-1] for points, _, _ in coordinates)
    y_min = min(points[0] for _, points, _ in coordinates)
    y_max = max(points[-1] for _, points, _ in coordinates)
    x_count = int(round((x_max - x_min) / x_step)) + 1
    y_count = int(round((y_max - y_min) / y_step)) + 1
    common_x = x_min + np.arange(x_count) * x_step
    common_y = y_min + np.arange(y_count) * y_step
    densities = np.zeros((len(coordinates), x_count, y_count), dtype=float)

    for group_index, (x_points, y_points, density) in enumerate(coordinates):
        x_offset_float = (x_points[0] - x_min) / x_step
        y_offset_float = (y_points[0] - y_min) / y_step
        x_offset = int(round(x_offset_float))
        y_offset = int(round(y_offset_float))

        if not np.isclose(x_offset_float, x_offset) or not np.isclose(
                y_offset_float, y_offset):
            raise ValueError(
                "Gaussian grid bounds must align to the common coordinate lattice."
            )

        x_slice = slice(x_offset, x_offset + len(x_points))
        y_slice = slice(y_offset, y_offset + len(y_points))

        if not np.allclose(common_x[x_slice], x_points) or not np.allclose(
                common_y[y_slice], y_points):
            raise ValueError(
                "Gaussian grid coordinates do not align after discretization."
            )

        densities[group_index, x_slice, y_slice] = density

    x_grid, y_grid = np.meshgrid(common_x, common_y, indexing="ij")
    return x_grid, y_grid, densities


def plot_nearest_group_regions(
    assignment_grid: np.ndarray,
    group_identifiers: Mapping[str, int],
    colours: Sequence[Any],
    *,
    original_points: Sequence[np.ndarray | pd.DataFrame] | None = None,
    alpha: float = 1.0,
    point_alpha: float = 0.8,
    point_size: float = 20.0,
    label_name: str = "Group",
    title: str = "Nearest-point group regions",
    figsize: tuple[float, float] = (10.0, 8.0),
    axis: Axes | None = None,
) -> tuple[Figure, Axes, np.ndarray]:
    """Plot nearest-point group regions using categorical colours.

    Parameters
    ----------
    assignment_grid : numpy.ndarray
        Grid with shape ``(n_x, n_y, 3)`` as returned by
        :func:`clustering.nearest_group_grid`. Its final dimension contains x
        coordinates, y coordinates, and natural-number group identifiers.
    group_identifiers : Mapping[str, int]
        Mapping from group names to the identifiers stored in
        ``assignment_grid``.
    colours : Sequence
        One Matplotlib-compatible colour for each mapping entry, in mapping
        iteration order.
    original_points : Sequence of numpy.ndarray or None, optional
        Original observations associated positionally with the mapping entries.
        Each array must have shape ``(n_observations, 2)``. Points use their
        group's unblended colour with a grey edge of line width one.
    alpha : float, default=1.0
        Opacity of the coloured assignment regions in the closed interval
        ``[0, 1]``.
    point_alpha : float, default=0.8
        Opacity of the original observation markers in the closed interval
        ``[0, 1]``.
    point_size : float, default=20.0
        Marker area used to draw the original observations.
    label_name : str, default="Group"
        Title displayed above the group legend.
    title : str, default="Nearest-point group regions"
        Plot title.
    figsize : tuple of float, default=(10.0, 8.0)
        Figure size used when ``axis`` is not supplied.
    axis : matplotlib.axes.Axes or None, optional
        Existing axis on which to draw. A new figure and axis are created when
        omitted.

    Returns
    -------
    figure : matplotlib.figure.Figure
        Figure containing the categorical region image and legend.
    axis : matplotlib.axes.Axes
        Axis containing the categorical region image.
    coloured_grid : numpy.ndarray
        Array with shape ``(n_x, n_y, 5)`` containing x coordinates, y
        coordinates, red, green, and blue respectively.

    Raises
    ------
    TypeError
        If the grid, mapping, identifiers, or colours have incompatible types.
    ValueError
        If the grid is malformed, identifiers are invalid or absent from the
        mapping, colours cannot be interpreted, or coordinates are not a
        regular increasing mesh.
    """
    if not isinstance(assignment_grid, np.ndarray):
        raise TypeError("assignment_grid must be a NumPy array.")
    if assignment_grid.ndim != 3 or assignment_grid.shape[2] != 3:
        raise ValueError("assignment_grid must have shape (n_x, n_y, 3).")
    if assignment_grid.shape[0] < 2 or assignment_grid.shape[1] < 2:
        raise ValueError("assignment_grid must contain at least two x and y points.")
    if not np.issubdtype(assignment_grid.dtype, np.number):
        raise TypeError("assignment_grid must contain numeric values.")
    if not np.isfinite(assignment_grid).all():
        raise ValueError("assignment_grid must contain only finite values.")
    if not isinstance(group_identifiers, Mapping):
        raise TypeError("group_identifiers must be a mapping.")
    if not group_identifiers:
        raise ValueError("group_identifiers must contain at least one group.")

    group_items = tuple(group_identifiers.items())
    colour_values = tuple(colours)
    point_groups = None if original_points is None else tuple(
        np.asarray(x) for x in original_points)

    if len(colour_values) != len(group_items):
        raise ValueError("colours must contain one colour per group.")
    if point_groups is not None and len(point_groups) != len(group_items):
        raise ValueError("original_points must contain one array per group.")
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must lie within [0, 1].")
    if not np.isfinite(point_alpha) or not 0 <= point_alpha <= 1:
        raise ValueError("point_alpha must lie within [0, 1].")
    if not np.isfinite(point_size) or point_size <= 0:
        raise ValueError("point_size must be finite and positive.")
    if not isinstance(label_name, str):
        raise TypeError("label_name must be a string.")
    if not isinstance(title, str):
        raise TypeError("title must be a string.")

    if point_groups is not None:
        for group_index, points in enumerate(point_groups):
            if not isinstance(points, np.ndarray):
                raise TypeError(
                    f"original_points[{group_index}] must be a NumPy array."
                )
            if points.ndim != 2 or points.shape[1] != 2:
                raise ValueError(
                    f"original_points[{group_index}] must have shape "
                    "(n_observations, 2)."
                )
            if not np.issubdtype(points.dtype, np.number):
                raise TypeError("Every original point array must be numeric.")
            if not np.isfinite(points).all():
                raise ValueError(
                    "Every original point array must contain only finite values."
                )

    identifier_values: list[int] = []

    for group_name, identifier in group_items:
        if not isinstance(group_name, str):
            raise TypeError("Every group name must be a string.")
        if isinstance(identifier, bool) or not isinstance(
            identifier,
            (int, np.integer),
        ):
            raise TypeError("Every group identifier must be an integer.")
        if identifier < 1:
            raise ValueError("Every group identifier must be a natural number.")

        identifier_values.append(int(identifier))

    if len(set(identifier_values)) != len(identifier_values):
        raise ValueError("Group identifiers must be unique.")

    try:
        rgb_colours = np.asarray(
            [to_rgb(colour) for colour in colour_values],
            dtype=float,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("Every colour must be understood by Matplotlib.") from error

    x_grid = np.asarray(assignment_grid[..., 0], dtype=float)
    y_grid = np.asarray(assignment_grid[..., 1], dtype=float)
    raw_assignments = np.asarray(assignment_grid[..., 2], dtype=float)
    assignments = np.rint(raw_assignments).astype(np.int64)

    if not np.allclose(raw_assignments, assignments):
        raise ValueError("Grid assignments must contain integer-valued identifiers.")

    unknown_identifiers = set(np.unique(assignments)) - set(identifier_values)
    if unknown_identifiers:
        raise ValueError(
            "assignment_grid contains identifiers absent from group_identifiers: "
            f"{sorted(unknown_identifiers)}."
        )

    x_points = x_grid[:, 0]
    y_points = y_grid[0, :]

    if not np.allclose(x_grid, x_points[:, np.newaxis]):
        raise ValueError("Each x coordinate must be constant along grid axis one.")
    if not np.allclose(y_grid, y_points[np.newaxis, :]):
        raise ValueError("Each y coordinate must be constant along grid axis zero.")

    x_differences = np.diff(x_points)
    y_differences = np.diff(y_points)

    if np.any(x_differences <= 0) or not np.allclose(
        x_differences,
        x_differences[0],
    ):
        raise ValueError("Grid x coordinates must be regular and increasing.")
    if np.any(y_differences <= 0) or not np.allclose(
        y_differences,
        y_differences[0],
    ):
        raise ValueError("Grid y coordinates must be regular and increasing.")

    identifier_colours = dict(zip(identifier_values, rgb_colours, strict=True))
    rgb = np.empty((*assignments.shape, 3), dtype=float)

    for identifier, colour in identifier_colours.items():
        rgb[assignments == identifier] = colour

    coloured_grid = np.concatenate(
        (x_grid[..., np.newaxis], y_grid[..., np.newaxis], rgb),
        axis=-1,
    )

    if axis is None:
        figure, axis = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        figure = axis.figure

    extent = (
        x_points[0] - x_differences[0] / 2,
        x_points[-1] + x_differences[0] / 2,
        y_points[0] - y_differences[0] / 2,
        y_points[-1] + y_differences[0] / 2,
    )
    axis.imshow(
        np.swapaxes(rgb, 0, 1),
        origin="lower",
        extent=extent,
        interpolation="nearest",
        aspect="equal",
        alpha=alpha,
    )

    if point_groups is not None:
        for points, colour in zip(point_groups, colour_values, strict=True):
            axis.scatter(
                points[:, 0],
                points[:, 1],
                color=colour,
                alpha=point_alpha,
                s=point_size,
                edgecolors="grey",
                linewidths=1,
                zorder=2,
            )

    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_title(title)
    handles = [
        Patch(facecolor=colour, alpha=alpha, label=group_name)
        for (group_name, _), colour in zip(
            group_items,
            colour_values,
            strict=True,
        )
    ]
    axis.legend(handles=handles, title=label_name, loc="upper right")
    return figure, axis, coloured_grid


__all__ = [
    "plot_gaussian_density_blend",
    "plot_heatmap",
    "plot_nearest_group_regions",
    "plot_pca",
    "plot_pca_tsne",
    "plot_umap",
]
