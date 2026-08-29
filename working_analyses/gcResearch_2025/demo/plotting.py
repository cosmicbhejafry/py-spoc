"""Dimensionality-reduction plots for the gcResearch demonstration results."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


RowIndexLevel = Literal["name", "modality", "domain"]
NanPolicy = Literal["ignore", "report", "row-omit", "col-omit"]


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
    pca_components: int = 50,
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
    features, metadata, retained_index = _prepare_input(data, standardize, nan_policy)
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
    embedding = _make_embedding(retained_index, coordinates, ("TSNE1", "TSNE2"))
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
    y_axis: RowIndexLevel = "name",
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
    if not isinstance(data.index, pd.MultiIndex) or data.index.nlevels != 3:
        raise ValueError(
            "data.index must be a three-level MultiIndex containing dataset "
            "name, data modality, and data domain."
        )
    if not isinstance(data.columns, pd.MultiIndex) or data.columns.nlevels != 2:
        raise ValueError(
            "data.columns must be a two-level MultiIndex containing Statistic "
            "and Reducer names."
        )

    level_positions = {"name": 0, "modality": 1, "domain": 2}

    if y_axis not in level_positions:
        options = ", ".join(repr(option) for option in level_positions)
        raise ValueError(f"y_axis must be one of {options}, but got {y_axis!r}.")

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
    selected_level = level_positions[y_axis]

    # Mergesort is stable, so rows sharing the selected label retain their
    # original relative ordering while becoming one contiguous visual group.
    order = np.argsort(
        normalized.index.get_level_values(selected_level).astype(str),
        kind="stable",
    )
    plotted_data = normalized.iloc[order]
    row_labels = plotted_data.index.get_level_values(selected_level)
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
    axis.set_ylabel(y_axis.capitalize())

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

    handled_data = _handle_nonfinite_values(data, nan_policy)

    if handled_data.empty:
        raise ValueError("No data remain after applying nan_policy.")

    features = _to_float_array(
        handled_data,
        value_description="All DataFrame feature columns",
    )

    if standardize:
        features = StandardScaler().fit_transform(features)

    retained_index_values = handled_data.index.to_list()
    modalities = tuple(value[1] for value in retained_index_values)
    domains = tuple(value[2] for value in retained_index_values)
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


__all__ = ["plot_heatmap", "plot_pca", "plot_pca_tsne", "plot_umap"]
