"""Spatial group-assignment utilities for the gcResearch ESA analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from collections.abc import Mapping
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree # pyright: ignore[reportAttributeAccessIssue]
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier


def nearest_group_grid(
    groups: Mapping[str, np.ndarray | pd.DataFrame],
    p_distance: float = 2.0,
    grid_step: float = 0.05,
    grid_buffer: float = 0.5,
) -> tuple[np.ndarray, dict[str, int]]:
    """Assign a discretized two-dimensional grid to nearest point groups.

    For every grid coordinate, the function finds the closest observed point
    under the requested :math:`L_p` distance. The coordinate is assigned to
    the group containing that point. This produces a nearest-neighbour spatial
    tessellation in which groups may own multiple disconnected regions when
    their observations are spatially separated.

    Grid bounds cover the complete collection of observations and are rounded
    outwards to integer multiples of ``grid_step``. Consequently, grids
    produced by separate calls use the same absolute discretization lattice
    when supplied with the same step.

    Parameters
    ----------
    groups : Mapping[str, numpy.ndarray or pandas.DataFrame]
        Mapping from group name to finite numeric points with shape
        ``(n_points, 2)``. Each group must contain at least one point. Mapping
        insertion order determines group identifiers and resolves exact
        equidistance ties. DataFrames are converted to NumPy arrays.
    p_distance : float, default=2.0
        Minkowski distance order. Must be at least one or positive infinity.
        For example, one selects Manhattan distance, two selects Euclidean
        distance, and infinity selects Chebyshev distance.
    grid_step : float, default=0.05
        Finite positive spacing between adjacent x and y grid coordinates.
        Every coordinate is an integer multiple of this value.

    Returns
    -------
    assignment_grid : numpy.ndarray
        Array with shape ``(n_x, n_y, 3)``. ``[..., 0]`` contains x
        coordinates, ``[..., 1]`` contains y coordinates, and ``[..., 2]``
        contains the 1-based natural-number identifier of the nearest group.
        Because a standard NumPy array has one common dtype, identifiers are
        stored as exactly integral floating-point values alongside the grid
        coordinates.
    group_identifiers : dict[str, int]
        Mapping from each input group name to its 1-based identifier.

    Raises
    ------
    TypeError
        If the mapping, names, arrays, or numeric parameters have incompatible
        types.
    ValueError
        If no groups are supplied, an array is empty or malformed, values are
        non-finite, or an execution parameter lies outside its valid range.

    Notes
    -----
    One spatial index is constructed per group. Each index returns only its
    nearest distance for every grid coordinate, avoiding allocation of a
    potentially very large ``n_grid_points`` by ``n_observed_points`` distance
    matrix.
    """
    p_value = _validate_p_distance(p_distance)
    step = _validate_grid_step(grid_step)
    validated_groups = _validate_groups(groups)

    all_points = np.concatenate(
        [points for _, points in validated_groups],
        axis=0,
    )
    x_points = _make_discretized_axis(
        float(all_points[:, 0].min()) - grid_buffer,
        float(all_points[:, 0].max()) + grid_buffer,
        step,
    )
    y_points = _make_discretized_axis(
        float(all_points[:, 1].min()) - grid_buffer,
        float(all_points[:, 1].max()) + grid_buffer,
        step,
    )
    x_grid, y_grid = np.meshgrid(x_points, y_points, indexing="ij")
    grid_coordinates = np.column_stack((x_grid.ravel(), y_grid.ravel()))
    nearest_distances = np.full(grid_coordinates.shape[0], np.inf)
    assignments = np.zeros(grid_coordinates.shape[0], dtype=np.int64)
    group_identifiers = {
        group_name: group_index
        for group_index, (group_name, _) in enumerate(validated_groups, start=1)
    }

    for group_name, points in validated_groups:
        group_identifier = group_identifiers[group_name]
        candidate_distances, _ = cKDTree(points).query(
            grid_coordinates,
            k=1,
            p=p_value,
        )

        # Strict comparison deliberately retains the earlier group when two
        # groups have points at exactly the same minimum distance.
        closer = candidate_distances < nearest_distances
        nearest_distances[closer] = candidate_distances[closer]
        assignments[closer] = group_identifier

    assignment_grid = np.stack(
        (
            x_grid,
            y_grid,
            assignments.reshape(x_grid.shape),
        ),
        axis=-1,
    )
    return assignment_grid, group_identifiers


def knn_group_grid(
    groups: Mapping[str, np.ndarray | pd.DataFrame],
    grid_step: float = 0.05,
    n_neighbors: int = 5,
    grid_buffer: float = 0.5
) -> tuple[np.ndarray, dict[str, int]]:
    """Classify a discretized two-dimensional grid using K-nearest neighbours.

    Input group membership supplies the supervised target class for every
    observed point. Group names are converted to 1-based identifiers in
    mapping insertion order, a K-nearest-neighbours classifier is fitted to
    the pooled observations, and every coordinate on the resolved grid is
    classified by majority vote among its nearest observations.

    Parameters
    ----------
    groups : Mapping[str, numpy.ndarray or pandas.DataFrame]
        Mapping from group name to finite numeric points with shape
        ``(n_points, 2)``. Each group must contain at least one observation.
        DataFrames are converted to NumPy arrays without changing the caller's
        objects.
    grid_step : float, default=0.05
        Finite positive spacing between adjacent x and y coordinates. Bounds
        cover all observations and are rounded outwards to integer multiples
        of this value.
    n_neighbors : int, default=5
        Number of neighbours participating in the uniform majority vote. Must
        be positive and no larger than the total number of observations.

    Returns
    -------
    assignment_grid : numpy.ndarray
        Array with shape ``(n_x, n_y, 3)``. ``[..., 0]`` contains x
        coordinates, ``[..., 1]`` contains y coordinates, and ``[..., 2]``
        contains the predicted 1-based class identifier.
    group_identifiers : dict[str, int]
        Mapping from each input group name to its 1-based class identifier.

    Raises
    ------
    TypeError
        If the mapping, group data, ``grid_step``, or ``n_neighbors`` has an
        incompatible type.
    ValueError
        If group data are empty, malformed, or non-finite, or an execution
        parameter lies outside its valid range.

    Notes
    -----
    Euclidean distance and uniform neighbour voting are used. When classes
    receive equal votes, scikit-learn selects the lower class identifier;
    therefore ties favour the group occurring earlier in mapping order.
    """
    step = _validate_grid_step(grid_step)
    validated_groups = _validate_groups(groups)

    if isinstance(n_neighbors, bool) or not isinstance(
        n_neighbors,
        (int, np.integer),
    ):
        raise TypeError("n_neighbors must be an integer.")

    neighbour_count = int(n_neighbors)

    if neighbour_count < 1:
        raise ValueError("n_neighbors must be at least one.")

    group_identifiers = {
        group_name: group_index
        for group_index, (group_name, _) in enumerate(validated_groups, start=1)
    }
    all_points = np.concatenate(
        [points for _, points in validated_groups],
        axis=0,
    )

    if neighbour_count > all_points.shape[0]:
        raise ValueError(
            "n_neighbors cannot exceed the total number of observations."
        )

    target_classes = np.concatenate(
        [
            np.full(points.shape[0], group_identifiers[group_name], dtype=np.int64)
            for group_name, points in validated_groups
        ]
    )
    x_points = _make_discretized_axis(
        float(all_points[:, 0].min()) - grid_buffer,
        float(all_points[:, 0].max()) + grid_buffer,
        step,
    )
    y_points = _make_discretized_axis(
        float(all_points[:, 1].min()) - grid_buffer,
        float(all_points[:, 1].max()) + grid_buffer,
        step,
    )
    x_grid, y_grid = np.meshgrid(x_points, y_points, indexing="ij")
    grid_coordinates = np.column_stack((x_grid.ravel(), y_grid.ravel()))
    classifier = KNeighborsClassifier(
        n_neighbors=neighbour_count,
        weights="uniform",
        metric="minkowski",
        p=2,
    )
    classifier.fit(all_points, target_classes)
    assignments = classifier.predict(grid_coordinates).reshape(x_grid.shape)
    assignment_grid = np.stack((x_grid, y_grid, assignments), axis=-1)
    return assignment_grid, group_identifiers


def kmeans_group_grid(
    groups: Mapping[str, np.ndarray | pd.DataFrame],
    grid_step: float = 0.05,
    n_runs: int = 1,
    seed: int | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    """Cluster pooled points with K-Means and classify a discretized grid.

    The input mapping is used only to collect separate point arrays and to
    determine the number of clusters. Its names and original group memberships
    do not participate in fitting. Each K-Means run independently partitions
    all pooled observations into ``len(groups)`` clusters.

    Since K-Means cluster indices are arbitrary, clusters from every run after
    the first are aligned to the first run's centres using a minimum-cost
    one-to-one assignment. Grid classifications are then decided by majority
    vote over the aligned runs. Voting ties are resolved pseudo-randomly using
    the generator initialized by ``seed``.

    Parameters
    ----------
    groups : Mapping[str, numpy.ndarray or pandas.DataFrame]
        Collection of finite numeric point arrays with shape ``(n_points, 2)``.
        Mapping keys and original memberships are ignored by K-Means. The
        number of mapping entries determines the number of clusters.
    grid_step : float, default=0.05
        Finite positive spacing between adjacent grid coordinates. Bounds
        cover all observations and are rounded outwards to integer multiples
        of this value.
    n_runs : int, default=1
        Number of independent single-initialization K-Means fits whose aligned
        grid predictions are combined by voting.
    seed : int or None, optional
        Non-negative seed used to initialize the random generator before the
        first run. Supplying an integer makes fitting and tie resolution
        reproducible.

    Returns
    -------
    assignment_grid : numpy.ndarray
        Array with shape ``(n_x, n_y, 3)``. ``[..., 0]`` contains x
        coordinates, ``[..., 1]`` contains y coordinates, and ``[..., 2]``
        contains the voted 1-based cluster identifier.
    cluster_identifiers : dict[str, int]
        Neutral cluster-name mapping of the form ``{"Cluster 1": 1, ...}``.
        Identifiers refer to the centre ordering established by the first run.

    Raises
    ------
    TypeError
        If group data or execution arguments have incompatible types.
    ValueError
        If group data are invalid, fewer distinct observations than clusters
        are available, or an execution argument lies outside its valid range.

    Notes
    -----
    Every fit uses K-Means++ initialization and exactly one initialization.
    Multiple initializations are represented explicitly by ``n_runs`` because
    their grid assignments, rather than only their inertia values, must be
    retained for voting.
    """
    step = _validate_grid_step(grid_step)
    validated_groups = _validate_groups(groups)

    if isinstance(n_runs, bool) or not isinstance(n_runs, (int, np.integer)):
        raise TypeError("n_runs must be an integer.")

    run_count = int(n_runs)

    if run_count < 1:
        raise ValueError("n_runs must be at least one.")
    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, (int, np.integer))
    ):
        raise TypeError("seed must be an integer or None.")
    if seed is not None and seed < 0:
        raise ValueError("seed must be non-negative or None.")

    all_points = np.concatenate(
        [points for _, points in validated_groups],
        axis=0,
    )
    cluster_count = len(validated_groups)

    if all_points.shape[0] < cluster_count:
        raise ValueError(
            "The total number of observations must be at least the number of "
            "clusters."
        )
    if np.unique(all_points, axis=0).shape[0] < cluster_count:
        raise ValueError(
            "K-Means requires at least one distinct observation per cluster."
        )

    x_points = _make_discretized_axis(
        float(all_points[:, 0].min()),
        float(all_points[:, 0].max()),
        step,
    )
    y_points = _make_discretized_axis(
        float(all_points[:, 1].min()),
        float(all_points[:, 1].max()),
        step,
    )
    x_grid, y_grid = np.meshgrid(x_points, y_points, indexing="ij")
    grid_coordinates = np.column_stack((x_grid.ravel(), y_grid.ravel()))
    generator = np.random.default_rng(seed)
    run_assignments = np.empty(
        (run_count, grid_coordinates.shape[0]),
        dtype=np.int64,
    )
    reference_centres: np.ndarray | None = None
    maximum_random_state = np.iinfo(np.int32).max

    for run_index in range(run_count):
        run_seed = int(generator.integers(0, maximum_random_state))
        estimator = KMeans(
            n_clusters=cluster_count,
            init="k-means++",
            n_init=1,
            random_state=run_seed,
        ).fit(all_points)
        predicted_clusters = estimator.predict(grid_coordinates)

        if reference_centres is None:
            reference_centres = estimator.cluster_centers_.copy()
            aligned_clusters = predicted_clusters
        else:
            # Rows are current-run clusters and columns are reference clusters.
            # The resulting lookup converts arbitrary current labels into the
            # stable label space established by the first run.
            centre_costs = cdist(estimator.cluster_centers_, reference_centres)
            current_labels, reference_labels = linear_sum_assignment(centre_costs)
            current_to_reference = np.empty(cluster_count, dtype=np.int64)
            current_to_reference[current_labels] = reference_labels
            aligned_clusters = current_to_reference[predicted_clusters]

        run_assignments[run_index] = aligned_clusters

    if run_count == 1:
        voted_assignments = run_assignments[0]
    else:
        vote_counts = np.stack(
            [
                np.count_nonzero(run_assignments == cluster_index, axis=0)
                for cluster_index in range(cluster_count)
            ],
            axis=0,
        )
        maximum_votes = vote_counts.max(axis=0, keepdims=True)
        tied_for_maximum = vote_counts == maximum_votes

        # Independent continuous random values select uniformly among tied
        # candidates through argmax. Non-candidates cannot win because their
        # values are set below the random interval [0, 1).
        tie_breakers = generator.random(vote_counts.shape)
        tie_breakers[~tied_for_maximum] = -1.0
        voted_assignments = np.argmax(tie_breakers, axis=0)

    # K-Means and the internal voting representation are zero-based. Public
    # grid identifiers follow the same 1-based contract as the other methods.
    public_assignments = voted_assignments.reshape(x_grid.shape) + 1
    assignment_grid = np.stack((x_grid, y_grid, public_assignments), axis=-1)
    cluster_identifiers = {
        f"Cluster {cluster_index}": cluster_index
        for cluster_index in range(1, cluster_count + 1)
    }
    return assignment_grid, cluster_identifiers


def _validate_groups(
    groups: Mapping[str, np.ndarray | pd.DataFrame],
) -> list[tuple[str, np.ndarray]]:
    """Validate named two-dimensional point groups and convert them to arrays."""
    if not isinstance(groups, Mapping):
        raise TypeError(
            "groups must be a mapping from names to NumPy arrays or DataFrames."
        )
    if not groups:
        raise ValueError("groups must contain at least one group.")

    validated_groups: list[tuple[str, np.ndarray]] = []

    for group_name, points in groups.items():
        if not isinstance(group_name, str):
            raise TypeError("Every group name must be a string.")
        if isinstance(points, pd.DataFrame):
            points = points.to_numpy(copy=False)
        if not isinstance(points, np.ndarray):
            raise TypeError(
                f"Group {group_name!r} must contain a NumPy array or DataFrame."
            )
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(
                f"Group {group_name!r} must have shape (n_points, 2)."
            )
        if points.shape[0] == 0:
            raise ValueError(f"Group {group_name!r} must contain at least one point.")
        try:
            # Conversion, rather than a dtype-family check, also accepts
            # numeric pandas extension dtypes and object-backed numeric data.
            numeric_points = np.asarray(points, dtype=float)
        except (TypeError, ValueError) as error:
            raise TypeError(
                f"Group {group_name!r} must contain numeric values."
            ) from error

        if not np.isfinite(numeric_points).all():
            raise ValueError(
                f"Group {group_name!r} must contain only finite values."
            )

        validated_groups.append((group_name, numeric_points))

    return validated_groups


def _validate_p_distance(p_distance: float) -> float:
    """Validate and normalize a Minkowski distance order."""
    if isinstance(p_distance, bool) or not isinstance(
        p_distance,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError("p_distance must be a real number.")

    p_value = float(p_distance)

    if np.isnan(p_value) or p_value < 1:
        raise ValueError("p_distance must be at least one or positive infinity.")

    return p_value


def _validate_grid_step(grid_step: float) -> float:
    """Validate and normalize grid spacing."""
    if isinstance(grid_step, bool) or not isinstance(
        grid_step,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError("grid_step must be a real number.")

    step = float(grid_step)

    if not np.isfinite(step) or step <= 0:
        raise ValueError("grid_step must be finite and positive.")

    return step


def _make_discretized_axis(
    minimum: float,
    maximum: float,
    grid_step: float,
) -> np.ndarray:
    """Round coordinate bounds outwards onto an absolute step lattice."""
    minimum_scaled = minimum / grid_step
    maximum_scaled = maximum / grid_step
    minimum_tolerance = 1e-12 * max(1.0, abs(minimum_scaled))
    maximum_tolerance = 1e-12 * max(1.0, abs(maximum_scaled))
    lower_index = int(np.floor(minimum_scaled + minimum_tolerance))
    upper_index = int(np.ceil(maximum_scaled - maximum_tolerance))

    # Preserve a meaningful axis and explicit spacing when all observations
    # share one coordinate on this dimension.
    if lower_index == upper_index:
        lower_index -= 1
        upper_index += 1

    return np.arange(lower_index, upper_index + 1, dtype=float) * grid_step


__all__ = ["kmeans_group_grid", "knn_group_grid", "nearest_group_grid"]
