"""Similarity measures for grouped probability-density models."""

from __future__ import annotations

import numpy as np
import pandas as pd

from collections.abc import Callable, Sequence
from typing import Literal, TypeAlias

Linkage: TypeAlias = Literal["single", "complete", "average"]
ProbabilityDensity: TypeAlias = Callable[[np.ndarray], float | np.floating]


def gaussian_group_similarity(
    probability_densities: Sequence[ProbabilityDensity],
    group_points: Sequence[np.ndarray | pd.DataFrame],
    group_names: Sequence[str],
    linkage: Linkage = "average",
    log: bool = False,
) -> pd.DataFrame:
    r"""Calculate cross-density similarities between fitted Gaussian groups.

    For groups :math:`A` and :math:`B`, a point pair
    :math:`(x, y) \in A \times B` has similarity

    .. math::

        s_{A,B}(x,y) = p_B(x)p_A(y),

    where :math:`p_A` and :math:`p_B` are the respective fitted probability
    density functions. The requested linkage summarizes these similarities
    over every cross-group point pair.

    Parameters
    ----------
    probability_densities : Sequence of Callable
        One fitted multivariate PDF per group. Each callable must accept one
        one-dimensional point and return a non-negative finite scalar density.
        Sequence order must match ``group_points`` and ``group_names``.
    group_points : Sequence of numpy.ndarray
        One finite numeric array with shape ``(n_points, n_dimensions)`` per
        group. Every group must contain at least one point and all arrays must
        have the same number of dimensions.
    group_names : Sequence of str
        Unique names corresponding positionally to the PDFs and point arrays.
        These become both index and column labels of the returned matrix.
    linkage : {"single", "complete", "average"}, default="average"
        Point-pair aggregation rule. ``"single"`` selects the greatest
        similarity, ``"complete"`` selects the smallest, and ``"average"``
        calculates the arithmetic mean.
    log : bool, default=False
        If ``True``, calculate log similarities by adding the natural
        logarithms of the two cross-density values rather than multiplying
        the densities. This avoids loss of precision from multiplying very
        small densities in higher-dimensional comparisons.

    Returns
    -------
    pandas.DataFrame
        Symmetric square similarity matrix indexed and columned by
        ``group_names``. The diagonal applies the same linkage rule to all
        within-group point pairs rather than being forced to one. Values are
        log similarities when ``log=True``.

    Raises
    ------
    TypeError
        If an input collection, PDF, group name, point array, ``log``, or
        evaluated density has an incompatible type.
    ValueError
        If collection lengths differ, inputs are empty or malformed, group
        names are duplicated, linkage is unsupported, or a PDF returns a
        negative or non-finite density.

    Notes
    -----
    For one group pair, evaluating every :math:`s(x,y)` explicitly would
    allocate an ``n_A`` by ``n_B`` outer-product matrix. Since PDF values are
    non-negative, the supported summaries factorize exactly:

    .. math::

        \max_{x,y} p_B(x)p_A(y)
        = \max_x p_B(x)\max_y p_A(y),

    with equivalent identities for the minimum and arithmetic mean. The
    implementation therefore performs ``n_A + n_B`` PDF evaluations and
    allocates only two vectors for each group pair.

    In log mode, a pairwise similarity is

    .. math::

        \log s_{A,B}(x,y) = \log p_B(x) + \log p_A(y).

    The linkage rule is applied in log space. Consequently, ``"average"``
    returns the mean log similarity (the logarithm of the geometric mean),
    rather than the logarithm of the ordinary arithmetic-mean similarity.
    A zero density contributes negative infinity.
    """
    if not isinstance(probability_densities, Sequence):
        raise TypeError("probability_densities must be a sequence.")
    if not isinstance(group_points, Sequence):
        raise TypeError("group_points must be a sequence.")
    if not isinstance(group_names, Sequence) or isinstance(group_names, str):
        raise TypeError("group_names must be a sequence of strings.")

    densities = tuple(probability_densities)
    points = tuple(np.array(p) for p in group_points)
    names = tuple(group_names)
    group_count = len(names)

    if group_count == 0:
        raise ValueError("At least one group must be supplied.")
    if len(densities) != group_count or len(points) != group_count:
        raise ValueError(
            "probability_densities, group_points, and group_names must have "
            "equal lengths."
        )
    if linkage not in ("single", "complete", "average"):
        raise ValueError(
            "linkage must be one of 'single', 'complete', or 'average'."
        )
    if not isinstance(log, bool):
        raise TypeError("log must be a boolean.")
    if not all(isinstance(name, str) for name in names):
        raise TypeError("Every group name must be a string.")
    if len(set(names)) != group_count:
        raise ValueError("Group names must be unique.")
    if not all(callable(density) for density in densities):
        raise TypeError("Every probability density must be callable.")

    validated_points: list[np.ndarray] = []
    dimensionality: int | None = None

    for group_index, group in enumerate(points):
        if not isinstance(group, np.ndarray):
            raise TypeError(f"group_points[{group_index}] must be a NumPy array.")
        if group.ndim != 2:
            raise ValueError(
                f"group_points[{group_index}] must have shape "
                "(n_points, n_dimensions)."
            )
        if group.shape[0] == 0 or group.shape[1] == 0:
            raise ValueError(
                f"group_points[{group_index}] must contain at least one point "
                "and one dimension."
            )

        try:
            numeric_group = np.asarray(group, dtype=float)
        except (TypeError, ValueError) as error:
            raise TypeError(
                f"group_points[{group_index}] must contain numeric values."
            ) from error

        if not np.isfinite(numeric_group).all():
            raise ValueError(
                f"group_points[{group_index}] must contain only finite values."
            )
        if dimensionality is None:
            dimensionality = numeric_group.shape[1]
        elif numeric_group.shape[1] != dimensionality:
            raise ValueError("Every point group must have the same dimensionality.")

        validated_points.append(numeric_group)

    similarities = np.empty((group_count, group_count), dtype=float)

    for first_index in range(group_count):
        for second_index in range(first_index, group_count):
            # Evaluate points from each group under the opposite group's PDF.
            # Calls remain scalar to support functools.partial objects and
            # other PDF callables that do not implement vectorized evaluation.
            first_cross_densities = _evaluate_pdf(
                densities[second_index],
                validated_points[first_index],
                pdf_name=names[second_index],
            )
            second_cross_densities = _evaluate_pdf(
                densities[first_index],
                validated_points[second_index],
                pdf_name=names[first_index],
            )

            if log:
                # log(0) is intentionally retained as -inf: a zero-density
                # pairing has zero similarity and therefore log similarity
                # -inf. Suppress only NumPy's expected divide-by-zero warning.
                with np.errstate(divide="ignore"):
                    first_values = np.log(first_cross_densities)
                    second_values = np.log(second_cross_densities)
            else:
                first_values = first_cross_densities
                second_values = second_cross_densities

            if linkage == "single":
                first_summary = first_values.max()
                second_summary = second_values.max()
            elif linkage == "complete":
                first_summary = first_values.min()
                second_summary = second_values.min()
            else:
                first_summary = first_values.mean()
                second_summary = second_values.mean()

            similarity = (
                first_summary + second_summary
                if log
                else first_summary * second_summary
            )

            # Negative infinity is a valid log similarity produced by a zero
            # density. NaN and positive infinity always indicate bad output.
            if np.isnan(similarity) or np.isposinf(similarity):
                raise ValueError(
                    f"Similarity between {names[first_index]!r} and "
                    f"{names[second_index]!r} is non-finite."
                )

            similarities[first_index, second_index] = similarity
            similarities[second_index, first_index] = similarity

    index = pd.Index(names, name="Group")
    return pd.DataFrame(similarities, index=index, columns=index.copy())


def _evaluate_pdf(
    probability_density: ProbabilityDensity,
    points: np.ndarray,
    *,
    pdf_name: str,
) -> np.ndarray:
    """Evaluate a scalar PDF over points and validate its density values."""
    evaluated = np.empty(points.shape[0], dtype=float)

    for point_index, point in enumerate(points):
        value = probability_density(point)

        if isinstance(value, bool) or not np.isscalar(value):
            raise TypeError(
                f"PDF for group {pdf_name!r} must return a numeric scalar."
            )

        try:
            density = float(value)
        except (TypeError, ValueError) as error:
            raise TypeError(
                f"PDF for group {pdf_name!r} must return a numeric scalar."
            ) from error

        if not np.isfinite(density):
            raise ValueError(
                f"PDF for group {pdf_name!r} returned a non-finite density "
                f"at point index {point_index}."
            )
        if density < 0:
            raise ValueError(
                f"PDF for group {pdf_name!r} returned a negative density "
                f"at point index {point_index}."
            )

        evaluated[point_index] = density

    return evaluated


__all__ = ["Linkage", "ProbabilityDensity", "gaussian_group_similarity"]
