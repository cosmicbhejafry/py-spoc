"""Tests for partial-correlation matrix estimation."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import pinvh
from sklearn.covariance import GraphicalLasso, LedoitWolf

from pyspoc.component.statistics.association.correlation import PartialCorrelation
from pyspoc.component.statistics.base import Statistic
from pyspoc._tag import Tag


def _partial_from_precision(precision: np.ndarray) -> np.ndarray:
    """Apply the mathematical precision-to-partial-correlation conversion."""
    diagonal = np.clip(np.diag(precision), a_min=0.0, a_max=None)
    scale = np.sqrt(np.outer(diagonal, diagonal))
    result = np.divide(
        -precision,
        scale,
        out=np.zeros_like(precision, dtype=float),
        where=scale > np.finfo(float).eps,
    )
    result = np.clip((result + result.T) / 2, -1.0, 1.0)
    np.fill_diagonal(result, 1.0)
    return result


@pytest.fixture
def correlated_data() -> np.ndarray:
    """Return reproducible, nonsingular observations with correlated features."""
    rng = np.random.default_rng(12)
    first = rng.normal(size=120)
    second = 0.7 * first + rng.normal(scale=0.5, size=120)
    third = -0.4 * first + 0.2 * second + rng.normal(scale=0.7, size=120)
    return np.column_stack((first, second, third))


def test_partial_correlation_is_a_statistic() -> None:
    """The concrete implementation should participate in Statistic orchestration."""
    assert isinstance(PartialCorrelation(), Statistic)


def test_pseudoinverse_matches_precision_normalization(
    correlated_data: np.ndarray,
) -> None:
    """The pseudoinverse path should implement the standard partial-correlation formula."""
    centered = correlated_data - np.mean(correlated_data, axis=0)
    covariance = centered.T @ centered / correlated_data.shape[0]
    expected = _partial_from_precision(pinvh(covariance))

    result = PartialCorrelation(inversion_method="pseudo")._compute(correlated_data)

    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("alias", ["pseudo", "pseudoinverse"])
def test_pseudoinverse_aliases_are_supported(
    alias: str,
    correlated_data: np.ndarray,
) -> None:
    """Supported pseudoinverse aliases should resolve identically."""
    expected = PartialCorrelation(inversion_method="pseudo")._compute(correlated_data)
    result = PartialCorrelation(inversion_method=alias)._compute(correlated_data)  # type: ignore[arg-type]
    np.testing.assert_allclose(result, expected)


def test_pseudoinverse_supports_singular_covariance() -> None:
    """Duplicated feature columns should not make pseudoinverse estimation fail."""
    first = np.arange(10, dtype=float)
    data = np.column_stack((first, first, np.square(first)))

    result = PartialCorrelation(inversion_method="pseudo")._compute(data)

    assert result.shape == (3, 3)
    assert np.all(np.isfinite(result))
    np.testing.assert_allclose(result, result.T)
    np.testing.assert_allclose(np.diag(result), 1.0)


def test_graphical_lasso_matches_sklearn_precision(
    correlated_data: np.ndarray,
) -> None:
    """A fixed Graphical LASSO penalty should use sklearn's sparse precision estimate."""
    alpha = 0.08
    precision = GraphicalLasso(alpha=alpha).fit(correlated_data).precision_
    expected = _partial_from_precision(precision)

    result = PartialCorrelation(
        inversion_method="graphical_lasso",
        alpha=alpha,
    )._compute(correlated_data)

    np.testing.assert_allclose(result, expected)


def test_ledoit_wolf_matches_sklearn_precision(
    correlated_data: np.ndarray,
) -> None:
    """Ledoit--Wolf should provide an automatically shrunk precision matrix."""
    precision = LedoitWolf(store_precision=True).fit(correlated_data).precision_
    expected = _partial_from_precision(precision)

    result = PartialCorrelation(inversion_method="ledoit_wolf")._compute(correlated_data)

    np.testing.assert_allclose(result, expected)


def test_squared_partial_correlations_discard_sign(
    correlated_data: np.ndarray,
) -> None:
    """The squared option should square every signed coefficient after normalization."""
    signed = PartialCorrelation(squared=False)._compute(correlated_data)
    squared = PartialCorrelation(squared=True)._compute(correlated_data)

    np.testing.assert_allclose(squared, np.square(signed))
    assert Tag.SIGNED in PartialCorrelation(squared=False).get_tags()
    assert Tag.SIGNED not in PartialCorrelation(squared=True).get_tags()


def test_observation_dimension_returns_observation_matrix() -> None:
    """Selecting dimension n should treat original rows as graph variables."""
    data = np.arange(24, dtype=float).reshape(6, 4)
    result = PartialCorrelation(dim="n").compute(data)
    assert result.shape == (data.shape[0], data.shape[0])


def test_nan_omit_removes_incomplete_observations(
    correlated_data: np.ndarray,
) -> None:
    """Listwise omission should equal explicitly removing the affected row."""
    with_nan = correlated_data.copy()
    with_nan[4, 1] = np.nan

    result = PartialCorrelation(nan_policy="row-omit").compute(with_nan)
    expected = PartialCorrelation(nan_policy="raise")._compute(
        np.delete(correlated_data, 4, axis=0)
    )
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("policy", ["omit", "propagate"])
def test_obsolete_nan_policies_are_rejected(policy: str) -> None:
    """Policies must explicitly identify the axis to omit."""
    with pytest.raises((TypeError, ValueError)):
        PartialCorrelation(nan_policy=policy)


def test_nan_raise_reports_missing_data(correlated_data: np.ndarray) -> None:
    """The strict missing-value policy should fail before covariance estimation."""
    data = correlated_data.copy()
    data[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        PartialCorrelation(nan_policy="raise")._compute(data)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"inversion_method": "unknown"},
        {"dim": "invalid"},
        {"nan_policy": "invalid"},
        {"squared": "yes"},
        {"cv_mode": "invalid"},
    ],
)
def test_invalid_estimator_configuration_is_rejected(kwargs: dict[str, object]) -> None:
    """Invalid precision-estimator arguments should fail during construction."""
    with pytest.raises((TypeError, ValueError)):
        PartialCorrelation(**kwargs)  # type: ignore[arg-type]
