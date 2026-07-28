"""Tests for the OrthogonalPCAEStatistic foundation."""

from unittest.mock import Mock

import numpy as np
import pytest

from pyspoc.statistics.dimreduce.orthopcae import _mixin as mixin_module
from pyspoc.statistics.dimreduce.orthopcae._base import (
    OrthogonalPCAEStatistic,
)
from pyspoc.settings import settings


class ResultStatistic(OrthogonalPCAEStatistic):
    """Minimal concrete statistic exposing estimator results for tests."""

    @property
    def name(self) -> str:
        return "Orthogonal PCAE test statistic"

    @property
    def identifier(self) -> str:
        return "orthopcae-test"

    @property
    def labels(self) -> tuple[str, ...]:
        return ("scalar", "non-linear")

    def _get_result(
            self,
            fitted_estimator: object,
            components: tuple[int, ...]) -> np.ndarray | float:
        self.received_components = components
        return fitted_estimator.selected  # type: ignore[attr-defined, no-any-return]


def test_integer_components_expand_to_one_based_sequence() -> None:
    """An integer component count should select every component up to it."""
    statistic = ResultStatistic(
        batch_size=4,
        components=3,
        max_bottleneck_dim=3,
    )

    assert statistic._components == (1, 2, 3)


def test_iterable_components_are_sorted_and_deduplicated() -> None:
    """Explicit component selections should use canonical ascending order."""
    statistic = ResultStatistic(
        batch_size=4,
        components=[3, 1, 3],
        max_bottleneck_dim=3,
    )

    assert statistic._components == (1, 3)


@pytest.mark.parametrize("components", [0, [], [-1, 1]])
def test_components_must_contain_positive_integers(
        components: int | list[int]) -> None:
    """Component selections should be nonempty and strictly positive."""
    with pytest.raises(ValueError):
        ResultStatistic(
            batch_size=4,
            components=components,
            max_bottleneck_dim=3,
        )


def test_configured_bottleneck_is_preserved_until_compute() -> None:
    """A configured bottleneck should not be expanded during initialization."""
    statistic = ResultStatistic(
        batch_size=4,
        components=[1, 3],
        max_bottleneck_dim=2,
    )

    assert statistic._max_bottleneck_dim == 2
    assert statistic._components == (1, 3)


def test_default_bottleneck_is_resolved_during_compute() -> None:
    """An omitted bottleneck should remain unresolved until data are known."""
    statistic = ResultStatistic(batch_size=4, components=2)

    assert statistic._max_bottleneck_dim is None


@pytest.mark.parametrize(
    ("argument_name", "argument_value"),
    [
        ("batch_size", True),
        ("train_steps", True),
        ("alpha", True),
    ],
)
def test_boolean_numeric_arguments_are_rejected(
        argument_name: str,
        argument_value: bool) -> None:
    """Boolean values should not satisfy numeric constructor semantics."""
    kwargs = {
        "batch_size": 4,
        "components": 2,
        argument_name: argument_value,
    }

    with pytest.raises(TypeError, match=argument_name):
        ResultStatistic(**kwargs)


def test_continuous_arguments_are_normalized_to_float() -> None:
    """Accepted integral values should have stable continuous storage types."""
    statistic = ResultStatistic(
        batch_size=4,
        components=2,
        alpha=1,
        burn_in_steps_prop=0,
    )

    assert statistic._alpha == 1.0
    assert isinstance(statistic._alpha, float)
    assert statistic._burn_in_steps_prop == 0.0
    assert isinstance(statistic._burn_in_steps_prop, float)


def test_compute_resolves_dimensions_and_delegates_to_cached_estimator(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Computation should resolve the estimator and select its result."""
    statistic = ResultStatistic(
        batch_size=4,
        components=[1, 2],
        train_steps=20,
        burn_in_steps_prop=0.2,
        alpha=0.3,
        max_bottleneck_dim=10,
        shuffle=False,
        random_seed=7,
    )
    data = np.arange(20, dtype=np.float32).reshape(5, 4)
    expected = np.array([0.25, 0.75])
    estimator = Mock()
    estimator.selected = expected
    estimator.fit.return_value = estimator
    get_or_create = Mock(return_value=estimator)
    monkeypatch.setattr(
        mixin_module.OrthogonalPCAEEstimator,
        "get_or_create",
        get_or_create,
    )

    result = statistic.compute(data)

    assert result is expected
    assert statistic._estimator_ is estimator
    assert statistic._max_bottleneck_dim == 10
    assert statistic.received_components == (1, 2)
    get_or_create.assert_called_once_with(
        data=data,
        batch_size=4,
        max_bottleneck_dim=4,
        train_steps=20,
        burn_in_steps_prop=0.2,
        alpha=0.3,
        shuffle=False,
        random_seed=7,
    )
    estimator.fit.assert_called_once_with(data)


def test_compute_uses_current_random_seed_when_not_overridden(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Estimator resolution should use the active package seed by default."""
    statistic = ResultStatistic(
        batch_size=4,
        components=2,
        max_bottleneck_dim=2,
    )
    data = np.ones((3, 2), dtype=np.float32)
    estimator = Mock()
    estimator.selected = 1.0
    estimator.fit.return_value = estimator
    get_or_create = Mock(return_value=estimator)
    monkeypatch.setattr(
        mixin_module.OrthogonalPCAEEstimator,
        "get_or_create",
        get_or_create,
    )

    with settings.override(random_seed=29):
        statistic.compute(data)

    assert get_or_create.call_args.kwargs["random_seed"] == 29


def test_compute_does_not_permanently_remove_unavailable_components(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Effective component filtering should be recalculated for each dataset."""
    statistic = ResultStatistic(
        batch_size=2,
        components=[1, 3],
    )
    estimator = Mock()
    estimator.selected = np.array([0.2, 0.8])
    estimator.fit.return_value = estimator
    monkeypatch.setattr(
        mixin_module.OrthogonalPCAEEstimator,
        "get_or_create",
        Mock(return_value=estimator),
    )

    statistic.compute(np.ones((2, 2), dtype=np.float32))
    assert statistic._components == (1, 3)
    assert statistic.received_components == (1,)

    statistic.compute(np.ones((5, 4), dtype=np.float32))
    assert statistic._components == (1, 3)
    assert statistic.received_components == (1, 3)
