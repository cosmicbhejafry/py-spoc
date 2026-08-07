"""Contract tests for concrete statistics built on OrthogonalPCAEStatistic."""

from unittest.mock import Mock

import numpy as np
import pytest

from pyspoc.rstatistics.dimreduce.orthopcae import rstatistics as rstatistics_module
from pyspoc.rstatistics.dimreduce.orthopcae.rstatistics import (
    OrthogonalPCAEVarianceElbow,
    OrthogonalPCAEVarianceExplainedRatio,
)
from pyspoc.statistics.dimreduce.orthopcae._state import (
    OrthogonalPCAEFittedState,
)


def test_variance_elbow_selects_scalar_estimator_result(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """The elbow statistic should expose the estimator's selected dimension."""
    statistic = OrthogonalPCAEVarianceElbow(
        batch_size=4,
        components=2,
        max_bottleneck_dim=2,
    )

    estimator = object()
    find_elbow = Mock(return_value=2)
    monkeypatch.setattr(rstatistics_module, "find_elbow_point", find_elbow)

    result = statistic._get_result(estimator, components=(1, 2))

    find_elbow.assert_called_once_with(
        fitted_estimator=estimator,
        components=(1, 2),
    )

    assert result == 2
    assert statistic.identifier == "opcae-var-elbow"
    assert "scalar" in statistic.labels


def test_variance_explained_selects_requested_components(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """The enclosing statistic should apply one-based component selection."""
    statistic = OrthogonalPCAEVarianceExplainedRatio(
        batch_size=4,
        components=[1, 3],
        max_bottleneck_dim=3,
    )
    estimator_result = np.array([0.2, 0.5, 0.8])

    fitted_state = OrthogonalPCAEFittedState(
        variance_explained=estimator_result,
        reconstruction_loss=np.array([0.8, 0.5, 0.2]),
        dimensions=np.array([1, 2, 3]),
        baseline_loss=1.0,
    )
    estimator = object()
    extract = Mock(return_value=fitted_state)
    monkeypatch.setattr(rstatistics_module, "extract_pcae_scree_data", extract)

    result = statistic._get_result(estimator, components=(1, 3))

    np.testing.assert_array_equal(result, np.array([0.2, 0.8]))
    extract.assert_called_once_with(fitted_estimator=estimator)
