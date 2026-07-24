"""Contract tests for concrete statistics built on OrthogonalPCAEStatistic."""

import numpy as np
from pyspoc.rstatistics.dimreduce.orthopcae.rstatistics import (
    OrthogonalPCAEVarianceElbow,
    OrthogonalPCAEVarianceExplained,
)


def test_variance_elbow_selects_scalar_estimator_result() -> None:
    """The elbow statistic should expose the estimator's selected dimension."""
    statistic = OrthogonalPCAEVarianceElbow(
        batch_size=4,
        components=2,
        max_bottleneck_dim=2,
    )

    result = statistic._get_result(
        {"optimal_bottleneck_dimension": 2},
        components=(1, 2),
    )

    assert result == 2
    assert statistic.identifier == "opcae-var-elbow"
    assert "scalar" in statistic.labels


def test_variance_explained_selects_requested_components() -> None:
    """The enclosing statistic should apply one-based component selection."""
    statistic = OrthogonalPCAEVarianceExplained(
        batch_size=4,
        components=[1, 3],
        max_bottleneck_dim=3,
    )
    estimator_result = np.array([0.2, 0.5, 0.8])

    result = statistic._get_result(
        {"pseudo_variance_explained": estimator_result},
        components=(1, 3),
    )

    np.testing.assert_array_equal(result, np.array([0.2, 0.8]))
