from __future__ import annotations

import numpy as np

from pyspoc.settings import settings
from pyspoc.component.statistics.association.distance.hsic import (
    HilbertSchmidtIndependenceCriterion,
    pairwise_hsic,
)
from hyppo.independence import Hsic


def test_statistic_computes_over_selected_dimension() -> None:
    rng = np.random.default_rng(8)
    data = rng.normal(size=(12, 4))
    with settings.override(max_worker_threads=2):
        statistic = HilbertSchmidtIndependenceCriterion(
            dim="n",
            biased=True,
            metric="linear",
            downsample=False,
        )

    expected, errors = pairwise_hsic(
        data.T,
        biased=True,
        metric="linear",
        max_workers=2,
    )

    np.testing.assert_allclose(statistic.compute(data), expected, rtol=2e-5, atol=2e-7)


def test_statistic_compared_to_hyppo() -> None:
    rng = np.random.default_rng(8)
    data = rng.normal(size=(12, 4))
    with settings.override(max_worker_threads=2):
        statistic = HilbertSchmidtIndependenceCriterion(
            dim="p",
            biased=True,
            metric="linear",
        )

    p = data.shape[1]
    expected = np.zeros(shape=(p,p))

    for i in range(p):
        for j in range(i, p):
            x, y = data[:,i], data[:,j]
            x = x.reshape(-1,1)
            y = y.reshape(-1,1)
            result = Hsic(compute_kernel="linear", bias=True).statistic(x,y)
            expected[i,j] = result

            if i != j:
                expected[j,i] = result

    # The public default deliberately downsamples to float32. Compare with a
    # float32-appropriate tolerance rather than NumPy's float64-tight default.
    np.testing.assert_allclose(statistic.compute(data), expected, rtol=2e-5, atol=2e-7)


def test_bias_is_recorded_in_configuration_with_shared_statistic_identifier() -> None:
    biased = HilbertSchmidtIndependenceCriterion(biased=True)
    unbiased = HilbertSchmidtIndependenceCriterion(biased=False)

    assert biased.get_identifier() == "hsic"
    assert biased.params["biased"] is True
    assert unbiased.params["biased"] is False
    assert unbiased.get_identifier() == "hsic"
