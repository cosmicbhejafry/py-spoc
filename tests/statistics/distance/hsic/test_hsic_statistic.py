from __future__ import annotations

import numpy as np

from pyspoc.settings import settings
from pyspoc.statistics.distance.hsic import (
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
        )

    expected = pairwise_hsic(
        data.T,
        biased=True,
        metric="linear",
        max_workers=2,
    )

    np.testing.assert_allclose(statistic.compute(data), expected)


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

    np.testing.assert_allclose(statistic.compute(data), expected)


def test_biased_identifier_is_instance_local() -> None:
    biased = HilbertSchmidtIndependenceCriterion(biased=True)
    unbiased = HilbertSchmidtIndependenceCriterion(biased=False)

    assert biased.identifier() == "hsic.biased"
    assert unbiased.identifier() == "hsic"
