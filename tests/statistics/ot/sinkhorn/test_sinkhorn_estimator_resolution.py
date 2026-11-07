"""Resolve cached estimator classes across the Sinkhorn statistic hierarchy."""

from typing import Generic, TypeVar

import pytest

pytest.importorskip("torch")
pytest.importorskip("geomloss.ot")

from pyspoc.statistics.ot.sinkhorn._base import (
    PairwiseSinkhornStatistic,
    PairwiseSinkhornEstimator)
from pyspoc.statistics.ot.sinkhorn.divergence import (
    PairwiseSinkhornDivergence,
    PairwiseSinkhornDivergenceEstimator)
from pyspoc.statistics.ot.sinkhorn.plan import (
    PairwiseSinkhornPlanEstimator,
    PairwiseSinkhornPlanEffectiveDestinationsSummary,
    PairwiseSinkhornPlanEntropy,
    
)


CASES = [
    pytest.param(PairwiseSinkhornDivergence, {}, PairwiseSinkhornDivergenceEstimator,
                 id="divergence"),
    pytest.param(PairwiseSinkhornPlanEntropy, {"entropy_type": "allocation"},
                 PairwiseSinkhornPlanEstimator, id="allocation-entropy"),
    pytest.param(PairwiseSinkhornPlanEntropy, {"entropy_type": "mass-weighted"},
                 PairwiseSinkhornPlanEstimator, id="conditional-entropy"),
    pytest.param(PairwiseSinkhornPlanEffectiveDestinationsSummary, {"summary_type": "mean"},
                 PairwiseSinkhornPlanEstimator, id="effective-destinations"),
]


@pytest.mark.parametrize("statistic_type, options, expected", CASES)
def test_concrete_statistic_construction_resolves_estimator(statistic_type, options, expected):
    statistic = statistic_type(loss="euclidean", **options)
    assert statistic._estimator_type is expected


@pytest.mark.parametrize("statistic_type, options, expected", CASES)
def test_plain_descendants_retain_estimator_specialization(statistic_type, options, expected):
    # Plain subclasses do not themselves declare generic arguments. Resolution
    # must follow the inheritance chain to the ancestor that supplied them.
    child = type("Child", (statistic_type,), {})
    grandchild = type("Grandchild", (child,), {})
    assert grandchild(loss="euclidean", **options)._estimator_type is expected


class MarkerMixin:
    pass


class MixedEntropy(MarkerMixin, PairwiseSinkhornPlanEntropy):
    pass


def test_unrelated_plain_mixin_is_ignored():
    statistic = MixedEntropy(loss="euclidean", entropy_type="allocation")
    assert statistic._estimator_type is PairwiseSinkhornPlanEstimator


EstimatorT = TypeVar("EstimatorT", bound=PairwiseSinkhornEstimator)


class IntermediateStatistic(PairwiseSinkhornStatistic[EstimatorT], Generic[EstimatorT]):
    def _pairwise_compute_indexed(self, x, y, index):
        return 0.0


class SpecializedStatistic(IntermediateStatistic[PairwiseSinkhornPlanEstimator]):
    pass


def test_type_variable_is_substituted_through_generic_intermediate():
    statistic = SpecializedStatistic(loss="euclidean")
    assert statistic._estimator_type is PairwiseSinkhornPlanEstimator


@pytest.mark.parametrize("statistic_type", [PairwiseSinkhornStatistic, IntermediateStatistic])
def test_unresolved_generic_raises_informative_error(statistic_type):
    with pytest.raises(TypeError, match="No concrete Sinkhorn estimator"):
        statistic_type._get_estimator_type()
