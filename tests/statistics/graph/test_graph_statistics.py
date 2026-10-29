"""Current graph APIs: numerical definitions and shared finalisation."""
import numpy as np
import pytest
from scipy.stats import wasserstein_distance

from pyspoc.component.statistics.graph import (
    CorrelationGraph, CovarianceGraph, DistributionGraph,
    ProximityGraph, RegressionGraph, ChowLiuTree,
)
from pyspoc.component.statistics.graph._base import GraphStatistic

DATA = np.random.default_rng(41).normal(size=(32, 4))


@pytest.mark.parametrize("graph_type, kwargs", [
    (CorrelationGraph, {}), (CovarianceGraph, {}),
    (RegressionGraph, {"estimator": "ridge"}),
    (ChowLiuTree, {"n_neighbors": 3}),
    (ProximityGraph, {"n_neighbors": 2}), (DistributionGraph, {}),
])
def test_current_graph_classes_return_feature_matrices(graph_type, kwargs):
    # Construct during execution rather than parametrisation so a constructor
    # regression cannot prevent the rest of this module from collecting.
    graph = graph_type(**kwargs)
    result = graph._compute(DATA)
    assert isinstance(graph, GraphStatistic)
    assert result.shape == (4, 4)
    assert np.isfinite(result).all()


def test_correlation_graph_pearson_weights():
    result = CorrelationGraph(estimator="Pearson", self_value=None)._compute(DATA)
    np.testing.assert_allclose(result, np.corrcoef(DATA, rowvar=False))


def test_distribution_graph_wasserstein_definition():
    result = DistributionGraph(metric="wasserstein")._compute(DATA)
    assert result[0, 1] == pytest.approx(wasserstein_distance(DATA[:, 0], DATA[:, 1]))
    np.testing.assert_allclose(result, result.T)
    np.testing.assert_allclose(np.diag(result), 0)


def test_proximity_complete_rbf_definition():
    result = ProximityGraph(topology="complete", weighting="rbf", gamma=0.25)._compute(DATA)
    assert result[0, 1] == pytest.approx(
        np.exp(-0.25 * np.square(DATA[:, 0] - DATA[:, 1]).sum()))
    np.testing.assert_allclose(result, result.T)


def test_proximity_mst_has_p_minus_one_edges():
    result = ProximityGraph(topology="mst", weighting="binary")._compute(DATA)
    assert np.count_nonzero(np.triu(result, 1)) == DATA.shape[1] - 1


def test_shared_thresholding_removes_both_signs():
    class FixedGraph(GraphStatistic):
        def _compute(self, data):
            return np.array([[1., -0.2, 0.8], [-0.2, 1., -0.9], [0.8, -0.9, 1.]])
    operation = FixedGraph(dim="p", nan_policy="raise",
                           enforce_adjacency_type="bipolar", zero_threshold=0.5)
    result = operation.compute(DATA[:, :3])
    np.testing.assert_array_equal(result, [[0, 0, 1], [0, 0, -1], [1, -1, 0]])


def test_proximity_constructor_capture_includes_defaults():
    operation = ProximityGraph(metric="minkowski", n_neighbors=2, p=3)
    assert operation.params["metric_kwargs"] == {"p": 3}
    assert operation.params["dim"] == "p"
    assert operation.params["n_neighbors"] == 2


@pytest.mark.parametrize("kwargs", [{"n_neighbors": 0}, {"radius": 0.0}, {"gamma": -1.0}])
def test_proximity_rejects_invalid_numeric_configuration(kwargs):
    with pytest.raises((TypeError, ValueError)):
        ProximityGraph(**kwargs)
