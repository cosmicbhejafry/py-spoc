"""Distance- and distribution-based feature graph Statistics."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from sklearn.metrics import pairwise_distances

import pyspoc._argchecking as argchk
from pyspoc.settings import settings
from pyspoc._tag import Tag, Type, Signature

from ._base import GraphStatistic



class ProximityGraph(GraphStatistic):
    """Build a complete, nearest-neighbour, radius, or MST feature graph.

    Features are treated as vectors in observation space. Edge selection always
    uses the configured pairwise distance, after which retained edges may store
    that distance, a binary indicator, or an RBF similarity.
    """

    _long_name = "Proximity Graph"
    _identifier = "graph-proximity"
    _type = Type.DESCRIPTIVE
    _signatures = (Signature.TOPOLOGY, Signature.DIMENSIONALITY)
    _tags = (Tag.GRAPH_WEIGHTED, Tag.SYMMETRICAL, Tag.DISTANCE)

    def __init__(
            self,
            metric: str = "euclidean",
            topology: Literal["complete", "knn", "radius", "mst"] = "knn",
            n_neighbors: int = 5,
            radius: float = 1.0,
            neighbor_mode: Literal["union", "mutual"] = "union",
            weighting: Literal["distance", "rbf", "binary"] = "rbf",
            gamma: float | None = None,
            dim: Literal["n", "p"] = "p",
            nan_policy: Literal[
                "raise", "row-omit", "col-omit"] = settings.current.data_nan_policy,
            **metric_kwargs: Any):

        self._n_neighbors = argchk.check_natural_number(n_neighbors, "n_neighbors")

        self._radius = argchk.check_float(
            radius,
            exclusive_minimum=0,
            arg_name="radius")
        
        self._gamma = (
            argchk.check_float(gamma, exclusive_minimum=0, arg_name="gamma")
            if gamma is not None
            else None)

        self._metric = metric
        self._topology = topology
        self._neighbor_mode = neighbor_mode
        self._weighting = weighting
        self._metric_kwargs = metric_kwargs
        super().__init__(dim=dim, nan_policy=nan_policy)


    def _compute(self, data: np.ndarray) -> np.ndarray:
        """Return an adjacency matrix connecting feature vectors by proximity."""
        
        distances = pairwise_distances(
            data.T,
            metric=self._metric,
            **self._metric_kwargs)
        
        if not np.all(np.isfinite(distances)):
            raise ValueError(
                f"The {self._metric!r} metric produced non-finite feature distances.")
        
        edge_mask = _select_edges(
            distances,
            topology=self._topology,
            n_neighbors=self._n_neighbors,
            radius=self._radius,
            neighbor_mode=self._neighbor_mode)
        
        gamma = self._gamma if self._gamma is not None else 1.0 / data.shape[0]
        
        return _weight_edges(
            distances,
            edge_mask,
            weighting=self._weighting,
            gamma=gamma)



class DistributionGraph(GraphStatistic):
    """Compare feature columns as empirical one-dimensional distributions.

    Wasserstein distance operates directly on observed values. Jensen-Shannon
    distance first estimates every feature distribution over common histogram
    bins, ensuring that corresponding probability-vector entries describe the
    same value intervals.
    """

    _long_name = "Distribution Graph"
    _identifier = "graph-distribution"
    _type = Type.DESCRIPTIVE
    _signatures = (Signature.TOPOLOGY, Signature.DIMENSIONALITY)
    _tags = (Tag.GRAPH_WEIGHTED, Tag.SYMMETRICAL, Tag.DISTRIBUTION)

    def __init__(
            self,
            metric: Literal["wasserstein", "jensen_shannon"] = "wasserstein",
            bins: int = 32,
            weighting: Literal["distance", "rbf", "binary"] = "distance",
            gamma: float | None = None,
            dim: Literal["n", "p"] = "p",
            nan_policy: Literal[
                "raise", "row-omit", "col-omit"] = settings.current.data_nan_policy):

        self._bins = argchk.check_integer(bins, minimum=2, arg_name="bins")
        self._metric = metric
        self._weighting = weighting
        self._gamma = (
            argchk.check_float(gamma, exclusive_minimum=0, arg_name="gamma")
            if gamma is not None
            else None)
            
        super().__init__(dim=dim, nan_policy=nan_policy)


    def _compute(self, data: np.ndarray) -> np.ndarray:
        """Return complete pairwise distribution distances or similarities."""

        n_features = data.shape[1]
        distances = np.zeros((n_features, n_features), dtype=float)
        histograms: list[np.ndarray] | None = None

        if self._metric == "jensen_shannon":
            lower = float(np.min(data))
            upper = float(np.max(data))
            if np.isclose(lower, upper):
                upper = lower + 1.0
            edges = np.linspace(lower, upper, self._bins + 1)
            histograms = [
                np.histogram(data[:, feature], bins=edges)[0].astype(float)
                for feature in range(n_features)
            ]

        assert histograms is not None
        
        for i in range(n_features):
            for j in range(i + 1, n_features):
                if self._metric == "wasserstein":
                    distance = wasserstein_distance(data[:, i], data[:, j])
                else:
                    distance = jensenshannon(histograms[i], histograms[j])

                distances[i, j] = distances[j, i] = distance

        edge_mask = ~np.eye(n_features, dtype=bool)
        gamma = self._gamma if self._gamma is not None else 1.0
        return _weight_edges(
            distances,
            edge_mask,
            weighting=self._weighting,
            gamma=gamma)


def _minimum_spanning_tree(distances: np.ndarray) -> np.ndarray:
    """Return a symmetric minimum-distance spanning tree using Kruskal's algorithm."""

    n_nodes = distances.shape[0]
    parents = np.arange(n_nodes)
    ranks = np.zeros(n_nodes, dtype=int)

    def find(node: int) -> int:
        while parents[node] != node:
            parents[node] = parents[parents[node]]
            node = int(parents[node])
        return node

    def union(left: int, right: int) -> bool:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return False
        if ranks[left_root] < ranks[right_root]:
            left_root, right_root = right_root, left_root
        parents[right_root] = left_root
        if ranks[left_root] == ranks[right_root]:
            ranks[left_root] += 1
        return True

    edges = [
        (float(distances[i, j]), i, j)
        for i in range(n_nodes)
        for j in range(i + 1, n_nodes)
    ]
    edges.sort(key=lambda edge: edge[0])

    mask = np.zeros_like(distances, dtype=bool)
    selected = 0
    for _, i, j in edges:
        if union(i, j):
            mask[i, j] = mask[j, i] = True
            selected += 1
            if selected == n_nodes - 1:
                break
    return mask


def _select_edges(
        distances: np.ndarray,
        *,
        topology: Literal["complete", "knn", "radius", "mst"],
        n_neighbors: int,
        radius: float,
        neighbor_mode: Literal["union", "mutual"]) -> np.ndarray:
    """Select graph edges from a complete symmetric distance matrix."""

    n_nodes = distances.shape[0]
    off_diagonal = ~np.eye(n_nodes, dtype=bool)

    if topology == "complete":
        return off_diagonal
    if topology == "radius":
        return (distances <= radius) & off_diagonal
    if topology == "mst":
        return _minimum_spanning_tree(distances)

    directed = np.zeros_like(distances, dtype=bool)
    count = min(n_neighbors, max(0, n_nodes - 1))
    for node in range(n_nodes):
        order = np.argsort(distances[node], kind="stable")
        neighbors = order[order != node][:count]
        directed[node, neighbors] = True
    return directed & directed.T if neighbor_mode == "mutual" else directed | directed.T


def _weight_edges(
        distances: np.ndarray,
        edge_mask: np.ndarray,
        *,
        weighting: Literal["distance", "rbf", "binary"],
        gamma: float) -> np.ndarray:

    """Apply the requested weights to a selected undirected edge set."""
    if weighting == "distance":
        weights = distances
    elif weighting == "rbf":
        weights = np.exp(-gamma * np.square(distances))
    else:
        weights = np.ones_like(distances, dtype=float)
    return np.where(edge_mask, weights, 0.0)
