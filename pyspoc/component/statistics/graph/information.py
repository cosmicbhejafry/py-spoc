"""Information-theoretic feature graph Statistics."""

from __future__ import annotations

import numpy as np
from sklearn.feature_selection import mutual_info_regression
from typing import Literal

import pyspoc._argchecking as argchk
from pyspoc._random import resolve_random_seed
from pyspoc._tag import Tag, Type, Signature
from pyspoc.settings import settings

from ._base import GraphStatistic


def _mutual_information_matrix(
        data: np.ndarray,
        *,
        n_neighbors: int,
        random_seed: int | None) -> np.ndarray:
    """Estimate a non-negative symmetric mutual-information matrix."""

    n_features = data.shape[1]
    result = np.zeros((n_features, n_features), dtype=float)

    for i in range(n_features):
        for j in range(i + 1, n_features):
            # sklearn exposes MI through a regression-oriented API. Averaging
            # both orientations removes finite-sample directional variation
            # while preserving the symmetry of theoretical mutual information.
            forward = mutual_info_regression(
                data[:, [i]],
                data[:, j],
                n_neighbors=n_neighbors,
                random_state=random_seed)[0]
            reverse = mutual_info_regression(
                data[:, [j]],
                data[:, i],
                n_neighbors=n_neighbors,
                random_state=random_seed)[0]
            result[i, j] = result[j, i] = max(0.0, (forward + reverse) / 2)

    return result


def _maximum_spanning_tree(weights: np.ndarray) -> np.ndarray:
    """Return a symmetric maximum-weight spanning forest using Kruskal's algorithm."""
    n_nodes = weights.shape[0]
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
        (float(weights[i, j]), i, j)
        for i in range(n_nodes)
        for j in range(i + 1, n_nodes)]
    edges.sort(key=lambda edge: edge[0], reverse=True)

    tree = np.zeros_like(weights, dtype=float)
    selected = 0

    for weight, i, j in edges:

        if union(i, j):
            tree[i, j] = tree[j, i] = weight
            selected += 1

            if selected == n_nodes - 1:
                break

    return np.clip(tree, 0.0)



class ChowLiuTree(GraphStatistic):
    """Approximate the feature distribution with a maximum-MI spanning tree.
    
    
    Parameters
    ---------
    dim : {"n", "p"}, default="p"
        Axis whose elements become variables in the returned matrix.
        ``"p"`` compares data columns; ``"n"`` compares data rows.
    nan_policy : {"raise", "row-omit", "col-omit"}, default=:attr:`settings.current.data_nan_policy`
        Treatment of missing elements in the data matrix.
        After orienting the requested variables,
        `"raise"` throws a :class:`ValueError` on encountering a NaN;
        `"row-omit"` removes complete rows containing a NaN;
        `"col-omit"` removes complete columns containg a NaN.
    
    """

    _long_name = "Chow-Liu Feature Tree"
    _identifier = "graph-chow-liu"
    _signatures = (Signature.DENSITY,Signature.TOPOLOGY)
    _type = Type.DESCRIPTIVE
    _tags = (Tag.GRAPH_FOREST,
             Tag.GRAPH_WEIGHTED,
             Tag.SYMMETRICAL,
             Tag.NONPARAMETRIC)

    def __init__(
            self,
            n_neighbors: int = 3,
            ensure_adjacency_type: Literal[
                "binary",
                "binary-weighted",
                None] = None,
            zero_threshold: float | None = None,
            dim: Literal["n", "p"] = "p",
            nan_policy: Literal[
                "raise", "row-omit", "col-omit"] = settings.current.data_nan_policy,
            random_seed: int | None = None):

        self._n_neighbors = argchk.check_natural_number(n_neighbors, "n_neighbors")
        adjacency_scaling = ("minmax" if ensure_adjacency_type is not None
            else None)

        super().__init__(
            dim=dim,
            nan_policy=nan_policy,
            ensure_adjacency_type=ensure_adjacency_type,
            adjacency_scaling=adjacency_scaling,
            zero_threshold=zero_threshold)


    def _compute(self, data: np.ndarray) -> np.ndarray:
        """Return a maximum spanning tree weighted by estimated mutual information."""
        
        if self._n_neighbors >= data.shape[0]:
            raise ValueError("n_neighbors must be smaller than the number of observations.")
        
        weights = _mutual_information_matrix(
            data,
            n_neighbors=self._n_neighbors,
            random_seed=resolve_random_seed(self.random_seed),
        )
        return _maximum_spanning_tree(weights)
