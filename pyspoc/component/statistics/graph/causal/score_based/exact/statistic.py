"""
Bayesian Information Criterion Exact Search Causal Discovery Graph Statistics.

Back-end algorithm implementations provided by Zheng, Yujia, et al.
"Causal-learn: Causal discovery in python." Journal of Machine Learning Research 25.60 (2024): 1-8.
"""
from __future__ import annotations

import numpy as np

from typing import Literal

from ._mixin import BICExactCausalGraphStatisticMixin
from ..._base import CausalGraphStatistic

from pyspoc._tag import Tag, Type, Signature



class BICExactCausalGraph(
    CausalGraphStatistic,
    BICExactCausalGraphStatisticMixin):

    _long_name = "Bayesian Information Criterion Exact Score-Based Causal Discovery Graph"
    _identifier = "graph-sbcd-ges"
    _types = (Type.DESCRIPTIVE,)
    _signatures = (Signature.CAUSAL,)
    _tags = (Tag.GRAPH_DIRECTED, Tag.GRAPH_ACYCLIC)


    def __init__(
            self,
            standardise: bool = True,
            search_method: Literal[
                "astar",
                "dp"] = "astar",
            use_path_extension: bool = True,
            dim: Literal["n", "p"] = "p",
            nan_policy: Literal["raise", "row-omit", "col-omit"] = "raise",
            *,
            use_astar_k_cycle_heuristic: bool = False,
            astar_k: int | None = 3):

        CausalGraphStatistic.__init__(
            self,
            standardise=standardise,
            dim=dim,
            nan_policy=nan_policy)
        
        BICExactCausalGraphStatisticMixin.__init__(
            self,
            search_method=search_method,
            use_path_extension=use_path_extension,
            use_astar_k_cycle_heuristic=use_astar_k_cycle_heuristic,
            astar_k=astar_k)
        

    def _compute(self, data: np.ndarray[tuple[int, int], np.dtype[np.floating]]
    ) -> np.ndarray[tuple[int, int], np.dtype[np.integer]]:

        matrix = data.T if self._dim == "n" else data
        estimator = self._get_fitted_estimator(matrix)
        record = estimator._get_model()
        adjacency = record[0]

        if adjacency is None:
            raise ValueError("No graph found for BIC exact fit.")

        if not isinstance(adjacency, np.ndarray):
            raise ValueError("Graph property of BIC exact fit is not the expected numpy array.")

        if adjacency.ndim != 2:
            raise ValueError("Graph property of BIC exact fit is not a matrix.")

        if adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError("Graph property of BIC exact fit is not a square matrix.")

        return adjacency.astype(np.integer)
    

    @classmethod
    def _standardise_adjacency(
            cls,
            graph: np.ndarray[tuple[int, int], np.dtype[np.integer]]
    ) -> np.ndarray[tuple[int, int], np.dtype[np.integer]]:
        # BIC Exact produces a DAG so already has required conventions.
        return graph
