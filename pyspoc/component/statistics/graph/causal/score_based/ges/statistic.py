"""
Greedy Equivalence Search Causal Discovery Graph Statistics

Back-end algorithm implementations provided by Zheng, Yujia, et al.
"Causal-learn: Causal discovery in python." Journal of Machine Learning Research 25.60 (2024): 1-8.
"""
from __future__ import annotations

import numpy as np

from typing import Literal

from ._mixin import GESCausalGraphStatisticMixin
from ..._base import CausalGraphStatistic

from pyspoc._tag import Tag, Type, Signature



class GESCausalGraph(
    GESCausalGraphStatisticMixin,
    CausalGraphStatistic):

    _long_name = "Greedy Equivalence Search Score-Based Causal Discovery Graph"
    _identifier = "graph-sbcd-ges"
    _types = (Type.DESCRIPTIVE,)
    _signatures = (Signature.CAUSAL,)
    _tags = (Tag.GRAPH_MIXED, Tag.SIGNED)


    def __init__(
            self,
            standardise: bool = True,
            score_func: Literal[
                "BIC",
                "BDeu",
                "CV_general",
                "marginal_general"] = "BIC",
            maxP: int | None = None,
            dim: Literal["n", "p"] = "p",
            nan_policy: Literal["raise", "row-omit", "col-omit"] = "raise",
            *,
            # BIC-specific parameter
            BIC_lambda: float = 0.5,
            # CV-specific parameters
            CV_k: int = 10,
            CV_lambda: float = 0.01):

        CausalGraphStatistic.__init__(
            self,
            standardise=standardise,
            dim=dim,
            nan_policy=nan_policy)
        
        GESCausalGraphStatisticMixin.__init__(
            self,
            score_func=score_func,
            maxP=maxP,
            BIC_lambda=BIC_lambda,
            CV_k=CV_k,
            CV_lambda=CV_lambda)
        

    def _compute(self, data: np.ndarray[tuple[int, int], np.dtype[np.floating]]
    ) -> np.ndarray[tuple[int, int], np.dtype[np.integer]]:

        matrix = data.T if self._dim == "n" else data
        estimator = self._get_fitted_estimator(matrix)
        record = estimator._get_model()
        adjacency = record.get("G")

        if adjacency is None:
            raise ValueError("No graph found for GES fit.")

        if not isinstance(adjacency, np.ndarray):
            raise ValueError("Graph property of GES fit is not the expected numpy array.")

        if adjacency.ndim != 2:
            raise ValueError("Graph property of GES fit is not a matrix.")

        if adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError("Graph property of GES fit is not a square matrix.")

        return adjacency.astype(np.integer)
    

    @classmethod
    def _standardise_adjacency(
            cls,
            adjacency: np.ndarray[tuple[int, int], np.dtype[np.integer]]
    ) -> np.ndarray[tuple[int, int], np.dtype[np.integer]]:
        
        p = adjacency.shape[0]
        
        for i in range(p):
            for j in range(i+1, p):

                # Switch X_i -> X_j directed edge encoding to DAG convention.
                if adjacency[i,j] == -1 and adjacency[j,i] == 1:
                    adjacency[j,i] = 0
                    adjacency[i,j] = 1

                # Switch X_j -> X_i directed edge encoding to DAG convention.
                if adjacency[j,i] == -1 and adjacency[i,j] == 1:
                    adjacency[i,j] = 0
                    adjacency[j,i] = 1

        return adjacency
