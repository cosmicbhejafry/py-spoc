"""
Bayesian Information Criterion Exact Search Causal Discovery Graph Statistics Estimator.
Estimator handles model fit and exposes results for supporting causal discovery based graph
Statistics.

Back-end algorithm implementations provided by Zheng, Yujia, et al.
"Causal-learn: Causal discovery in python." Journal of Machine Learning Research 25.60 (2024): 1-8.
"""
from __future__ import annotations

import numpy as np

from typing import Any, Dict, Literal

from pyspoc.settings import settings
from pyspoc._estimators import LazyFittedCachedEstimatorMixin



class BICExactCausalGraphEstimator(
    LazyFittedCachedEstimatorMixin[
        tuple[np.ndarray[tuple[int, int]], Dict[str, Any]]]):


    def __init__(
            self,
            search_method: Literal[
                "astar",
                "dp"],
            use_path_extension: bool,
            
            *,
            use_astar_k_cycle_heuristic: bool,
            astar_k: int | None):

        self._search_method = search_method
        self._use_path_extension = use_path_extension
        self._use_astar_k_cycle_heuristic = use_astar_k_cycle_heuristic
        self._astar_k = astar_k


    def _fit_estimator(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]]
    ) -> tuple[np.ndarray[tuple[int, int]], Dict[str, Any]]:
        
        from causallearn.search.ScoreBased.ExactSearch import bic_exact_search
        astar_params = {}

        if self._search_method == "astar":
            astar_params["use_k_cycle_heuristic"] = self._use_astar_k_cycle_heuristic
            astar_params["k"] = self._astar_k

        # or customized parameters
        Record = bic_exact_search(
            data,
            search_method=self._search_method,
            use_path_extension=self._use_path_extension,
            verbose=settings.current.verbose,
            **astar_params)

        return Record
