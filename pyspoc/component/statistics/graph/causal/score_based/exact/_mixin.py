"""
Bayesian Information Criterion Exact Search Causal Discovery Graph Statistics Mixin.
Acts as a thin API between back-end estimator and Statistic results.

Back-end algorithm implementations provided by Zheng, Yujia, et al.
"Causal-learn: Causal discovery in python." Journal of Machine Learning Research 25.60 (2024): 1-8.
"""
from __future__ import annotations

import numpy as np

from typing import Literal

import pyspoc._argchecking as argchk
from ._estimator import BICExactCausalGraphEstimator



class BICExactCausalGraphStatisticMixin:


    def __init__(
            self,
            search_method: Literal[
                "astar",
                "dp"] = "astar",
            use_path_extension: bool = True,
            
            *,
            use_astar_k_cycle_heuristic: bool = False,
            astar_k: int | None = 3):
        
        self._search_method = search_method
        self._use_path_extension = use_path_extension
        self._use_astar_k_cycle_heuristic = use_astar_k_cycle_heuristic
        
        if search_method == "astar" and astar_k is None:
            raise ValueError("astar_k must be provided when search_method is 'astar'.")

        self._astar_k = (
            argchk.check_natural_number(astar_k, arg_name="astar_k")
            if astar_k is not None else None)


    def _get_fitted_estimator(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]]
    ) -> BICExactCausalGraphEstimator:

        estimator = BICExactCausalGraphEstimator.get_or_create(
            data=data,
            search_method=self._search_method,
            use_path_extension=self._use_path_extension,
            use_astar_k_cycle_heuristic=self._use_astar_k_cycle_heuristic,
            astar_k=self._astar_k)
        
        estimator.fit(data)
        return estimator
