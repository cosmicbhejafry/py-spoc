"""
Greedy Equivalence Search Causal Discovery Graph Statistics Mixin.
Acts as a thin API between back-end estimator and Statistic results.

Back-end algorithm implementations provided by Zheng, Yujia, et al.
"Causal-learn: Causal discovery in python." Journal of Machine Learning Research 25.60 (2024): 1-8.
"""
from __future__ import annotations

import numpy as np

from typing import Literal

import pyspoc._argchecking as argchk

from ._estimator import GESCausalGraphEstimator



class GESCausalGraphStatisticMixin:


    def __init__(
            self,
            score_func: Literal[
                "BIC",
                "BDeu",
                "CV_general",
                "marginal_general"] = "BIC",
            maxP: int | None = None,
            *,
            # BIC-specific parameter
            BIC_lambda: float = 0.5,

            # CV-specific parameters
            CV_k: int = 10,
            CV_lambda: float = 0.01):
        
        self._score_func = score_func
        self._maxP = (argchk.check_natural_number(maxP, arg_name="maxP")
                      if maxP is not None else None)
        self._BIC_lambda = argchk.check_float(BIC_lambda, minimum=0, arg_name="BIC_lambda")
        self._CV_k = argchk.check_natural_number(CV_k, arg_name="CV_k")
        self._CV_lambda = argchk.check_float(CV_lambda, minimum=0, arg_name="CV_lambda")


    def _get_fitted_estimator(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]]
    ) -> GESCausalGraphEstimator:

        estimator = GESCausalGraphEstimator.get_or_create(
            data=data,
            score_func = self._score_func,
            maxP = self._maxP,
            BIC_lambda = self._BIC_lambda,
            CV_k = self._CV_k,
            CV_lambda = self._CV_lambda)
        estimator.fit(data)
        return estimator
