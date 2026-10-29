"""
Greedy Equivalence Search Causal Discovery Graph Statistics Estimator.
Estimator handles model fit and exposes results for supporting causal discovery based graph
Statistics.

Back-end algorithm implementations provided by Zheng, Yujia, et al.
"Causal-learn: Causal discovery in python." Journal of Machine Learning Research 25.60 (2024): 1-8.
"""
from __future__ import annotations

import numpy as np

from typing import Any, Dict, Literal

import pyspoc._argchecking as argchk
from pyspoc._estimators import LazyFittedCachedEstimatorMixin



class GESCausalGraphEstimator(
    LazyFittedCachedEstimatorMixin[Dict[str, Any]]):


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


    def _fit_estimator(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]]
    ) -> Dict[str, Any]:
        
        from causallearn.search.ScoreBased.GES import ges
        cv_params = {}

        if self._score_func == "CV_general":
            cv_params["kfold"] = self._CV_k,
            cv_params["lambda"] = self._CV_lambda

        lambda_value = self._BIC_lambda if self._score_func == "BIC" else None
        score_func = "local_score_" + self._score_func
        
        # or customized parameters
        Record = ges(
            data,
            score_func = score_func,
            maxP = self._maxP,
            lambda_value = lambda_value,
            parameters=cv_params)

        return Record
