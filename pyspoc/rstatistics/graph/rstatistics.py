"""
Causal discovery based graph ReducedStatistics.

Back-end algorithm implementations provided by Zheng, Yujia, et al.
"Causal-learn: Causal discovery in python." Journal of Machine Learning Research 25.60 (2024): 1-8.
"""

import numpy as np

from collections.abc import Collection

from pyspoc.rstatistics.base import ReducedStatistic
from pyspoc.statistics.graph.causal.score_based.ges._mixin import GESCausalGraphStatisticMixin



class GESCausalGraphInsertCount(
    GESCausalGraphStatisticMixin,
    ReducedStatistic):


    def _compute(self, data: np.ndarray[tuple[int, int], np.dtype[np.floating]]
    ) -> (np.ndarray[tuple[int], np.dtype[np.floating]] |
         np.ndarray[tuple[()], np.dtype[np.floating]] |
         np.floating |
         float):

        estimator = self._get_fitted_estimator(data)
        record = estimator._get_model()
        inserts = record.get("update1")

        if inserts is None:
            raise ValueError("No inserts found in GES fit.")

        if not isinstance(inserts, Collection):
            raise ValueError("Inserts property of GES fit does not have a length property.")

        return float(len(inserts))


class GESCausalGraphDeleteCount(
    GESCausalGraphStatisticMixin,
    ReducedStatistic):
    

    def _compute(self, data: np.ndarray[tuple[int, int], np.dtype[np.floating]]
    ) -> (np.ndarray[tuple[int], np.dtype[np.floating]] |
         np.ndarray[tuple[()], np.dtype[np.floating]] |
         np.floating |
         float):

        estimator = self._get_fitted_estimator(data)
        record = estimator._get_model()
        deletes = record.get("update2")

        if deletes is None:
            raise ValueError("No deletes found in GES fit.")

        if not isinstance(deletes, Collection):
            raise ValueError("Deletes property of GES fit does not have a length property.")

        return float(len(deletes))


class GESCausalGraphScore(
    GESCausalGraphStatisticMixin,
    ReducedStatistic):
    

    def _compute(self, data: np.ndarray[tuple[int, int], np.dtype[np.floating]]
    ) -> (np.ndarray[tuple[int], np.dtype[np.floating]] |
         np.ndarray[tuple[()], np.dtype[np.floating]] |
         np.floating |
         float):

        estimator = self._get_fitted_estimator(data)
        record = estimator._get_model()
        score = record.get("score")

        if score is None:
            raise ValueError("No score found for GES fit.")

        if isinstance(score, np.ndarray):
            score = score.item()

        if isinstance(score, float):
            raise ValueError("Score property of GES fit is not the expected floating point value.")

        return score
