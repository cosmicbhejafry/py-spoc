import numpy as np

import pyspoc._argchecking as argchk
from typing import Literal

from pyspoc.settings import settings
from pyspoc.statistics.base import PairwiseStatistic


class PairwiseQuantileDistance(PairwiseStatistic):

    _check_p = True
    
    def __init__(
            self,
            p: float,
            dim: Literal["n", "p"] = "p",
            nan_policy: Literal[
                "raise", "row-omit", "col-omit"] = settings.current.data_nan_policy):
         
        if self._check_p:
            p = argchk.check_float(p, exclusive_minimum=0, arg_name="p")

        self._p = p

        super().__init__(
             dim=dim,
             nan_policy=nan_policy,
             is_ordered=True,
             symmetry_type="exact",
             internal_parallelism="serial",
             thread_safety="safe")


    def _pairwise_compute(
            self,
            x: np.ndarray[tuple[int], np.dtype[np.floating]],
            y: np.ndarray[tuple[int], np.dtype[np.floating]]
    ) -> np.floating | float:
         
        return np.mean(np.abs(x - y) ** self._p)
