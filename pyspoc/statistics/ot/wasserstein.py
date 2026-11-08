import numpy as np

from typing import Literal

import pyspoc._argchecking as argchk

from pyspoc.settings import settings
from pyspoc.statistics.distribution.distance import PairwiseQuantileDistance


class PairwiseWassersteinDistance(PairwiseQuantileDistance):

    def __init__(
            self,
            p: float,
            dim: Literal["n", "p"] = "p",
            nan_policy: Literal[
                "raise", "row-omit", "col-omit"] = settings.current.data_nan_policy):
                  
         p = argchk.check_float(p, minimum=1, arg_name="p")
         self._check_p = False

         super().__init__(
             p=p,
             dim=dim,
             nan_policy=nan_policy)


    def _pairwise_compute(
            self,
            x: np.ndarray[tuple[int], np.dtype[np.floating]],
            y: np.ndarray[tuple[int], np.dtype[np.floating]]
    ) -> np.floating | float:
         
         Q_p = super()._pairwise_compute(x,y)

        # Application of outer exponent grants Wasserstein distance for p >= 1.
         return Q_p ** (1.0 / self._p)
