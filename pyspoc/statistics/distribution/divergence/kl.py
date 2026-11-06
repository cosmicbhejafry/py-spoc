import numpy as np

from typing import Literal
from scipy.special import kl_div

from pyspoc.settings import settings
from pyspoc.statistics.base import PairwiseStatistic


class KullbackLeiblerDivergence(PairwiseStatistic):


    def __init__(
            self,
            dim: Literal["n", "p"] = "p",
            nan_policy: Literal[
                "raise", "row-omit", "col-omit"] = settings.current.data_nan_policy):
        
        super().__init__(
            dim=dim,
            nan_policy=nan_policy,
            is_ordered=False,
            symmetry_type=None,
            internal_parallelism="serial")


    def _pairwise_compute(
            self,
            x: np.ndarray[tuple[int], np.dtype[np.floating]],
            y: np.ndarray[tuple[int], np.dtype[np.floating]]
    ) -> np.floating:

        return kl_div(x,y)
