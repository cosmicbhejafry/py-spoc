import numpy as np

from typing import Literal

from pyspoc.settings import settings
from pyspoc.rstatistics.base import ReducedStatistic

from ripser import ripser
import numpy as np
from persim import plot_diagrams
from persim.persistent_entropy import *
from persim.landscapes import (
    PersLandscapeApprox,
    average_approx,
    snap_pl,
    plot_landscape,
    plot_landscape_simple,
)

class PersistentEntropyLandscapeApprox(ReducedStatistic):

    _name = "TopologyRstatistics"
    _identifier = "topology"
    _labels = ["vector"]

    STATISTICS_DICT = {}
    
    STATISTICS_DICT["persistent_entropy_h0"] = (["p_h0"])
    STATISTICS_DICT["persistent_entropy_h1"] = (["p_h1"])
    STATISTICS_DICT["persistent_landscape_h1_L1_norm"] = (["p_landscape_h1_L1"])
    STATISTICS_DICT["persistent_landscape_h1_L2_norm"] = (["p_landscape_h1_L2"])
    STATISTICS_DICT["persistent_landscape_h1_sup_norm"] = (["p_landscape_h1_sup"])
    
    def __init__(
            self,
            dim: Literal["n", "p"] = "p",
            nan_policy: Literal[
                "raise", "row-omit", "col-omit"] = settings.current.data_nan_policy,
            stats=None):
        
        self.stats = stats
        unknown = set(self.stats) - set(self.STATISTICS_DICT.keys())
        if len(unknown) > 0:
            raise ValueError(f"unknown stats {unknown}\nsupported list- {self.STATISTICS_DICT.keys()}")
        
        super().__init__(dim=dim, nan_policy=nan_policy)

    @property
    def name(self) -> str:
        return self._name
    
    
    @property
    def identifier(self) -> str:
        return self._identifier
    
    @property
    def tags(self) -> list[str]:
        return self._labels

    def _compute(self, data: np.ndarray) -> np.ndarray | float:

        dgms = ripser(data, maxdim=1)["dgms"]

        h0 = dgms[0]
        h1 = dgms[1]
        pers_landsc = PersLandscapeApprox(dgms=dgms, hom_deg=1)

        return_lst = []
        for stat_val in self.stats:
            
            if stat_val == "persistent_entropy_h0":
                return_lst.append(persistent_entropy([h0])[0])
                
            if stat_val == "persistent_entropy_h1":
                return_lst.append(persistent_entropy([h1])[0])
                
            if stat_val == "persistent_landscape_h1_L1_norm":
                return_lst.append(pers_landsc.p_norm(p=1))

            if stat_val == "persistent_landscape_h1_L2_norm":
                return_lst.append(pers_landsc.p_norm(p=2))

            if stat_val == "persistent_landscape_h1_sup_norm":
                return_lst.append(pers_landsc.sup_norm())
                
        # pers_entropy_lst = [persistent_entropy([h0])[0] , persistent_entropy([h1])[0]]
        # # https://www.jmlr.org/papers/volume16/bubenik15a/bubenik15a.pdf
        # pers_landsc_lst = [pers_landsc.p_norm(p=1),pers_landsc.p_norm(p=2), pers_landsc.sup_norm()]
        # return_lst = pers_entropy_lst + pers_landsc_lst
        
        return np.array(return_lst)
        