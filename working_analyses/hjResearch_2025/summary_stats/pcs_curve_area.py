import numpy as np
from pyspoc import ReducedStatistic
from pyspoc.rstatistics import PCABase
from typing import Union


class PCAVarExplainedAreaFromNull(PCABase):

    name = "PCA - area between var explained curve and x=y"
    identifier = "pca-varExp-area"
    labels = ["linear", "dimension"]

    def __init__(self, components: list[int]):
        super().__init__(components=components)

    def _area_quad_from_vertex(self,upper_vert, lower_vert):
        # order vertices correctly
        # calculate matrix determinant
        pass

    def compute(self, data: np.ndarray) -> np.ndarray:
        """
        right now im assuming that the var exp curve is always above x=y (will depend on # comp)
        TODO maybe consider implicitly setting it to min(n_sample,n_col)!!
        """
        # returns the pca fit - TODO modify underlying class to reflect this
        pca = self._get_pca(data)
        summed_var = np.cumsum(pca.explained_variance_ratio_)
        null_curve = np.linspace(0,1,len(summed_var))        
        # indices = [i - 1 for i in self._components]

        #iteratively find quad areas

        area_val = 0

        return area_val
