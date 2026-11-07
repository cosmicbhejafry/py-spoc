import numpy as np

from pyspoc._tag import Tag, Type, Signature
from pyspoc.statistics.clustering._mixin import ClusteringMixin
from pyspoc.statistics.clustering.kclustering._base import KClusteringEstimator


class KClusteringInertiaMixin(ClusteringMixin):

    _tags = (Tag.NONLINEAR, Tag.NONPARAMETRIC, Tag.CLUSTERING)
    _signatures = (Signature.BLOBBINESS,)
    _type = Type.ERROR
    
    
    def _get_result(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]],
            fitted_estimator: KClusteringEstimator) -> float | np.floating:
                
        return fitted_estimator._compute_inertia()
