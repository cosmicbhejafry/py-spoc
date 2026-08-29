import numpy as np
import warnings

from typing import Literal
from pyspoc.statistics.base import PairwiseStatistic
from pyspoc.settings import settings
from pyspoc.statistics.distance.hsic.func import pairwise_hsic

IMPORT_WARNINGS = []


class AdditiveNoiseModel(PairwiseStatistic):

    _name = "Additive Noise Model"
    _identifier = "anm"
    _labels = ["unsigned", "causal", "unordered", "linear", "directed"]

    def __init__(self):
        super().__init__(dim="p",
                         is_ordered=True,
                         symmetry_type="yes")

    @property
    def name(self) -> str:
        return self._name

    @property
    def identifier(self) -> str:
        return self._identifier

    @property
    def labels(self) -> list[str]:
        return self._labels

    # monkey-patch the anm_score function
    @staticmethod
    def corrected_anm_score(x, y):
        x_2d = x.reshape(-1, 1) if x.ndim == 1 else x

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            from sklearn.gaussian_process import GaussianProcessRegressor
            from cdt.causality.pairwise.ANM import normalized_hsic
            IMPORT_WARNINGS.extend(w)
        
        gp = GaussianProcessRegressor(
            random_state=settings.current.random_seed).fit(x_2d, y)
        y_predict = gp.predict(x_2d).reshape(-1, 1)
        
        indepscore = normalized_hsic(y_predict - y, x_2d)
        return indepscore

    anm_score = corrected_anm_score

    def pairwise_compute(self, x: np.ndarray, y: np.ndarray) -> np.ndarray | float:
        return self.anm_score(x, y)


class ConditionalDistributionSimilarity(PairwiseStatistic):

    _name = "Conditional Distribution Similarity Statistic"
    _identifier = "cds"
    _labels = ["unsigned", "causal", "unordered", "nonlinear", "directed"]

    @property
    def name(self) -> str:
        return self._name

    @property
    def identifier(self) -> str:
        return self._identifier

    @property
    def labels(self) -> list[str]:
        return self._labels

    def __init__(self):
        super().__init__(dim="p", is_ordered=False)

    def pairwise_compute(self, x: np.ndarray, y: np.ndarray) -> np.ndarray | float:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from cdt.causality.pairwise import CDS
            IMPORT_WARNINGS.extend(w)

        return float(CDS().cds_score(x, y))


class RegressionErrorCausalInference(PairwiseStatistic):

    _name = "Regression Error-based Causal Inference"
    _identifier = "reci"
    _labels = ["unsigned", "causal", "unordered", "nonlinear", "directed"]

    @property
    def name(self) -> str:
        return self._name

    @property
    def identifier(self) -> str:
        return self._identifier

    @property
    def labels(self) -> list[str]:
        return self._labels


    def __init__(self):
        super().__init__(dim="p", is_ordered=False)

    def pairwise_compute(self, x: np.ndarray, y: np.ndarray) -> np.ndarray | float:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from cdt.causality.pairwise import RECI
            IMPORT_WARNINGS.extend(w)

        return RECI().b_fit_score(x, y)


class InformationGeometricConditionalIndependence(PairwiseStatistic):

    _name = "Information-Geometric Conditional Independence"
    _identifier = "igci"
    _labels = ["causal", "directed", "nonlinear", "unsigned", "unordered"]

    @property
    def name(self) -> str:
        return self._name

    @property
    def identifier(self) -> str:
        return self._identifier

    @property
    def labels(self) -> list[str]:
        return self._labels

    def __init__(self, dim: Literal["n", "p"] = "p"):
        super().__init__(dim=dim, is_ordered=False, symmetry_type="negative")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from cdt.causality.pairwise import IGCI
            IMPORT_WARNINGS.extend(w)

        self._igci = IGCI()

    def pairwise_compute(self, x: np.ndarray, y: np.ndarray) -> np.ndarray | float:
        

        return self._igci.predict_proba((x, y))
