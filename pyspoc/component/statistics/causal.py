import numpy as np
import warnings
import time

from typing import Literal
from pyspoc.component.statistics.base import PairwiseStatistic
from pyspoc.settings import settings
from pyspoc._tag import Tag, Signature, Type

# NOTE: This will be used instead of normalized_hsic when ready.
#from pyspoc.statistics.distance.hsic.func import pairwise_hsic

IMPORT_WARNINGS = []

# NOTE: Consider using causal-learn's ANM. Benchmark against existing
# implementation to see which is most efficient.
# from causallearn.search.FCMBased.ANM.ANM import ANM


class AdditiveNoiseModel(PairwiseStatistic):

    _long_name = "Additive Noise Model"
    _identifier = "anm"
    _signatures = (Signature.CAUSAL,)
    _types = (Type.DESCRIPTIVE,)
    _tags = (Tag.DIRECTED,)

    def __init__(
            self,
            dim: Literal["n", "p"] = "p",
            nan_policy: Literal[
                "raise", "row-omit", "col-omit"] = settings.current.data_nan_policy):
        
        super().__init__(
            dim=dim,
            nan_policy=nan_policy,
            is_ordered=True,
            symmetry_type="exact")

    @classmethod
    def get_long_name(cls) -> str:
        return cls._long_name

    @classmethod
    def get_identifier(cls) -> str:
        return cls._identifier

    @classmethod
    def get_signatures(cls) -> tuple[Signature, ...]:
        return cls._signatures

    @classmethod
    def get_type(cls) -> tuple[Type, ...]:
        return cls._types

    @classmethod
    def get_tags(cls) -> tuple[Tag, ...]:
        return cls._tags

    
    # monkey-patch the anm_score function
    @staticmethod
    def corrected_anm_score(x, y):
        x_2d = x.reshape(-1, 1) if x.ndim == 1 else x

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            from sklearn.gaussian_process import GaussianProcessRegressor
            from cdt.causality.pairwise.ANM import normalized_hsic
            IMPORT_WARNINGS.extend(w)
            
        pre_gp = time.perf_counter()
        gp = GaussianProcessRegressor(
            random_state=settings.current.random_seed).fit(x_2d, y)
        y_predict = gp.predict(x_2d).reshape(-1, 1) # pyright: ignore[reportAttributeAccessIssue]
        post_gp = time.perf_counter()
        print("GP:", post_gp - pre_gp)
        
        indepscore = normalized_hsic(y_predict - y, x_2d)
        post_hsic = time.perf_counter()
        print("HSIC:", post_hsic - post_gp)
        return indepscore

    anm_score = corrected_anm_score

    def _pairwise_compute(self, x: np.ndarray, y: np.ndarray
    ) -> np.ndarray | float:
        return self.anm_score(x, y)


class ConditionalDistributionSimilarity(PairwiseStatistic):

    _long_name = "Conditional Distribution Similarity Statistic"
    _identifier = "cds"
    _signatures = (Signature.CAUSAL,)
    _types = (Type.DESCRIPTIVE,)
    _tags = (Tag.DIRECTED, Tag.NONLINEAR)

    @classmethod
    def get_long_name(cls) -> str:
        return cls._long_name

    @classmethod
    def get_identifier(cls) -> str:
        return cls._identifier

    @classmethod
    def get_signatures(cls) -> tuple[Signature, ...]:
        return cls._signatures

    @classmethod
    def get_type(cls) -> tuple[Type, ...]:
        return cls._types

    @classmethod
    def get_tags(cls) -> tuple[Tag, ...]:
        return cls._tags


    def __init__(
            self,
            dim: Literal["n", "p"] = "p",
            nan_policy: Literal[
                "raise", "row-omit", "col-omit"] = settings.current.data_nan_policy):
        
        super().__init__(
            dim=dim,
            nan_policy=nan_policy,
            is_ordered=False)

    def _pairwise_compute(self, x: np.ndarray, y: np.ndarray
    ) -> np.ndarray | float:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from cdt.causality.pairwise import CDS
            IMPORT_WARNINGS.extend(w)

        return float(CDS().cds_score(x, y))


class RegressionErrorCausalInference(PairwiseStatistic):

    _long_name = "Regression Error-based Causal Inference"
    _identifier = "reci"
    _signatures = (Signature.CAUSAL,)
    _types = (Type.DESCRIPTIVE,)
    _tags = (Tag.DIRECTED, Tag.NONLINEAR)

    @classmethod
    def get_long_name(cls) -> str:
        return cls._long_name

    @classmethod
    def get_identifier(cls) -> str:
        return cls._identifier

    @classmethod
    def get_signatures(cls) -> tuple[Signature, ...]:
        return cls._signatures

    @classmethod
    def get_type(cls) -> tuple[Type, ...]:
        return cls._types

    @classmethod
    def get_tags(cls) -> tuple[Tag, ...]:
        return cls._tags

    def __init__(
            self,
            dim: Literal["n", "p"] = "p",
            nan_policy: Literal[
                "raise", "row-omit", "col-omit"] = settings.current.data_nan_policy):
        
        super().__init__(
            dim=dim,
            nan_policy=nan_policy,
            is_ordered=False)

    def _pairwise_compute(self, x: np.ndarray, y: np.ndarray
    ) -> np.ndarray | float:
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from cdt.causality.pairwise import RECI
            IMPORT_WARNINGS.extend(w)

        return RECI().b_fit_score(x, y)


class InformationGeometricConditionalIndependence(PairwiseStatistic):

    _long_name = "Information-Geometric Conditional Independence"
    _identifier = "igci"
    _signatures = (Signature.CAUSAL,)
    _types = (Type.DESCRIPTIVE,)
    _tags = (Tag.DIRECTED, Tag.NONLINEAR)

    @classmethod
    def get_long_name(cls) -> str:
        return cls._long_name

    @classmethod
    def get_identifier(cls) -> str:
        return cls._identifier

    @classmethod
    def get_signatures(cls) -> tuple[Signature, ...]:
        return cls._signatures

    @classmethod
    def get_type(cls) -> tuple[Type, ...]:
        return cls._types

    @classmethod
    def get_tags(cls) -> tuple[Tag, ...]:
        return cls._tags

    def __init__(
            self,
            dim: Literal["n", "p"] = "p",
            nan_policy: Literal[
                "raise", "row-omit", "col-omit"] = settings.current.data_nan_policy):
        
        super().__init__(
            dim=dim,
            nan_policy=nan_policy,
            is_ordered=False,
            symmetry_type="negative")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from cdt.causality.pairwise import IGCI
            IMPORT_WARNINGS.extend(w)

        self._igci = IGCI()

    def _pairwise_compute(self, x: np.ndarray, y: np.ndarray
    ) -> np.ndarray | float:
        return self._igci.predict_proba((x, y))
