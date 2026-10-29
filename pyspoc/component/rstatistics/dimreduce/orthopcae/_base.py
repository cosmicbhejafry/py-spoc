from __future__ import annotations

import numpy as np

from abc import abstractmethod
from typing import Literal, Iterable, TYPE_CHECKING

from pyspoc.component.rstatistics.base import ReducedStatistic
from pyspoc.component.statistics.dimreduce.orthopcae._mixin import OrthogonalPCAEMixin
from pyspoc._tag import Tag, Type, Signature
from pyspoc.settings import settings

if TYPE_CHECKING:
    from pyspoc.component.statistics.dimreduce.orthopcae._estimator import OrthogonalPCAEEstimator



class OrthogonalPCAEReducedStatistic(
    OrthogonalPCAEMixin,
    ReducedStatistic):

    _types = (Type.DESCRIPTIVE,)
    _signatures = (Signature.TOPOLOGY, Signature.DIMENSIONALITY)
    _tags = (Tag.NONLINEAR, Tag.NONPARAMETRIC)


    def __init__(
            self,
            components: int | Iterable[int],
            batch_size: int | None = None,
            train_steps: int = 10000,
            burn_in_steps_prop: float = 0.1,
            alpha: float = 0.1,
            max_bottleneck_dim: int | None = None,
            shuffle: bool = True,
            random_seed: int | None = None,
            dim: Literal["n", "p"] = "p",
            nan_policy: Literal[
                "raise", "row-omit", "col-omit"] = settings.current.data_nan_policy):

        OrthogonalPCAEMixin.__init__(
            self,
            components=components,
            batch_size=batch_size,
            train_steps=train_steps,
            burn_in_steps_prop=burn_in_steps_prop,
            alpha=alpha,
            max_bottleneck_dim=max_bottleneck_dim,
            shuffle=shuffle,
            random_seed=random_seed)

        ReducedStatistic.__init__(
            self,
            dim=dim,
            nan_policy=nan_policy)


    def _compute(self, data: np.ndarray) -> np.ndarray | np.floating | float:
        resolved_parameters = self._resolve_parameters(data)
        fitted_estimator = self._get_fitted_estimator(data, resolved_parameters)
        return self._get_result(fitted_estimator, resolved_parameters.components)
        

    @abstractmethod
    def _get_result(
        self,
        fitted_estimator: OrthogonalPCAEEstimator,
        components: tuple[int, ...]) -> np.ndarray | np.floating | float:
        pass
