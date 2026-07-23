from typing import Any, Union, Iterable, Optional, Literal
from numpy import ndarray
from abc import ABC, abstractmethod

from pyspoc.statistic import Statistic

from .estimator import OrthogonalPCAEEstimator


class OrthogonalPCAEStatistic(Statistic, ABC):
        
    def __init__(
        self,
        batch_size: int,
        components: Union[int, Iterable[int]],
        max_bottleneck_dim: Optional[int] = None,
        train_steps: int = 10000,
        burn_in_steps_prop: float = 0.1,
        alpha: float = 0.1,
        compute_model_type: Literal["current", "optimal"] = "optimal",
        shuffle: bool = True):

        self.batch_size = batch_size
        self.components = components
        self.max_bottleneck_dim = max_bottleneck_dim
        self.train_steps = train_steps
        self.burn_in_steps_prop = burn_in_steps_prop
        self.alpha = alpha
        self.compute_model_type = compute_model_type
        self.shuffle = shuffle
        self.estimator_ = None

    def compute(self, data: ndarray) -> ndarray | float:
        self.estimator_ = OrthogonalPCAEEstimator.get_or_create(
            data=data,
            batch_size=self.batch_size,
            components=self.components,
            max_bottleneck_dim=self.max_bottleneck_dim,
            train_steps=self.train_steps,
            burn_in_steps_prop=self.burn_in_steps_prop,
            alpha=self.alpha,
            compute_model_type=self.compute_model_type,
            shuffle=self.shuffle)
        
        results = self.estimator_.compute(data)
        return self._get_result(results)
        

    @abstractmethod
    def _get_result(self, results: dict[str, Any]) -> Union[ndarray, float]:
        pass
