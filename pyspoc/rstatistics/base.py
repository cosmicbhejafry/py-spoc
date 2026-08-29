from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

from pyspoc.statistics.base import Statistic

if TYPE_CHECKING:
    from pyspoc.dataset import Dataset
    import numpy as np


class ReducedStatistic(Statistic, ABC):

    """
    Abstract Statistic class for computing a fully reduced statistical output, acting as both a Statistic and
    Reducer in a single operation.

    Output is NOT subject to further processing by applicable Reducers and only flattened, if required.
    """

    def calculate(self, dataset: Dataset) -> np.ndarray:
        # temporarily uncache results, TODO: fix caching mechanisms
        self.uncache(dataset)

        result = super().calculate(dataset)
        return result.flatten()

    @staticmethod
    def _get_component_type() -> type:
        return ReducedStatistic