"""Public pySPoC API, with component packages exposed at the package root.

The component package installs aliases for both packages and nested modules,
so public and implementation imports share module, class and cache identity.
"""

from .component import (
    statistics,
    rstatistics,
    reducers
)

from .component.statistics.base import (
    Statistic, PairwiseStatistic)

from .component.rstatistics.base import ReducedStatistic
from .component.reducers.base import Reducer
from .config.config import Config
from .calculator import Calculator
from .component._base import info
from .dataset import Dataset

__all__ = [
    "statistics",
    "rstatistics",
    "reducers",
    "Statistic",
    "PairwiseStatistic",
    "ReducedStatistic",
    "Reducer",
    "Config",
    "Calculator",
    "info",
    "Dataset"]
