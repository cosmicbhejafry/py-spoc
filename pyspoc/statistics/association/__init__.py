"""Distance statistics exposed through a lazy public API."""
from __future__ import annotations

from typing import TYPE_CHECKING

from pyspoc._lazy import lazy_exports


if TYPE_CHECKING:
    # Visible to Pylance and other static type checkers, but not executed.
    from .distance import (
        DistanceCorrelation,
        HellerHellerGorfine)
    from .hsic import HilbertSchmidtIndependenceCriterion


_LAZY_IMPORTS = {
    "DistanceCorrelation": (".distance", "DistanceCorrelation"),
    "HellerHellerGorfine": (".distance", "HellerHellerGorfine"),
    "HilbertSchmidtIndependenceCriterion": {".hsic", "HilbertSchmidtIndependenceCriterion"}
}

__all__ = [
    "DistanceCorrelation",
    "HellerHellerGorfine",
    "HilbertSchmidtIndependenceCriterion",
]
__getattr__, __dir__ = lazy_exports(globals(), _LAZY_IMPORTS)
