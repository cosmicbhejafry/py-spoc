"""OT statistics exposed through a lazy public API. This guards against optional
module import errors if importing only Wasserstein.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from pyspoc._lazy import lazy_exports


if TYPE_CHECKING:
    # Visible to Pylance and other static type checkers, but not executed.
    from .sinkhorn import (
        PairwiseSinkhornDivergence,
        PairwiseSinkhornPlanEntropy,
        PairwiseSinkhornPlanEffectiveDestinationsSummary)
    from .wasserstein import PairwiseWassersteinDistance


_LAZY_IMPORTS = {
    "PairwiseSinkhornDivergence": (".sinkhorn", "PairwiseSinkhornDivergence"),
    "PairwiseSinkhornPlanEntropy": (".sinkhorn", "PairwiseSinkhornPlanEntropy"),
    "PairwiseSinkhornPlanEffectiveDestinationsSummary": {
        ".sinkhorn", "PairwiseSinkhornPlanEffectiveDestinationsSummary"},
    "PairwiseWassersteinDistance": {".wasserstein", "PairwiseWassersteinDistance"}
}

__all__ = [
    "PairwiseSinkhornDivergence",
    "PairwiseSinkhornPlanEntropy",
    "PairwiseSinkhornPlanEffectiveDestinationsSummary",
    "PairwiseWassersteinDistance"
]
__getattr__, __dir__ = lazy_exports(globals(), _LAZY_IMPORTS)
