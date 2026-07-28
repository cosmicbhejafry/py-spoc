import numpy as np

from dataclasses import dataclass
from typing import Any

import numpy.typing as npt


@dataclass(slots=True)
class KMeansFittedState:
    labels: npt.NDArray[np.integer[Any]]
    cluster_centers: npt.NDArray[np.floating[Any]]
    inertia: float
    n_iter: int
