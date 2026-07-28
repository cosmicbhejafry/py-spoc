import numpy as np

from dataclasses import dataclass

@dataclass(slots=True)
class OrthogonalPCAEFittedState:
    variance_explained: np.ndarray
    reconstruction_loss: np.ndarray
    dimensions: np.ndarray
    baseline_loss: float
