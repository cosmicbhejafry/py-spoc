"""Detached fitted-state containers for OrthogonalPCAE results."""

import numpy as np

from dataclasses import dataclass


@dataclass(slots=True)
class OrthogonalPCAEFittedState:
    """Store detached scree information derived from a fitted model.

    Parameters
    ----------
    variance_explained : numpy.ndarray
        Pseudo-variance-explained value for each bottleneck dimension.
    reconstruction_loss : numpy.ndarray
        Mean reconstruction loss for each bottleneck dimension.
    dimensions : numpy.ndarray
        One-based bottleneck dimensions corresponding to the result arrays.
    baseline_loss : float
        Reconstruction loss obtained from the feature-wise mean baseline.

    Notes
    -----
    The arrays are copied by the extraction routine before construction, so
    consumers may modify this state without mutating the fitted estimator.
    """

    variance_explained: np.ndarray
    reconstruction_loss: np.ndarray
    dimensions: np.ndarray
    baseline_loss: float
