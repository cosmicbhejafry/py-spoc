from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import TYPE_CHECKING

from ._base import OrthogonalPCAEReducedStatistic
from pyspoc._base import copy_array
from pyspoc.statistics.dimreduce.orthopcae._state import OrthogonalPCAEFittedState

if TYPE_CHECKING:
    from pyspoc.statistics.dimreduce.orthopcae._estimator import OrthogonalPCAEEstimator


class OrthogonalPCAEVarianceExplainedRatio(OrthogonalPCAEReducedStatistic):

    _name = "Orthogonal Principal Component Autoencoder - Variance Explained Ratio"
    _identifier = "opcae-var"
    _labels = ["vector", "non-linear"]

    @property
    def name(self) -> str:
        return self._name

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(self._labels)

    @property
    def identifier(self) -> str:
        return self._identifier

    def _get_result(
            self,
            fitted_estimator: OrthogonalPCAEEstimator,
            components: tuple[int, ...]) -> np.ndarray | float:

        fitted_state = extract_pcae_scree_data(fitted_estimator=fitted_estimator)
        var_explained = fitted_state.variance_explained
        indices = np.asarray(components, dtype=int) - 1
        return var_explained[indices]


class OrthogonalPCAEVarianceElbow(OrthogonalPCAEReducedStatistic):

    _name = "Orthogonal Principal Component Autoencoder Variance Explained Elbow"
    _identifier = "opcae-var-elbow"
    _labels = ["scalar", "non-linear"]

    @property
    def name(self) -> str:
        return self._name

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(self._labels)

    @property
    def identifier(self) -> str:
        return self._identifier

    def _get_result(
            self,
            fitted_estimator: OrthogonalPCAEEstimator,
            components: tuple[int, ...]) -> np.ndarray | float:

        return float(
            find_elbow_point(
                fitted_estimator=fitted_estimator,
                components=components))


def extract_pcae_scree_data(
        fitted_estimator: OrthogonalPCAEEstimator) -> OrthogonalPCAEFittedState:
    """
    Computes a progressive reconstruction loss distribution.
    Analogous to PCA's cumulative variance explained curve.
    """
    sse_loss_fn = nn.MSELoss(reduction="sum")
    model = fitted_estimator._get_model()
    model_device = fitted_estimator._get_model_device()
    data = fitted_estimator._get_attached_dataset()
    if data is None:
        raise RuntimeError("The fitted estimator has no attached dataset.")

    data_tensor = fitted_estimator._prepare_data(data)
    data_loader = fitted_estimator._prepare_loader(data_tensor, model_device, False)
    normalized_data = fitted_estimator._normalize_data(data_tensor)
    dimensions = list(range(1, model.bottleneck + 1))
    loss_distribution = []
    mse_divisor = normalized_data.numel()

    with torch.inference_mode():

        data_mean = normalized_data.mean(dim=0, keepdim=True)
        baseline_mse = sse_loss_fn(
            data_mean.expand_as(normalized_data),
            normalized_data).item() / mse_divisor

        # Calculate loss at each progressive dimension isolation step
        for d in dimensions:
            sse = 0

            for batch in data_loader:
                batch = batch[0].to(model_device)
                batch_recon = model(batch, active_dim=d)
                sse += sse_loss_fn(batch, batch_recon).item()

            mse = sse / mse_divisor
            loss_distribution.append(mse)

    # Get baseline variance of the dataset (reconstructing with 0 dimensions)
    # This acts similarly to the total sum of squares (TSS).
    #baseline_mse = loss_distribution[0]

    # Calculate pseudo "Variance Explained" percentage
    # R^2 style: (Baseline Error - Bottleneck Error) / Baseline Error
    if np.isclose(baseline_mse, 0.0):
        cum_variance_explained = np.zeros(
            len(loss_distribution),
            dtype=float,
        )
    else:
        cum_variance_explained = np.array([
            max(0.0, baseline_mse - mse) / baseline_mse
            for mse in loss_distribution
        ])

    variance_explained = np.diff(cum_variance_explained, prepend=0)

    return OrthogonalPCAEFittedState(
        variance_explained = copy_array(variance_explained),
        dimensions = copy_array(dimensions),
        reconstruction_loss = copy_array(loss_distribution),
        baseline_loss = baseline_mse)


def find_elbow_point(
        fitted_estimator: OrthogonalPCAEEstimator,
        components: tuple[int, ...]) -> int:
    """Standard maximum distance geometric elbow detector."""

    fitted_state = extract_pcae_scree_data(fitted_estimator)
    indices = np.asarray(components, dtype=int) - 1
    selected_dims, selected_var_expl = \
        fitted_state.dimensions[indices], fitted_state.variance_explained[indices]
    n_points = min(selected_dims.shape[0], selected_var_expl.shape[0])

    if n_points == 0:
        raise ValueError("Cannot find elbow point in empty data.")

    if n_points <= 2:
        return int(selected_dims[np.argmax(selected_var_expl)])

    if np.all(np.isclose(selected_var_expl, 0.0)):
        return 0

    coords = np.vstack((selected_dims, selected_var_expl)).T
    first_pt, last_pt = coords[0], coords[-1]
    line_vec = last_pt - first_pt
    line_vec_norm = line_vec / np.linalg.norm(line_vec)

    vec_from_first = coords - first_pt
    scalar_product = np.sum(vec_from_first * line_vec_norm, axis=1)
    vec_to_line = vec_from_first - np.outer(scalar_product, line_vec_norm)

    distances = np.linalg.norm(vec_to_line, axis=1)
    return int(selected_dims[np.argmax(distances)])
