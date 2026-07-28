"""Tests for Orthogonal PCA autoencoder numerical helpers."""

from unittest.mock import Mock

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pyspoc.rstatistics.dimreduce.orthopcae import (
    rstatistics as rstatistics_module,
)
from pyspoc.rstatistics.dimreduce.orthopcae.rstatistics import (
    extract_pcae_scree_data,
    find_elbow_point,
)
from pyspoc.statistics.dimreduce.orthopcae._estimator import (
    OrthogonalPCAEEstimator,
)
from pyspoc.statistics.dimreduce.orthopcae._model import OrthogonalPCAE
from pyspoc.statistics.dimreduce.orthopcae._state import (
    OrthogonalPCAEFittedState,
)


class ProgressiveReconstructor(nn.Module):
    """Return a controlled reconstruction for each active dimension."""

    max_bottleneck = 3

    def forward(
            self,
            data: torch.Tensor,
            *,
            active_dim: int) -> torch.Tensor:
        if active_dim == 1:
            return torch.zeros_like(data)

        if active_dim == 2:
            return data * 0.5

        return data


class MaskTrackingOrthogonalPCAE(OrthogonalPCAE):
    """Record whether each training forward pass enables bottleneck masking."""

    def __init__(self) -> None:
        super().__init__(
            input_dim=3,
            max_bottleneck_dim=2,
            random_seed=2,
        )
        self.mask_history: list[bool] = []

    def forward(
            self,
            data: torch.Tensor,
            *,
            active_dim: int | None = None,
            mask: bool = False) -> torch.Tensor:
        self.mask_history.append(mask)
        return super().forward(data, active_dim=active_dim, mask=mask)


class FittedEstimatorStub:
    """Provide the fitted-estimator protocol required by scree extraction."""

    def __init__(self, data: np.ndarray) -> None:
        self._data = data
        self._model = ProgressiveReconstructor()

    def _get_model(self) -> ProgressiveReconstructor:
        return self._model

    def _get_model_device(self) -> torch.device:
        return torch.device("cpu")

    def _get_attached_dataset(self) -> np.ndarray:
        return self._data

    def _prepare_data(self, data: np.ndarray | torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(data, dtype=torch.float32)

    def _normalize_data(self, data: torch.Tensor) -> torch.Tensor:
        return data

    def _prepare_loader(
            self,
            data: torch.Tensor,
            device: torch.device,
            shuffle: bool) -> DataLoader:
        return DataLoader(TensorDataset(data), batch_size=1, shuffle=shuffle)


def test_find_elbow_point_returns_maximum_distance_coordinate(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """The detector should identify the point furthest from the chord."""
    fitted_state = OrthogonalPCAEFittedState(
        dimensions=np.array([1, 2, 3, 4]),
        variance_explained=np.array([0.1, 0.7, 0.85, 0.9]),
        reconstruction_loss=np.zeros(4),
        baseline_loss=1.0,
    )
    estimator = Mock()
    monkeypatch.setattr(
        rstatistics_module,
        "extract_pcae_scree_data",
        Mock(return_value=fitted_state),
    )

    assert find_elbow_point(estimator, components=(1, 2, 3, 4)) == 2


def test_extract_scree_data_reports_progressive_losses() -> None:
    """Scree extraction should aggregate batch SSE into dataset-wide MSE."""
    data = np.array([[-1.0], [1.0]], dtype=np.float32)

    result = extract_pcae_scree_data(FittedEstimatorStub(data))

    np.testing.assert_array_equal(result.dimensions, np.array([1, 2, 3]))
    np.testing.assert_allclose(
        result.reconstruction_loss,
        np.array([1.0, 0.25, 0.0]),
    )
    np.testing.assert_allclose(
        result.variance_explained,
        np.array([0.0, 0.75, 1.0]),
    )
    assert result.baseline_loss == pytest.approx(1.0)


def test_extract_scree_data_handles_zero_variance_baseline() -> None:
    """Constant data should produce finite zero pseudo-variance values."""
    data = np.zeros((4, 1), dtype=np.float32)

    result = extract_pcae_scree_data(FittedEstimatorStub(data))

    np.testing.assert_array_equal(result.variance_explained, np.zeros(3))


def test_training_returns_metric_for_each_processed_batch() -> None:
    """A small training run should populate every documented result field."""
    estimator = OrthogonalPCAEEstimator(
        batch_size=4,
        max_bottleneck_dim=2,
        train_steps=1,
        random_seed=2,
    )
    estimator._burn_in_epochs_ = 1
    model = OrthogonalPCAE(
        input_dim=3,
        max_bottleneck_dim=2,
        random_seed=2,
    )
    data = torch.randn(4, 3)
    loader = DataLoader(TensorDataset(data), batch_size=4)

    result = estimator._train_adaptive_orthogonal_pcae(
        model=model,
        data_loader=loader,
        device=torch.device("cpu"),
        epochs=1,
        alpha=0.1,
    )

    assert result["model"] is model
    assert np.isfinite(result["final_loss"])
    assert len(result["lambda_history"]) == 1
    assert len(result["recon_loss_history"]) == 1
    assert len(result["ortho_loss_history"]) == 1
    assert 1e-5 <= result["lambda_history"][0] <= 10.0
    assert estimator.current_epoch == 1


def test_additional_training_does_not_repeat_burn_in() -> None:
    """Burn-in should apply only to the estimator's initial training run."""
    estimator = OrthogonalPCAEEstimator(
        batch_size=4,
        max_bottleneck_dim=2,
        train_steps=2,
        random_seed=2,
    )
    estimator._burn_in_epochs_ = 1
    model = MaskTrackingOrthogonalPCAE()
    data = torch.randn(4, 3)
    loader = DataLoader(TensorDataset(data), batch_size=4)

    estimator._train_adaptive_orthogonal_pcae(
        model=model,
        data_loader=loader,
        device=torch.device("cpu"),
        epochs=2,
    )
    estimator._train_adaptive_orthogonal_pcae(
        model=model,
        data_loader=loader,
        device=torch.device("cpu"),
        epochs=1,
    )

    assert model.mask_history == [False, True, True]
