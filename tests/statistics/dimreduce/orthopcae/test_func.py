"""Tests for Orthogonal PCA autoencoder numerical helpers."""

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pyspoc.statistics.dimreduce.orthopcae._func import (
    extract_pcae_scree_data,
    _find_elbow_point,
    train_adaptive_orthogonal_pcae,
)
from pyspoc.statistics.dimreduce.orthopcae._module import OrthogonalPCAE


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


def test_find_elbow_point_returns_maximum_distance_coordinate() -> None:
    """The detector should identify the point furthest from the chord."""
    dimensions = [1, 2, 3, 4]
    losses = [10.0, 4.0, 2.5, 2.0]

    assert _find_elbow_point(dimensions, losses) == 2


def test_extract_scree_data_reports_progressive_losses() -> None:
    """Scree extraction should report losses, ratios, baseline, and elbow."""
    data = torch.tensor([[-1.0], [1.0]])

    result = extract_pcae_scree_data(ProgressiveReconstructor(), data)

    assert result["dimensions"] == [1, 2, 3]
    assert result["reconstruction_loss"] == pytest.approx([1.0, 0.25, 0.0])
    np.testing.assert_allclose(
        result["pseudo_variance_explained"],
        np.array([0.0, 0.75, 1.0]),
    )
    assert result["baseline_loss"] == pytest.approx(1.0)
    assert result["optimal_bottleneck_dimension"] == 2


def test_extract_scree_data_handles_zero_variance_baseline() -> None:
    """Constant data should produce finite zero pseudo-variance values."""
    data = torch.zeros((4, 1))

    result = extract_pcae_scree_data(ProgressiveReconstructor(), data)

    np.testing.assert_array_equal(
        result["pseudo_variance_explained"],
        np.zeros(3),
    )
    assert result["optimal_bottleneck_dimension"] == 0


def test_training_returns_metric_for_each_processed_batch() -> None:
    """A small training run should populate every documented result field."""
    torch.manual_seed(2)
    model = OrthogonalPCAE(input_dim=3, max_bottleneck_dim=2)
    data = torch.randn(4, 3)
    loader = DataLoader(TensorDataset(data), batch_size=4)

    result = train_adaptive_orthogonal_pcae(
        model=model,
        data_loader=loader,
        device=torch.device("cpu"),
        current_epoch=0,
        epochs=1,
        burn_in_epochs=1,
        alpha=0.1,
    )

    assert result["model"] is model
    assert result["optimal_model"] is model
    assert result["optimal_epoch"] == 0
    assert np.isfinite(result["optimal_epoch_loss"])
    assert len(result["lambda_history"]) == 1
    assert len(result["recon_loss_history"]) == 1
    assert len(result["ortho_loss_history"]) == 1
    assert 1e-5 <= result["lambda_history"][0] <= 10.0
