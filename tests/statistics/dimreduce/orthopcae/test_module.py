"""Tests for the Orthogonal PCA autoencoder module."""

import numpy as np
import pytest
import torch

from pyspoc.statistics.dimreduce.orthopcae._module import OrthogonalPCAE


def test_initialization_constructs_expected_layer_shapes() -> None:
    """Encoder and decoder layers should reflect the input and bottleneck."""
    model = OrthogonalPCAE(input_dim=5, max_bottleneck_dim=3)

    assert model.enc_mean.in_features == 5
    assert model.enc_mean.out_features == 3
    assert model.enc_project.in_features == 64
    assert model.enc_project.out_features == 3
    assert model.decoder[-1].out_features == 5


def test_mask_pool_contains_each_active_dimension_once() -> None:
    """A refreshed mask pool should be a permutation of valid dimensions."""
    model = OrthogonalPCAE(input_dim=4, max_bottleneck_dim=4)

    assert sorted(model.mask_pool.tolist()) == [1, 2, 3, 4]


def test_pop_mask_idx_removes_and_returns_first_entry() -> None:
    """Popping a mask should consume exactly one pool entry."""
    model = OrthogonalPCAE(input_dim=3, max_bottleneck_dim=3)
    model.mask_pool = np.array([2, 1, 3])

    result = model._pop_mask_idx()

    assert result == 2
    np.testing.assert_array_equal(model.mask_pool, np.array([1, 3]))


def test_forward_preserves_batch_and_input_dimensions() -> None:
    """Reconstruction should have the same shape as its input."""
    model = OrthogonalPCAE(input_dim=5, max_bottleneck_dim=3)
    model.eval()
    data = torch.randn(4, 5)

    result = model(data, active_dim=2)

    assert result.shape == data.shape


def test_forward_clamps_requested_active_dimension() -> None:
    """Active dimensions outside the valid range should be clamped."""
    torch.manual_seed(1)
    model = OrthogonalPCAE(input_dim=4, max_bottleneck_dim=3)
    model.eval()
    data = torch.randn(5, 4)

    below_minimum = model(data, active_dim=0)
    minimum = model(data, active_dim=1)
    above_maximum = model(data, active_dim=100)
    maximum = model(data, active_dim=3)

    torch.testing.assert_close(below_minimum, minimum)
    torch.testing.assert_close(above_maximum, maximum)


def test_training_mask_refreshes_an_empty_pool(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Masked training should refresh and consume an exhausted mask pool."""
    model = OrthogonalPCAE(input_dim=3, max_bottleneck_dim=2)
    model.train()
    model.mask_pool = np.array([], dtype=int)
    monkeypatch.setattr(
        model,
        "_get_refreshed_mask_pool",
        lambda: np.array([1, 2]),
    )

    model(torch.randn(4, 3), mask=True)

    np.testing.assert_array_equal(model.mask_pool, np.array([2]))


def test_orthogonality_loss_is_zero_for_orthonormal_rows() -> None:
    """The penalty should vanish when projection rows are orthonormal."""
    model = OrthogonalPCAE(input_dim=3, max_bottleneck_dim=3)

    with torch.no_grad():
        model.enc_project.weight.zero_()
        model.enc_project.weight[:, :3] = torch.eye(3)

    assert model.get_orthogonality_loss().item() == pytest.approx(0.0)
