"""Tests for OrthogonalPCAEEstimator data and training orchestration."""

from unittest.mock import Mock

import numpy as np
import pytest
import torch

from pyspoc.statistics.dimreduce.orthopcae import _estimator as estimator_module
from pyspoc.statistics.dimreduce.orthopcae._estimator import (
    OrthogonalPCAEEstimator,
)


@pytest.fixture
def estimator() -> OrthogonalPCAEEstimator:
    """Return a small CPU estimator suitable for isolated unit tests."""
    result = OrthogonalPCAEEstimator(
        batch_size=4,
        max_bottleneck_dim=2,
        train_steps=12,
    )
    result.device = torch.device("cpu")
    return result


def test_constructor_rejects_incompatible_annotated_types() -> None:
    """RuntimeTypeCheckedMixin should validate estimator initialization."""
    with pytest.raises(TypeError, match="batch_size"):
        OrthogonalPCAEEstimator(
            batch_size="four",  # type: ignore[arg-type]
            max_bottleneck_dim=2,
        )


def test_get_or_create_reuses_equivalent_orthopcae_estimator() -> None:
    """Equivalent constructor requests should resolve to one cached instance."""
    OrthogonalPCAEEstimator._reset_cache()
    data = np.arange(8, dtype=np.float32).reshape(4, 2)
    kwargs = {
        "batch_size": 4,
        "max_bottleneck_dim": 2,
        "train_steps": 12,
    }

    first = OrthogonalPCAEEstimator.get_or_create(data=data, **kwargs)
    second = OrthogonalPCAEEstimator.get_or_create(
        data=data.copy(),
        **kwargs,
    )

    assert second is first
    attached_data = first._get_attached_dataset()
    assert attached_data is not data
    np.testing.assert_array_equal(attached_data, data)


def test_prepare_data_casts_numpy_arrays_to_contiguous_float32(
        estimator: OrthogonalPCAEEstimator) -> None:
    """NumPy inputs should become contiguous float32 tensors."""
    data = np.arange(24, dtype=np.float64).reshape(4, 6)[:, ::2]
    assert not data.flags["C_CONTIGUOUS"]

    result = estimator._prepare_data(data)

    assert result.dtype == torch.float32
    assert result.is_contiguous()
    np.testing.assert_allclose(result.numpy(), data.astype(np.float32))


def test_prepare_data_preserves_float32_tensor_identity(
        estimator: OrthogonalPCAEEstimator) -> None:
    """Already compatible tensors should not be copied."""
    data = torch.randn(3, 2, dtype=torch.float32)

    assert estimator._prepare_data(data) is data


def test_normalize_data_records_column_statistics_and_handles_constants(
        estimator: OrthogonalPCAEEstimator) -> None:
    """Normalization should center columns without dividing by zero."""
    data = torch.tensor(
        [
            [1.0, 5.0],
            [2.0, 5.0],
            [3.0, 5.0],
        ],
    )

    result = estimator._normalize_data(data)

    torch.testing.assert_close(result.mean(dim=0), torch.zeros(2))
    assert result[:, 0].std().item() == pytest.approx(1.0)
    torch.testing.assert_close(result[:, 1], torch.zeros(3))
    assert estimator.scale_[1].item() == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("shuffle", "expected_drop_last"),
    [(True, True), (False, False)],
)
def test_prepare_loader_only_drops_partial_training_batches(
        estimator: OrthogonalPCAEEstimator,
        shuffle: bool,
        expected_drop_last: bool) -> None:
    """Partial batches should be dropped only for shuffled training data."""
    estimator.batch_size = 4
    estimator.shuffle = shuffle

    loader = estimator._prepare_loader(torch.randn(5, 2))

    assert loader.batch_size == 4
    assert loader.drop_last is expected_drop_last


@pytest.mark.parametrize(
    ("n_rows", "batch_size", "train_steps", "expected_epochs"),
    [
        (10, 4, 12, 4),
        (8, 4, 12, 6),
        (2, 10, 3, 3),
    ],
)
def test_get_train_epochs_converts_steps_to_complete_epochs(
        estimator: OrthogonalPCAEEstimator,
        n_rows: int,
        batch_size: int,
        train_steps: int,
        expected_epochs: int) -> None:
    """Requested optimizer steps should be rounded up to whole epochs."""
    estimator.batch_size = batch_size
    estimator.train_steps = train_steps

    assert estimator._get_train_epochs(n_rows) == expected_epochs


def test_get_model_requires_training(
        estimator: OrthogonalPCAEEstimator) -> None:
    """Model selection should fail before a model has been constructed."""
    with pytest.raises(ValueError, match="compute"):
        estimator._get_model("current")


def test_get_model_selects_current_or_optimal_model(
        estimator: OrthogonalPCAEEstimator) -> None:
    """Model selection should honor the requested model type."""
    current = Mock()
    optimal = Mock()
    estimator.model_ = current
    estimator.optimal_model_ = optimal

    assert estimator._get_model("current") is current
    assert estimator._get_model("optimal") is optimal


def test_compute_constructs_trains_and_extracts_statistics(
        estimator: OrthogonalPCAEEstimator,
        monkeypatch: pytest.MonkeyPatch) -> None:
    """First computation should perform the complete estimator lifecycle."""
    data = np.arange(6, dtype=np.float32).reshape(3, 2)
    model = Mock()
    model.to.return_value = model
    model_factory = Mock(return_value=model)
    train = Mock()
    expected = {"pseudo_variance_explained": np.array([0.2, 0.8])}
    extract = Mock(return_value=expected)
    monkeypatch.setattr(estimator_module, "OrthogonalPCAE", model_factory)
    monkeypatch.setattr(estimator, "_train", train)
    monkeypatch.setattr(estimator, "_compute_fitted", extract)

    result = estimator.compute(data)

    assert result is expected
    attached_data = estimator._get_attached_dataset()
    assert attached_data is not data
    np.testing.assert_array_equal(attached_data, data)
    assert estimator._pyspoc_is_fitted
    assert estimator.batch_size == 3
    assert estimator.train_epochs_ == 12
    assert estimator.burn_in_epochs_ == 2
    assert estimator.model_ is model
    model_factory.assert_called_once_with(
        2,
        2,
        random_seed=estimator.random_seed,
    )
    model.to.assert_called_once_with(estimator.device)
    train.assert_called_once_with(data, 12)
    extract.assert_called_once_with(data)
    assert estimator._get_lru() > 0.0


def test_compute_reuses_fitted_optimal_model_without_retraining(
        estimator: OrthogonalPCAEEstimator,
        monkeypatch: pytest.MonkeyPatch) -> None:
    """An estimator with an optimal model should only extract new results."""
    data = np.ones((3, 2), dtype=np.float32)
    estimator._pyspoc_is_fitted = True
    estimator._set_attached_dataset(data)
    estimator.optimal_model_ = Mock()
    train = Mock()
    expected = {"optimal_bottleneck_dimension": 2}
    extract = Mock(return_value=expected)
    monkeypatch.setattr(estimator, "_train", train)
    monkeypatch.setattr(estimator, "_compute_fitted", extract)

    result = estimator.compute(data)

    assert result is expected
    attached_data = estimator._get_attached_dataset()
    assert attached_data is not data
    np.testing.assert_array_equal(attached_data, data)
    train.assert_not_called()
    extract.assert_called_once_with(data)


def test_compute_rechecks_fitted_state_after_acquiring_training_lock(
        estimator: OrthogonalPCAEEstimator,
        monkeypatch: pytest.MonkeyPatch) -> None:
    """A caller that loses the fitting race should not train again."""
    data = np.ones((3, 2), dtype=np.float32)
    estimator._set_attached_dataset(data)
    train = Mock()
    expected = {"optimal_bottleneck_dimension": 2}
    extract = Mock(return_value=expected)

    class FittingRaceLock:
        """Simulate another thread completing fitting before lock acquisition."""

        def __enter__(self) -> None:
            estimator._pyspoc_is_fitted = True

        def __exit__(self, *args: object) -> None:
            return None

    estimator._pyspoc_fitting_lock = FittingRaceLock()  # type: ignore[assignment]
    monkeypatch.setattr(estimator, "_train", train)
    monkeypatch.setattr(estimator, "_compute_fitted", extract)

    result = estimator.compute(data)

    assert result is expected
    train.assert_not_called()
    extract.assert_called_once_with(data)


def test_train_accumulates_histories_and_records_better_model(
        estimator: OrthogonalPCAEEstimator,
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Training metrics should be accumulated and the best model retained."""
    estimator.model_ = Mock()
    estimator.train_epochs_ = 2
    estimator.burn_in_epochs_ = 1
    tensor = torch.randn(4, 2)
    loader = Mock()
    optimal = Mock()
    train_result = {
        "lambda_history": [0.1, 0.2],
        "recon_loss_history": [(0, 2.0), (1, 1.0)],
        "ortho_loss_history": [(0, 0.5), (1, 0.4)],
        "optimal_model": optimal,
        "optimal_epoch": 1,
        "optimal_epoch_loss": 1.4,
    }
    train_func = Mock(return_value=train_result)
    monkeypatch.setattr(estimator, "_prepare_data", Mock(return_value=tensor))
    monkeypatch.setattr(estimator, "_normalize_data", Mock(return_value=tensor))
    monkeypatch.setattr(estimator, "_prepare_loader", Mock(return_value=loader))
    monkeypatch.setattr(
        estimator_module.f,
        "train_adaptive_orthogonal_pcae",
        train_func,
    )

    estimator._train(np.ones((4, 2)), epochs=2)

    assert estimator.lambda_history == [0.1, 0.2]
    assert estimator.recon_loss_history == [(0, 2.0), (1, 1.0)]
    assert estimator.ortho_loss_history == [(0, 0.5), (1, 0.4)]
    assert estimator.current_epoch == 2
    assert estimator.optimal_model_ is optimal
    assert estimator.optimal_epoch_ == 1
    assert estimator.optimal_loss_ == pytest.approx(1.4)
    assert not estimator._pyspoc_is_fitted
    train_func.assert_called_once_with(
        estimator.model_,
        loader,
        estimator.device,
        current_epoch=0,
        epochs=2,
        burn_in_epochs=1,
        alpha=estimator.alpha,
    )
