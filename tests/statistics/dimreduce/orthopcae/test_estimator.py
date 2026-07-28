"""Tests for OrthogonalPCAEEstimator data and training orchestration."""

from unittest.mock import Mock, call

import numpy as np
import pytest
import torch

from pyspoc.statistics.dimreduce.orthopcae import _estimator as estimator_module
from pyspoc.statistics.dimreduce.orthopcae._estimator import (
    OrthogonalPCAEEstimator,
)
from pyspoc.settings import settings


@pytest.fixture
def estimator() -> OrthogonalPCAEEstimator:
    """Return a small CPU estimator suitable for isolated unit tests."""
    result = OrthogonalPCAEEstimator(
        batch_size=4,
        max_bottleneck_dim=2,
        train_steps=12,
    )
    result._training_device = torch.device("cpu")
    return result


def test_constructor_rejects_incompatible_annotated_types() -> None:
    """RuntimeTypeCheckedMixin should validate estimator initialization."""
    with pytest.raises(TypeError, match="batch_size"):
        OrthogonalPCAEEstimator(
            batch_size="four",  # type: ignore[arg-type]
            max_bottleneck_dim=2,
        )


def test_constructor_uses_current_random_seed_by_default() -> None:
    """The active package setting should seed a directly created estimator."""
    with settings.override(random_seed=17):
        estimator = OrthogonalPCAEEstimator(
            batch_size=4,
            max_bottleneck_dim=2,
        )

    assert estimator.random_seed == 17
    assert estimator._loader_rng.initial_seed() == 17


def test_constructor_random_seed_overrides_current_setting() -> None:
    """An explicit estimator seed should take precedence over the setting."""
    with settings.override(random_seed=17):
        estimator = OrthogonalPCAEEstimator(
            batch_size=4,
            max_bottleneck_dim=2,
            random_seed=23,
        )

    assert estimator.random_seed == 23
    assert estimator._loader_rng.initial_seed() == 23


def test_constructor_state_is_exposed_through_read_only_properties(
        estimator: OrthogonalPCAEEstimator) -> None:
    """Constructor and training state should be observable but not rebindable."""
    assert estimator.batch_size == 4
    assert estimator.shuffle
    assert estimator.train_steps == 12
    assert estimator.burn_in_steps_prop == pytest.approx(0.1)
    assert estimator.max_bottleneck_dim == 2
    assert estimator.alpha == pytest.approx(0.1)
    assert estimator.random_seed == settings.current.random_seed
    assert estimator.current_epoch == 0
    assert estimator.training_device == torch.device("cpu")
    assert estimator.train_epochs is None
    assert estimator.burn_in_epochs is None
    assert estimator.mean is None
    assert estimator.scale is None
    assert estimator.final_loss is None

    with pytest.raises(AttributeError):
        estimator.batch_size = 8  # type: ignore[misc]


def test_history_properties_return_immutable_snapshots(
        estimator: OrthogonalPCAEEstimator) -> None:
    """History access should not expose the estimator's mutable lists."""
    estimator._lambda_history.extend([0.1, 0.2])
    estimator._recon_loss_history.append((0, 1.5))
    estimator._ortho_loss_history.append((0, 0.5))

    lambda_history = estimator.lambda_history
    recon_history = estimator.recon_loss_history
    ortho_history = estimator.ortho_loss_history
    estimator._lambda_history.append(0.3)

    assert lambda_history == (0.1, 0.2)
    assert recon_history == ((0, 1.5),)
    assert ortho_history == ((0, 0.5),)


def test_tensor_and_generator_properties_return_defensive_copies(
        estimator: OrthogonalPCAEEstimator) -> None:
    """Mutable tensors and generator state should not leak through properties."""
    estimator._mean_ = torch.tensor([1.0, 2.0])
    estimator._scale_ = torch.tensor([3.0, 4.0])

    mean = estimator.mean
    scale = estimator.scale
    loader_rng = estimator.loader_rng

    assert mean is not None
    assert scale is not None
    mean[0] = 99.0
    scale[0] = 99.0
    torch.testing.assert_close(estimator._mean_, torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(estimator._scale_, torch.tensor([3.0, 4.0]))
    assert loader_rng is not estimator._loader_rng
    torch.testing.assert_close(
        loader_rng.get_state(),
        estimator._loader_rng.get_state(),
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


def test_get_or_create_distinguishes_effective_setting_seeds() -> None:
    """Different active default seeds should produce different cache entries."""
    OrthogonalPCAEEstimator._reset_cache()
    data = np.arange(8, dtype=np.float32).reshape(4, 2)
    kwargs = {
        "batch_size": 4,
        "max_bottleneck_dim": 2,
        "train_steps": 12,
    }

    with settings.override(random_seed=17):
        first = OrthogonalPCAEEstimator.get_or_create(data=data, **kwargs)

    with settings.override(random_seed=23):
        second = OrthogonalPCAEEstimator.get_or_create(data=data, **kwargs)

    assert second is not first
    assert first.random_seed == 17
    assert second.random_seed == 23


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
    assert estimator._scale_[1].item() == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("shuffle", "expected_drop_last"),
    [(True, True), (False, False)],
)
def test_prepare_loader_only_drops_partial_training_batches(
        estimator: OrthogonalPCAEEstimator,
        shuffle: bool,
        expected_drop_last: bool) -> None:
    """Partial batches should be dropped only for shuffled training data."""
    estimator._batch_size = 4
    estimator._shuffle = shuffle

    loader = estimator._prepare_loader(
        torch.randn(5, 2),
        torch.device("cpu"),
        shuffle,
    )

    assert loader.batch_size == 4
    assert loader.drop_last is expected_drop_last


@pytest.mark.parametrize(
    ("n_rows", "batch_size", "train_steps", "expected_epochs"),
    [
        (10, 4, 12, 6),
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
    estimator._batch_size = batch_size
    estimator._train_steps = train_steps

    assert estimator._get_train_epochs(n_rows) == expected_epochs


def test_get_train_epochs_counts_partial_unshuffled_batch(
        estimator: OrthogonalPCAEEstimator) -> None:
    """Unshuffled training should count the partial batch it retains."""
    estimator._batch_size = 4
    estimator._train_steps = 12
    estimator._shuffle = False

    assert estimator._get_train_epochs(10) == 4


def test_get_model_requires_training(
        estimator: OrthogonalPCAEEstimator) -> None:
    """Model selection should fail before a model has been constructed."""
    with pytest.raises(ValueError, match="fit"):
        estimator._get_model()


def test_get_model_returns_fitted_model(
        estimator: OrthogonalPCAEEstimator) -> None:
    """Model access should return the estimator's fitted model."""
    model = Mock()
    estimator._model_ = model

    assert estimator._get_model() is model


def test_fit_constructs_trains_and_returns_estimator(
        estimator: OrthogonalPCAEEstimator,
        monkeypatch: pytest.MonkeyPatch) -> None:
    """First fitting should construct, train, and finalize the model."""
    data = np.arange(6, dtype=np.float32).reshape(3, 2)
    model = Mock()
    model.to.return_value = model
    model_factory = Mock(return_value=model)
    train = Mock()
    monkeypatch.setattr(estimator_module, "OrthogonalPCAE", model_factory)
    monkeypatch.setattr(estimator, "_train", train)

    result = estimator.fit(data)

    assert result is estimator
    attached_data = estimator._get_attached_dataset()
    assert attached_data is not data
    np.testing.assert_array_equal(attached_data, data)
    assert estimator._pyspoc_is_fitted
    assert estimator._batch_size == 3
    assert estimator._train_epochs_ == 12
    assert estimator.burn_in_epochs == 2
    assert estimator._model_ is model
    model_factory.assert_called_once_with(
        2,
        2,
        random_seed=estimator.random_seed,
    )
    assert model.to.call_args_list == [
        call(estimator._training_device),
        call(torch.device("cpu")),
    ]
    train.assert_called_once_with(data, 12)
    model.eval.assert_called_once_with()
    assert estimator._get_lru() > 0.0


def test_fit_reuses_fitted_model_without_retraining(
        estimator: OrthogonalPCAEEstimator,
        monkeypatch: pytest.MonkeyPatch) -> None:
    """A fitted estimator should validate its data without training again."""
    data = np.ones((3, 2), dtype=np.float32)
    estimator._pyspoc_is_fitted = True
    estimator._set_attached_dataset(data)
    train = Mock()
    monkeypatch.setattr(estimator, "_train", train)

    result = estimator.fit(data)

    assert result is estimator
    attached_data = estimator._get_attached_dataset()
    assert attached_data is not data
    np.testing.assert_array_equal(attached_data, data)
    train.assert_not_called()


def test_compute_rechecks_fitted_state_after_acquiring_training_lock(
        estimator: OrthogonalPCAEEstimator,
        monkeypatch: pytest.MonkeyPatch) -> None:
    """A caller that loses the fitting race should not train again."""
    data = np.ones((3, 2), dtype=np.float32)
    estimator._set_attached_dataset(data)
    train = Mock()

    class FittingRaceLock:
        """Simulate another thread completing fitting before lock acquisition."""

        def __enter__(self) -> None:
            estimator._pyspoc_is_fitted = True

        def __exit__(self, *args: object) -> None:
            return None

    estimator._pyspoc_fitting_lock = FittingRaceLock()  # type: ignore[assignment]
    monkeypatch.setattr(estimator, "_train", train)

    result = estimator.fit(data)

    assert result is estimator
    train.assert_not_called()


def test_train_accumulates_histories_and_records_final_loss(
        estimator: OrthogonalPCAEEstimator,
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Training metrics and the final loss should be retained."""
    estimator._model_ = Mock()
    estimator._train_epochs_ = 2
    estimator._burn_in_epochs_ = 1
    loader = Mock()
    train_result = {
        "lambda_history": [0.1, 0.2],
        "recon_loss_history": [(0, 2.0), (1, 1.0)],
        "ortho_loss_history": [(0, 0.5), (1, 0.4)],
        "final_loss": 1.4,
    }
    train_func = Mock(return_value=train_result)
    monkeypatch.setattr(
        estimator,
        "_get_model_device",
        Mock(return_value=torch.device("cpu")),
    )
    monkeypatch.setattr(estimator, "_prepare_loader", Mock(return_value=loader))
    monkeypatch.setattr(estimator, "_train_adaptive_orthogonal_pcae", train_func)

    estimator._train(np.ones((4, 2)), epochs=2)

    assert estimator._lambda_history == [0.1, 0.2]
    assert estimator._recon_loss_history == [(0, 2.0), (1, 1.0)]
    assert estimator._ortho_loss_history == [(0, 0.5), (1, 0.4)]
    assert estimator.final_loss == pytest.approx(1.4)
    assert not estimator._pyspoc_is_fitted
    train_func.assert_called_once_with(
        estimator._model_,
        loader,
        estimator._training_device,
        epochs=2,
        alpha=estimator._alpha,
    )
