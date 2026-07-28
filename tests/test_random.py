"""Tests for shared random-seed policy and generator construction."""

import builtins

import numpy as np
import pytest
import torch

from pyspoc import OptionalDependencyMissingError
from pyspoc._random import RandomSeedMixin, resolve_random_seed
from pyspoc._random.numpy import make_numpy_generator
from pyspoc._random.torch import make_torch_generator
from pyspoc.settings import settings


class RandomComponent(RandomSeedMixin):
    """Minimal component exposing the shared seed policy."""

    def __init__(self, random_seed: int | None) -> None:
        self.initialized = True


class FrozenRandomComponent(RandomComponent):
    """Component that freezes its effective seed during construction."""

    _freeze_random_seed = True


class DefaultRandomComponent(RandomSeedMixin):
    """Component relying entirely on the package-wide random seed."""

    def __init__(self) -> None:
        self.initialized = True


def test_resolve_random_seed_uses_current_setting_by_default() -> None:
    """An omitted seed should resolve from the active settings snapshot."""
    with settings.override(random_seed=17):
        assert resolve_random_seed(None) == 17


def test_resolve_random_seed_prefers_explicit_override() -> None:
    """An explicit seed should take precedence over package settings."""
    with settings.override(random_seed=17):
        assert resolve_random_seed(23) == 23


def test_dynamic_seed_observes_context_at_access_time() -> None:
    """An unfrozen component should observe the active computation context."""
    component = RandomComponent(None)

    with settings.override(random_seed=29):
        assert component.random_seed == 29

    assert component.random_seed == settings.current.random_seed


def test_explicit_seed_remains_stable_across_contexts() -> None:
    """An explicit component override should not follow setting changes."""
    component = RandomComponent(31)

    with settings.override(random_seed=37):
        assert component.random_seed == 31


def test_frozen_default_remains_stable_across_contexts() -> None:
    """A cached component should retain its construction-time default."""
    with settings.override(random_seed=41):
        component = FrozenRandomComponent(None)

    with settings.override(random_seed=43):
        assert component.random_seed == 41


def test_mixin_uses_default_when_constructor_omits_seed_argument() -> None:
    """Inheritance alone should provide the package-wide seed."""
    component = DefaultRandomComponent()

    with settings.override(random_seed=45):
        assert component.random_seed == 45


def test_numpy_generator_is_seeded_independently() -> None:
    """Equivalent NumPy generators should produce equivalent sequences."""
    first = make_numpy_generator(47)
    second = make_numpy_generator(47)

    np.testing.assert_array_equal(
        first.integers(0, 100, size=10),
        second.integers(0, 100, size=10),
    )


def test_torch_generator_is_seeded_independently() -> None:
    """Equivalent Torch generators should produce equivalent sequences."""
    first = make_torch_generator(53)
    second = make_torch_generator(53)

    torch.testing.assert_close(
        torch.rand(10, generator=first),
        torch.rand(10, generator=second),
    )
    assert first.device.type == "cpu"


def test_mixin_reports_missing_torch_as_optional_dependency(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed lazy Torch import should raise pySPoC's specific exception."""
    original_import = builtins.__import__

    def import_without_torch(
            name: str,
            *args: object,
            **kwargs: object) -> object:
        if name == "torch":
            raise ImportError("Torch is unavailable.")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_torch)

    with pytest.raises(OptionalDependencyMissingError) as exc_info:
        RandomSeedMixin.make_torch_generator(59)

    assert exc_info.value.dependency == "torch"
    assert exc_info.value.feature == "PyTorch random-number generation"
