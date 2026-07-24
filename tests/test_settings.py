"""Tests for package-wide and context-local settings."""

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context

import pytest

from pyspoc.settings import Settings, SettingsValues


def test_settings_use_typed_defaults() -> None:
    """The current snapshot should initially be the package defaults."""
    defaults = SettingsValues(verbose=True, numba_mode="python")
    settings = Settings(defaults)

    assert settings.defaults is defaults
    assert settings.current is defaults


def test_configure_replaces_package_defaults() -> None:
    """Permanent changes should publish a new immutable defaults snapshot."""
    settings = Settings()
    original = settings.defaults

    updated = settings.configure(verbose=True, max_cache_results=20)

    assert updated is settings.defaults
    assert settings.current is updated
    assert updated.verbose is True
    assert updated.max_cache_results == 20
    assert original.verbose is False
    assert original.max_cache_results == 10


def test_override_is_context_local_and_nested() -> None:
    """Nested overrides should inherit and restore their enclosing snapshot."""
    settings = Settings()

    with settings.override(verbose=True):
        enclosing = settings.current
        assert enclosing.verbose is True
        assert enclosing.numba_mode == "auto"

        with settings.override(numba_mode="python"):
            assert settings.current.verbose is True
            assert settings.current.numba_mode == "python"

        assert settings.current is enclosing

    assert settings.current is settings.defaults


def test_override_restores_after_exception() -> None:
    """Exceptional exits should restore the exact previous snapshot."""
    settings = Settings()

    with pytest.raises(RuntimeError, match="test failure"):
        with settings.override(numba_mode="python"):
            raise RuntimeError("test failure")

    assert settings.current is settings.defaults
    assert settings.current.numba_mode == "auto"


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"unknown_setting": True}, "Unknown setting"),
        ({"verbose": "yes"}, "Invalid value"),
        ({"numba_mode": "sometimes"}, "Invalid value"),
    ],
)
def test_changes_are_validated(
        changes: dict[str, object],
        message: str) -> None:
    """Unknown names and incompatible runtime values should be rejected."""
    settings = Settings()

    with pytest.raises(TypeError, match=message):
        with settings.override(**changes):
            pass

    assert settings.current is settings.defaults


def test_thread_context_is_propagated_explicitly() -> None:
    """A copied context should carry an override into a worker thread."""
    settings = Settings()

    with ThreadPoolExecutor(max_workers=1) as executor:
        with settings.override(numba_mode="python"):
            context = copy_context()
            future = executor.submit(
                context.run,
                lambda: settings.current.numba_mode,
            )

        assert future.result() == "python"
        assert settings.current.numba_mode == "auto"
