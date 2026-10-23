"""Tests for configuration module discovery."""

from unittest.mock import patch

import pytest

from pyspoc.config import Config


@pytest.mark.parametrize(
    ("module_reference", "missing_name"),
    [
        ("pyspoc.statistics.missing", "pyspoc.statistics.missing"),
        ("pyspoc.statistics.missing.child", "pyspoc.statistics.missing"),
    ],
)
def test_get_package_module_ignores_missing_requested_path(
    module_reference: str,
    missing_name: str,
) -> None:
    """A genuinely absent requested module should remain a failed lookup."""
    error = ModuleNotFoundError(name=missing_name)

    with patch("pyspoc.config.importlib.import_module", side_effect=error):
        assert Config._get_package_module(module_reference) is None


def test_get_package_module_preserves_missing_dependency() -> None:
    """An unavailable dependency imported by a target module must propagate."""
    error = ModuleNotFoundError(name="optional_dependency")

    with (
        patch("pyspoc.config.importlib.import_module", side_effect=error),
        pytest.raises(ModuleNotFoundError) as exception_info,
    ):
        Config._get_package_module("pyspoc.statistics.clustering.kmeans")

    assert exception_info.value is error


def test_is_internal_module_preserves_missing_dependency() -> None:
    """The secondary module probe must not hide broken internal imports."""
    error = ModuleNotFoundError(name="optional_dependency")

    with (
        patch("pyspoc.config.importlib.import_module", side_effect=error),
        pytest.raises(ModuleNotFoundError) as exception_info,
    ):
        Config._is_internal_module("pyspoc.statistics.clustering.kmeans")

    assert exception_info.value is error
