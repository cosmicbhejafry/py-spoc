"""Tests for pySPoC's public exception types."""

from pyspoc import OptionalDependencyMissingError


def test_optional_dependency_error_is_a_structured_import_error() -> None:
    """Callers should be able to catch and inspect missing dependencies."""
    error = OptionalDependencyMissingError(
        "torch",
        feature="model training",
        install_hint="Install pySPoC with the 'extended' extra.",
    )

    assert isinstance(error, ImportError)
    assert error.dependency == "torch"
    assert error.name == "torch"
    assert error.feature == "model training"
    assert error.install_hint == "Install pySPoC with the 'extended' extra."
    assert str(error) == (
        "Optional dependency 'torch' is required to use model training. "
        "Install pySPoC with the 'extended' extra."
    )
