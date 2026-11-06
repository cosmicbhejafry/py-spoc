"""Tests for reusable package-level lazy exports."""

from importlib import import_module
from types import SimpleNamespace

import pytest

from pyspoc._lazy import lazy_exports


def test_export_setup_and_discovery_do_not_import(monkeypatch):
    def unexpected_import(*args, **kwargs):
        pytest.fail("Setting up or discovering lazy exports must not import them")

    monkeypatch.setattr("pyspoc._lazy.import_module", unexpected_import)
    namespace = {"__name__": "example.package", "existing": 1}
    _, directory = lazy_exports(namespace, {"Export": (".implementation", "Original")})
    assert directory() == ["Export", "__name__", "existing"]
    assert "Export" not in namespace


@pytest.mark.parametrize("module_name", [".implementation", "external.implementation"])
def test_resolution_uses_caller_namespace_and_preserves_identity(monkeypatch, module_name):
    original = object()
    calls = []

    def fake_import(name, package):
        calls.append((name, package))
        return SimpleNamespace(Original=original)

    monkeypatch.setattr("pyspoc._lazy.import_module", fake_import)
    namespace = {"__name__": "example.package"}
    resolve, _ = lazy_exports(namespace, {"Export": (module_name, "Original")})
    assert resolve("Export") is original
    assert namespace["Export"] is original
    assert calls == [(module_name, "example.package")]


def test_missing_export_raises_attribute_error_without_importing(monkeypatch):
    def unexpected_import(*args, **kwargs):
        pytest.fail("Unknown attributes must not cause imports")

    monkeypatch.setattr("pyspoc._lazy.import_module", unexpected_import)
    resolve, _ = lazy_exports({"__name__": "example.package"}, {})
    with pytest.raises(AttributeError, match="example.package.*Missing"):
        resolve("Missing")


def test_import_failures_propagate_and_can_be_retried(monkeypatch):
    namespace = {"__name__": "example.package"}
    resolve, _ = lazy_exports(namespace, {"Export": (".implementation", "Original")})
    error = ModuleNotFoundError("Missing optional dependency", name="optional_dependency")

    def fail_import(*args, **kwargs):
        raise error

    monkeypatch.setattr("pyspoc._lazy.import_module", fail_import)
    with pytest.raises(ModuleNotFoundError) as captured:
        resolve("Export")
    assert captured.value is error
    assert "Export" not in namespace

    original = object()
    monkeypatch.setattr("pyspoc._lazy.import_module",
                        lambda *args, **kwargs: SimpleNamespace(Original=original))
    assert resolve("Export") is original


def test_missing_implementation_attribute_is_not_cached(monkeypatch):
    namespace = {"__name__": "example.package"}
    resolve, _ = lazy_exports(namespace, {"Export": (".implementation", "Original")})
    monkeypatch.setattr("pyspoc._lazy.import_module", lambda *args, **kwargs: SimpleNamespace())
    with pytest.raises(AttributeError):
        resolve("Export")
    assert "Export" not in namespace


def test_each_calling_package_has_its_own_cache(monkeypatch):
    monkeypatch.setattr("pyspoc._lazy.import_module",
                        lambda name, package: SimpleNamespace(Original=package))
    first = {"__name__": "example.first"}
    second = {"__name__": "example.second"}
    exports = {"Export": (".implementation", "Original")}
    resolve_first, _ = lazy_exports(first, exports)
    resolve_second, _ = lazy_exports(second, exports)
    assert resolve_first("Export") == "example.first"
    assert "Export" not in second
    assert resolve_second("Export") == "example.second"


def test_distance_package_exports_quantile_from_new_location_and_caches(monkeypatch):
    package = import_module("pyspoc.statistics.association.distance")
    implementation = import_module("pyspoc.statistics.distribution.distance.quantile")
    # Remove only this export to exercise a cold lookup even if another test
    # already imported it. Normal module attribute access must bypass the hook
    # once the object is cached in the package namespace.
    if "PairwiseQuantileDistance" in vars(package):
        monkeypatch.delattr(package, "PairwiseQuantileDistance")
    assert "PairwiseQuantileDistance" in dir(package)
    assert package.PairwiseQuantileDistance is implementation.PairwiseQuantileDistance
    monkeypatch.setattr(package, "__getattr__",
                        lambda name: pytest.fail("Cached export called the lazy hook"))
    assert package.PairwiseQuantileDistance is implementation.PairwiseQuantileDistance
