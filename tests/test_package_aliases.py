"""Public component imports must share implementation module/class identity."""

from __future__ import annotations

import importlib
import subprocess
import sys

import pytest


@pytest.mark.parametrize("suffix", [
    "statistics", "rstatistics", "reducers",
    "statistics.base", "rstatistics.base", "reducers.base",
    "statistics.association.distance.hsic.func",
    "statistics.clustering.kclustering.kmeans._base",
    "rstatistics.ranks", "reducers.basic",
])
def test_public_and_implementation_imports_are_the_same_module(suffix):
    public = importlib.import_module(f"pyspoc.{suffix}")
    canonical = importlib.import_module(f"pyspoc.component.{suffix}")

    assert public is canonical
    assert sys.modules[f"pyspoc.{suffix}"] is canonical
    assert public.__name__ == f"pyspoc.component.{suffix}"
    assert public.__spec__.name == public.__name__
    assert public.__package__ == canonical.__package__


def test_from_imports_and_flat_root_exports_preserve_class_identity():
    import pyspoc
    from pyspoc.statistics.base import Statistic as PublicStatistic
    from pyspoc.component.statistics.base import Statistic
    from pyspoc.rstatistics.base import ReducedStatistic as PublicReducedStatistic
    from pyspoc.component.rstatistics.base import ReducedStatistic
    from pyspoc.reducers.base import Reducer as PublicReducer
    from pyspoc.component.reducers.base import Reducer

    assert PublicStatistic is Statistic is pyspoc.Statistic
    assert PublicReducedStatistic is ReducedStatistic is pyspoc.ReducedStatistic
    assert PublicReducer is Reducer is pyspoc.Reducer


def test_concrete_estimators_share_their_cache_class():
    from pyspoc.statistics.clustering.kclustering.kmeans._base import KMeansEstimator
    from pyspoc.component.statistics.clustering.kclustering.kmeans import _base

    assert KMeansEstimator is _base.KMeansEstimator
    assert KMeansEstimator._get_cache() is _base.KMeansEstimator._get_cache()


@pytest.mark.parametrize("first_path", ["public", "canonical"])
def test_aliases_execute_a_new_module_only_once(tmp_path, monkeypatch, first_path):
    import pyspoc.component.statistics as parent

    # Extend only the test's package search path. This module does not already
    # exist in sys.modules, so both import orders exercise real loader behaviour.
    name = "_pyspoc_alias_identity_probe"
    (tmp_path / f"{name}.py").write_text(
        "import pyspoc.component.statistics as parent\n"
        "parent._alias_probe_executions += 1\n"
        "class Marker:\n"
        "    pass\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(parent, "__path__", [*parent.__path__, str(tmp_path)])
    monkeypatch.setattr(parent, "_alias_probe_executions", 0, raising=False)
    names = [f"pyspoc.statistics.{name}", f"pyspoc.component.statistics.{name}"]
    if first_path == "canonical":
        names.reverse()

    try:
        first = importlib.import_module(names[0])
        second = importlib.import_module(names[1])
        assert first is second
        assert first.Marker is second.Marker
        assert isinstance(first.Marker(), second.Marker)
        assert parent._alias_probe_executions == 1
        assert first.__spec__.name == f"pyspoc.component.statistics.{name}"

        # reload() should follow canonical metadata, rather than execute once
        # under each name or replace one alias with a different module object.
        assert importlib.reload(first) is second
        assert parent._alias_probe_executions == 2
        assert sys.modules[names[0]] is sys.modules[names[1]]
    finally:
        for module_name in names:
            sys.modules.pop(module_name, None)
        if hasattr(parent, name):
            delattr(parent, name)


def test_alias_import_does_not_hide_missing_dependencies(tmp_path, monkeypatch):
    import pyspoc.component.statistics as parent

    name = "_pyspoc_alias_missing_dependency_probe"
    missing = "_pyspoc_missing_dependency_for_alias_test"
    (tmp_path / f"{name}.py").write_text(f"import {missing}\n", encoding="utf-8")
    monkeypatch.setattr(parent, "__path__", [*parent.__path__, str(tmp_path)])
    with pytest.raises(ModuleNotFoundError) as error:
        importlib.import_module(f"pyspoc.statistics.{name}")
    assert error.value.name == missing
    assert f"pyspoc.statistics.{name}" not in sys.modules
    assert f"pyspoc.component.statistics.{name}" not in sys.modules


def test_missing_public_submodule_reports_the_requested_namespace():
    with pytest.raises(ModuleNotFoundError) as error:
        importlib.import_module("pyspoc.reducers._pyspoc_nonexistent.child")
    assert error.value.name == "pyspoc.reducers._pyspoc_nonexistent"


def test_public_statistics_is_not_the_standard_library_module():
    import statistics
    import pyspoc.statistics as public
    import pyspoc.component.statistics as canonical

    assert public is canonical
    assert public is not statistics


@pytest.mark.parametrize("suffix", ["statistics.base", "rstatistics.base", "reducers.base"])
@pytest.mark.parametrize("public_first", [True, False])
def test_both_import_orders_work_in_a_fresh_interpreter(suffix, public_first):
    # conftest has already imported pyspoc in this process. A child interpreter
    # is needed to verify the public entry point from a genuinely cold start.
    public, canonical = f"pyspoc.{suffix}", f"pyspoc.component.{suffix}"
    names = [public, canonical] if public_first else [canonical, public]
    script = (
        "import sys, importlib\n"
        "sys.modules.update(oct2py=None, jpype=None)\n"
        f"first = importlib.import_module({names[0]!r})\n"
        f"second = importlib.import_module({names[1]!r})\n"
        "assert first is second\n"
        f"assert first.__spec__.name == {canonical!r}\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_reloading_root_does_not_install_duplicate_import_finders():
    import pyspoc

    before = [
        finder for finder in sys.meta_path
        if getattr(finder, "_pyspoc_flat_component_imports", False)
    ]
    importlib.reload(pyspoc)
    after = [
        finder for finder in sys.meta_path
        if getattr(finder, "_pyspoc_flat_component_imports", False)
    ]
    assert len(before) == 1
    assert after == before
