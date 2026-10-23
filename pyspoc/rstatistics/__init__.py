from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from types import ModuleType

from pyspoc.rstatistics.base import ReducedStatistic


_IMPORT_ERRORS: dict[str, Exception] = {}


def _iter_modules() -> list[ModuleType]:
    """Import all modules in this package and its subpackages."""
    modules: list[ModuleType] = []
    package_root = Path(__path__[0])

    for module_path in package_root.rglob("*.py"):
        if module_path.name == "__init__.py":
            continue

        module_parts = module_path.relative_to(package_root).with_suffix("").parts
        module_name = ".".join((__name__, *module_parts))

        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            _IMPORT_ERRORS[module_name] = exc
            continue

        modules.append(module)

    return modules


def _iter_reduced_statistic_classes() -> dict[str, type[ReducedStatistic]]:
    """Find ReducedStatistic subclasses defined under this package."""
    statistic_classes: dict[str, type[ReducedStatistic]] = {}

    for module in _iter_modules():
        for class_name, class_obj in inspect.getmembers(module, inspect.isclass):
            if class_obj is ReducedStatistic:
                continue

            if class_obj.__module__ != module.__name__:
                continue

            if inspect.isabstract(class_obj):
                continue

            if issubclass(class_obj, ReducedStatistic):
                statistic_classes[class_name] = class_obj

    return statistic_classes


_REDUCED_STATISTIC_CLASSES = _iter_reduced_statistic_classes()
globals().update(_REDUCED_STATISTIC_CLASSES)

__all__ = sorted(_REDUCED_STATISTIC_CLASSES)  # pyright: ignore[reportUnsupportedDunderAll]
