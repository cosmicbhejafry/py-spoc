"""Distance statistics exposed through a lazy public API."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    # Visible to Pylance and other static type checkers, but not executed.
    from .basic import (
        DistanceCorrelation,
        HellerHellerGorfine,
    )
    from .hsic import HilbertSchmidtIndependenceCriterion


__all__ = [
    "DistanceCorrelation",
    "HellerHellerGorfine",
    "HilbertSchmidtIndependenceCriterion",
]


_LAZY_IMPORTS = {
    "DistanceCorrelation": (".basic", "DistanceCorrelation"),
    "HellerHellerGorfine": (".basic", "HellerHellerGorfine"),
    "HilbertSchmidtIndependenceCriterion": (
        ".hsic",
        "HilbertSchmidtIndependenceCriterion",
    ),
}


def __getattr__(name: str) -> Any:
    """Load a public object when it is first accessed."""
    try:
        module_name, attribute_name = _LAZY_IMPORTS[name]
    except KeyError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from None

    module = import_module(module_name, package=__name__)
    value = getattr(module, attribute_name)

    # Cache the resolved object so later access avoids __getattr__.
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazily exported names in interactive discovery."""
    return sorted(set(globals()) | set(__all__))
