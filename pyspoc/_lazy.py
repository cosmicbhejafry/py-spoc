"""Reusable lazy exports for package-level public APIs."""

from collections.abc import Callable, Mapping
from importlib import import_module
from typing import Any


def lazy_exports(
    namespace: dict[str, Any],
    exports: Mapping[str, tuple[str, str]],
) -> tuple[Callable[[str], Any], Callable[[], list[str]]]:
    """Create lazy attribute and discovery hooks for a package.

    Parameters
    ----------
    namespace : dict[str, Any]
        Calling package's ``globals()`` dictionary. It must contain
        ``__name__``; resolved objects are cached directly in this dictionary.
    exports : Mapping[str, tuple[str, str]]
        Public names mapped to ``(module_name, attribute_name)`` pairs. Module
        names may be absolute or relative to the calling package.

    Returns
    -------
    Callable[[str], Any]
        Function to assign to the package's ``__getattr__``.
    Callable[[], list[str]]
        Function to assign to the package's ``__dir__``.

    Notes
    -----
    Creating these hooks imports none of the exported modules. Successful
    lookups cache the original object, preserving identity with direct imports.
    Import errors propagate unchanged and failed lookups are not cached.

    Declare ``__all__`` and explicit ``TYPE_CHECKING`` imports in each package
    to preserve star-import behaviour and static editor discovery respectively.
    Star imports resolve every name in ``__all__`` and are therefore not lazy.
    """
    package = namespace["__name__"]
    # Keep this package's export declarations stable if the caller later mutates
    # its source mapping. Only the actual package namespace must remain shared.
    export_map = dict(exports)

    def resolve(name: str) -> Any:
        """Resolve and cache an export on its first attribute lookup."""
        try:
            module_name, attribute_name = export_map[name]
        except KeyError:
            raise AttributeError(
                f"module {package!r} has no attribute {name!r}"
            ) from None

        # Relative paths must use the caller's package, not this helper's module.
        # Do not catch import errors: optional-dependency failures should retain
        # their original traceback rather than appearing as missing exports.
        module = import_module(module_name, package=package)
        value = getattr(module, attribute_name)
        namespace[name] = value
        return value

    def directory() -> list[str]:
        """List existing and lazy attributes without resolving any exports."""
        return sorted(set(namespace) | set(export_map))

    return resolve, directory
