"""Public exception types raised by pySPoC."""

from __future__ import annotations


class OptionalDependencyMissingError(ImportError):
    """Indicate that a requested feature requires an unavailable dependency.

    Parameters
    ----------
    dependency : str
        Import or distribution name of the missing optional dependency.
    feature : str or None, default=None
        Feature that requires the dependency, if a more specific description
        is useful to the caller.
    install_hint : str or None, default=None
        Optional installation guidance appended to the exception message.

    Attributes
    ----------
    dependency : str
        Import or distribution name supplied by the raising code.
    feature : str or None
        Feature that could not be used.
    install_hint : str or None
        Installation guidance supplied by the raising code.

    Notes
    -----
    This exception subclasses :class:`ImportError`, so existing handlers for
    failed imports remain compatible. Callers that want to distinguish an
    optional dependency from other import failures can catch this type more
    specifically.
    """

    def __init__(
            self,
            dependency: str,
            *,
            feature: str | None = None,
            install_hint: str | None = None) -> None:
        self.dependency = dependency
        self.feature = feature
        self.install_hint = install_hint

        message = f"Optional dependency {dependency!r} is required"
        if feature is not None:
            message += f" to use {feature}"
        message += "."

        if install_hint is not None:
            message += f" {install_hint}"

        super().__init__(message, name=dependency)
