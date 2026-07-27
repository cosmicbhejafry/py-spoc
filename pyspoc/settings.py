"""Package-wide defaults and context-local setting overrides."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, fields, replace
from typing import Any, Literal, cast, get_type_hints

from typeguard import TypeCheckError, check_type


NumbaMode = Literal["auto", "numba", "python"]
NumbaErrorModel = Literal["numpy", "python"]


@dataclass(frozen=True, slots=True)
class SettingsValues:
    """Represent one immutable, fully typed pySPoC settings snapshot.

    Attributes
    ----------
    max_cache_results : int, default=10
        Maximum number of results retained by caches that use the package
        setting.
    verbose : bool, default=False
        Whether supported operations should emit additional diagnostics.
    random_seed : int, default=0
        Library-wide random seed used by statistics with non-deterministic
        solvers.
    numba_error_model : {"numpy", "python"}, default="numpy"
        Error model supplied when compiling supported Numba functions.
    numba_fastmath : bool, default=False
        Whether supported Numba functions should enable fast-math
        optimizations during compilation.
    numba_boundschecking : bool, default=False
        Whether supported Numba functions should perform bounds checking.
    numba_caching : bool, default=False
        Whether Numba should cache supported compiled functions on disk.
    numba_mode : {"auto", "numba", "python"}, default="auto"
        Execution policy for functions with both Numba and Python
        implementations. ``"auto"`` permits fallback, ``"numba"`` requires
        the Numba implementation, and ``"python"`` bypasses Numba.

    Notes
    -----
    Instances are frozen so code cannot accidentally change a snapshot that
    may be visible in another execution context. Use
    :meth:`Settings.configure` to replace package defaults or
    :meth:`Settings.override` to create a temporary context-local snapshot.
    """

    max_cache_results: int = 10
    verbose: bool = False
    random_state: int = 0

    numba_error_model: NumbaErrorModel = "numpy"
    numba_fastmath: bool = False
    numba_boundschecking: bool = False
    numba_caching: bool = False
    numba_mode: NumbaMode = "auto"


class Settings:
    """Manage package defaults and context-local settings snapshots.

    The manager owns one package-wide :attr:`defaults` snapshot. Its
    :attr:`current` property returns a temporary context-local snapshot when
    an override is active, and the defaults otherwise. Consumers should
    therefore read settings through ``settings.current.<name>``.

    Parameters
    ----------
    defaults : SettingsValues, optional
        Initial package-wide defaults. A new :class:`SettingsValues` instance
        is used when this argument is omitted.

    Attributes
    ----------
    defaults : SettingsValues
        Package-wide settings used where no context-local override is active.
    current : SettingsValues
        Effective typed settings snapshot for the current execution context.

    Notes
    -----
    Context variables propagate naturally to compatible asynchronous tasks,
    but not necessarily to worker threads. A thread-pool dispatcher should
    submit each task through a separate :func:`contextvars.copy_context`
    result when the caller's overrides must be inherited.

    Examples
    --------
    Read a package default through the effective snapshot:

    >>> local_settings = Settings()
    >>> local_settings.current.numba_mode
    'auto'

    Temporarily replace one or more values:

    >>> with local_settings.override(numba_mode="python", verbose=True):
    ...     local_settings.current.numba_mode
    'python'
    >>> local_settings.current.numba_mode
    'auto'
    """

    def __init__(self, defaults: SettingsValues | None = None) -> None:
        """Initialize the settings manager.

        Parameters
        ----------
        defaults : SettingsValues, optional
            Initial package-wide settings. The default is a newly constructed
            :class:`SettingsValues` instance.

        Returns
        -------
        None
            The manager is initialized in place.
        """
        # Keep permanent defaults separate from temporary context state.
        self._defaults = defaults if defaults is not None else SettingsValues()

        # ``None`` means that this context has no override and should observe
        # the latest package-wide defaults.
        self._override: ContextVar[SettingsValues | None] = ContextVar(
            "pyspoc_settings_override",
            default=None,
        )

    @property
    def defaults(self) -> SettingsValues:
        """Return the package-wide defaults snapshot.

        Returns
        -------
        SettingsValues
            Immutable settings used by contexts without an active override.
        """
        return self._defaults

    @property
    def current(self) -> SettingsValues:
        """Return the effective settings for the current context.

        Returns
        -------
        SettingsValues
            Active context-local snapshot when one exists; otherwise the
            package-wide :attr:`defaults`.
        """
        # Context-local state takes precedence without mutating shared defaults.
        override = self._override.get()

        if override is not None:
            return override

        return self._defaults

    def configure(self, **changes: object) -> SettingsValues:
        """Permanently update package-wide default settings.

        Parameters
        ----------
        **changes : object, optional
            Setting names mapped to their new default values. Names and runtime
            types are validated against :class:`SettingsValues`.

        Returns
        -------
        SettingsValues
            New immutable package-wide defaults snapshot.

        Raises
        ------
        TypeError
            If a name is unknown or a value does not satisfy the setting's
            annotation.

        Notes
        -----
        Existing active overrides remain stable snapshots. They observe the
        new defaults after their context managers exit.
        """
        # Build and validate a replacement before publishing it so a failed
        # update cannot leave the manager partially configured.
        updated = self._replace_values(self._defaults, changes)
        self._defaults = updated
        return updated

    @contextmanager
    def override(self, **changes: object) -> Iterator[None]:
        """Temporarily override settings in the current execution context.

        Parameters
        ----------
        **changes : object, optional
            Setting names mapped to temporary values. Names and runtime types
            are validated against :class:`SettingsValues`.

        Yields
        ------
        None
            Control passes to the body of the ``with`` statement while the
            temporary snapshot is active.

        Raises
        ------
        TypeError
            If a name is unknown or a value does not satisfy the setting's
            annotation.

        Notes
        -----
        The replacement starts from :attr:`current`, so nested overrides retain
        values established by enclosing contexts. The token returned by
        :meth:`ContextVar.set` restores the exact previous snapshot even when
        the managed block raises an exception.

        Examples
        --------
        Nested overrides restore their enclosing values:

        >>> local_settings = Settings()
        >>> with local_settings.override(verbose=True):
        ...     with local_settings.override(numba_mode="python"):
        ...         assert local_settings.current.verbose
        ...     assert local_settings.current.verbose
        >>> assert not local_settings.current.verbose
        """
        # Construct the complete immutable snapshot before changing context
        # state, ensuring validation failures do not activate partial settings.
        updated = self._replace_values(self.current, changes)

        # The token records the previous value for reliable nested restoration.
        token = self._override.set(updated)

        try:
            # Pause here while the body of the ``with`` statement executes
            # under the temporary snapshot.
            yield
        finally:
            # Restore the prior snapshot on normal and exceptional exits.
            self._override.reset(token)

    @staticmethod
    def _replace_values(
            values: SettingsValues,
            changes: Mapping[str, object]) -> SettingsValues:
        """Return a validated replacement settings snapshot.

        Parameters
        ----------
        values : SettingsValues
            Snapshot on which the requested changes should be based.
        changes : Mapping[str, object]
            Setting names mapped to replacement values.

        Returns
        -------
        SettingsValues
            New immutable snapshot containing the validated changes.

        Raises
        ------
        TypeError
            If a setting name is unknown or a replacement value does not
            satisfy the field's runtime annotation.
        """
        valid_names = {settings_field.name for settings_field in fields(values)}
        unknown_names = set(changes) - valid_names

        if unknown_names:
            names = ", ".join(sorted(unknown_names))
            raise TypeError(f"Unknown setting(s): {names}")

        type_hints = get_type_hints(SettingsValues)

        for name, value in changes.items():
            try:
                # Validate each proposed value before constructing the new
                # frozen dataclass, whose annotations are not enforced by
                # dataclasses themselves.
                check_type(value, type_hints[name])
            except TypeCheckError as error:
                raise TypeError(
                    f"Invalid value for setting {name!r}: {error}"
                ) from error

        # The loop above has performed the field-specific checks dynamically.
        # Static type checkers cannot narrow a Mapping[str, object] from those
        # checks, so cast only at the dataclass replacement boundary.
        return replace(values, **cast(Any, changes))


# Export one stable manager so every importer observes the same defaults and
# resolves overrides from its own execution context.
settings = Settings()
