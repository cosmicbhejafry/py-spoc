"""Automatic initialization hooks shared by pySPoC capability mixins."""

from __future__ import annotations

import inspect

from functools import wraps
from typing import Any


_AUTO_INITIALIZED_ATTRIBUTE = "_pyspoc_auto_initialized"
_AUTO_INITIALIZATION_DEPTH_ATTRIBUTE = "_pyspoc_auto_initialization_depth"


class AutoInitializedMixin:
    """Run cooperative capability hooks around declared constructors.

    Subclasses receive a constructor wrapper only when they declare their own
    ``__init__`` method. Inherited constructors are left untouched, preventing
    intermediate mixins from wrapping ``object.__init__`` or rebinding a
    concrete subclass call against the wrong constructor.

    When a wrapped subclass constructor calls another wrapped constructor via
    ``super()``, only the outermost constructor runs the capability hooks. Its
    signature represents the complete concrete configuration; nested base
    constructors often accept only a subset and must not overwrite it.
    """

    # Constructors temporarily shadow this default on each instance. The
    # declaration keeps the lifecycle attribute visible to type checkers
    # without retaining it after initialization completes.
    _pyspoc_auto_initialization_depth: int = 0

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Wrap a directly declared constructor with initialization hooks."""
        super().__init_subclass__(**kwargs)

        declared_init = cls.__dict__.get("__init__")

        if not inspect.isfunction(declared_init):
            return

        if getattr(declared_init, _AUTO_INITIALIZED_ATTRIBUTE, False):
            return

        @wraps(declared_init)
        def wrapped_init(self: AutoInitializedMixin,
                         *args: Any,
                         **kwargs: Any) -> None:
            signature = inspect.signature(declared_init)
            bound = signature.bind(self, *args, **kwargs)
            bound.apply_defaults()

            init_args = dict(bound.arguments)
            init_args.pop("self", None)

            depth = self._pyspoc_auto_initialization_depth
            setattr(
                self,
                _AUTO_INITIALIZATION_DEPTH_ATTRIBUTE,
                depth + 1,
            )

            try:
                if depth == 0:
                    self._before_component_init(init_args)

                declared_init(self, *args, **kwargs)

                if depth == 0:
                    self._after_component_init(init_args)
            finally:
                remaining_depth = self._pyspoc_auto_initialization_depth - 1

                if remaining_depth == 0:
                    delattr(self, _AUTO_INITIALIZATION_DEPTH_ATTRIBUTE)
                else:
                    setattr(
                        self,
                        _AUTO_INITIALIZATION_DEPTH_ATTRIBUTE,
                        remaining_depth,
                    )

        setattr(wrapped_init, _AUTO_INITIALIZED_ATTRIBUTE, True)
        cls.__init__ = wrapped_init

    def _before_component_init(
            self,
            init_args: dict[str, Any]) -> None:
        """Run capability initialization before the concrete constructor."""

    def _after_component_init(
            self,
            init_args: dict[str, Any]) -> None:
        """Run capability initialization after successful construction."""
