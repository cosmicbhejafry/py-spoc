"""Automatic initialization hooks shared by pySPoC capability mixins."""

from __future__ import annotations

import inspect

from functools import wraps
from typing import Any


_AUTO_INITIALIZED_ATTRIBUTE = "_pyspoc_auto_initialized"


class AutoInitializedMixin:
    """Run cooperative capability hooks around declared constructors.

    Subclasses receive a constructor wrapper only when they declare their own
    ``__init__`` method. Inherited constructors are left untouched, preventing
    intermediate mixins from wrapping ``object.__init__`` or rebinding a
    concrete subclass call against the wrong constructor.
    """

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

            self._before_component_init(init_args)
            declared_init(self, *args, **kwargs)
            self._after_component_init(init_args)

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
