from __future__ import annotations

import inspect
import numpy as np

from abc import ABC, abstractmethod
from collections.abc import Callable, Collection, Mapping
from functools import wraps
from types import MappingProxyType
from typing import final, Any, Optional, TYPE_CHECKING, TypeVar, cast
from textwrap import dedent

from pyspoc._argchecking import RuntimeTypeCheckedMixin
from pyspoc.settings import settings
from .types import NumpyFloatCubicTensorUpTo2D

if TYPE_CHECKING:
    from pyspoc.config import Config
    from pyspoc.calculator import Calculator


_INITIALIZATION_CAPTURED_ATTRIBUTE = "_pyspoc_initialization_captured"
TInit = TypeVar("TInit", bound=Callable[..., None])
TResult = TypeVar("TResult", bound=NumpyFloatCubicTensorUpTo2D)


class Component(RuntimeTypeCheckedMixin, ABC):

    _component_init_capture_depth: int = 0
    _active_calculator = None

    """
    Base object for both Statistic and Reducer components.

    Provides functionality for constructing configuration files.
    """


    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Instrument a subclass constructor to capture its normalized arguments.

        Parameters
        ----------
        **kwargs : Any, optional
            Class-creation arguments forwarded through the cooperative
            ``__init_subclass__`` chain.

        Returns
        -------
        None
            The subclass is modified in place when it declares an ``__init__``
            method of its own.

        Notes
        -----
        Constructors inherited unchanged from a parent are already
        instrumented at the point where they were declared. Wrapping only
        directly declared constructors avoids duplicate wrappers and preserves
        normal cooperative multiple-inheritance behavior.
        """
        # Let the runtime type-checking mixin process the constructor first.
        # Its wrapper preserves the original signature, so argument capture can
        # bind against the same public API seen by callers and introspection.
        super().__init_subclass__(**kwargs)

        declared_init = cls.__dict__.get("__init__")

        if not inspect.isfunction(declared_init):
            return

        if getattr(declared_init, _INITIALIZATION_CAPTURED_ATTRIBUTE, False):
            return

        cls.__init__ = cls._wrap_initialization_argument_capture(declared_init)


    @classmethod
    def _wrap_initialization_argument_capture(cls, init: TInit) -> TInit:
        """Wrap a constructor and record its normalized call arguments.

        Parameters
        ----------
        init : Callable[..., None]
            Directly declared constructor to instrument. Its signature must be
            bindable by :func:`inspect.signature`.

        Returns
        -------
        Callable[..., None]
            Signature-preserving constructor wrapper that stores the outermost
            constructor's normalized arguments after successful initialization.

        Notes
        -----
        A concrete constructor may call one or more wrapped parent
        constructors through ``super()``. A per-instance depth counter ensures
        that only the outermost call is recorded, so parent arguments cannot
        overwrite the concrete component's public initialization parameters.
        """
        signature = inspect.signature(init)

        @wraps(init)
        def wrapped_init(self: Component, *args: Any, **kwargs: Any) -> None:
            # Bind before construction so invalid call shapes raise Python's
            # usual argument error without partially updating component state.
            bound = signature.bind(self, *args, **kwargs)
            bound.apply_defaults()

            normalized_arguments = dict(bound.arguments)
            normalized_arguments.pop("self", None)

            # Several constructors may participate in one cooperative
            # initialization chain. Only the call made by the user should
            # become the component's recorded parameter set.
            depth = self._component_init_capture_depth
            self._component_init_capture_depth = depth + 1

            try:
                init(self, *args, **kwargs)

                # Record only after construction succeeds. Failed constructors
                # therefore never expose a misleading, apparently valid set of
                # initialization parameters.
                if depth == 0:
                    self._params = normalized_arguments
            finally:
                remaining_depth = self._component_init_capture_depth - 1

                if remaining_depth == 0:
                    del self._component_init_capture_depth
                else:
                    self._component_init_capture_depth = remaining_depth

        setattr(wrapped_init, _INITIALIZATION_CAPTURED_ATTRIBUTE, True)
        return cast(TInit, wrapped_init)
    

    def __init__(self, short_name: str, labels: Collection[str]):
        """Initialize metadata and mutable runtime state for a component.

        Parameters
        ----------
        short_name : str
            Concise identifier used when referring to the component.
        labels : Collection[str]
            Descriptive labels associated with the component.

        Returns
        -------
        None
            Component state is initialized in place.
        """
        self._short_name = short_name
        self._labels = labels
        self._cfg: Optional[Config] = None
        self._scheme: Optional[str] = None
        self._params: dict[str, Any] = {}
        self._cache = dict()


    def _set_active_calculator(self, calculator: Calculator):
        self._active_calculator = calculator


    @final
    def set_config(self, cfg: Config):
        self._cfg = cfg


    @final
    @property
    def cfg(self):
        return self._cfg


    @final
    def set_scheme(self, scheme: str):
        self._scheme = scheme


    @final
    @property
    def scheme(self):
        return self._scheme


    @final
    @property
    def params(self) -> Mapping[str, Any]:
        """Return a read-only view of normalized initialization arguments.

        Returns
        -------
        Mapping[str, Any]
            Constructor arguments supplied to the outermost component
            initializer, normalized to parameter names with declared defaults
            applied.

        Notes
        -----
        The mapping itself cannot be changed through this property. Values are
        retained by reference, so mutable objects stored inside the mapping are
        not recursively made immutable.
        """
        return MappingProxyType(self._params)


    @final
    @property
    def name(self) -> str:
        return self.__class__.__name__


    @final
    @property
    def short_name(self) -> str:
        return self._short_name


    @final
    @property
    def labels(self) -> Collection[str]:
        return self._labels


    @classmethod
    @abstractmethod
    def _get_component_type(cls) -> type:
        pass


    @staticmethod
    def _prepare_component_result(result: TResult) -> TResult:
        if not isinstance(result, np.ndarray):
            return result

        if settings.current.result_array_policy == "copy":
            result = np.array(result, copy=True)

        result.flags.writeable = False
        return result


    @final
    def __str__(self):
        self_type = type(self)
        full_name = _get_fully_qualified_type_name(self_type)
        cfg_name = self.cfg.name if self.cfg is not None else "None"
        component_type_name = type(self).__name__

        return dedent(
            f"""
            {component_type_name}: {full_name}
            Name: {self.name}
            Active Parameters: {self.params}
            Associated Configuration: {cfg_name}
            """)


    @final
    @classmethod
    def __info__(cls):
        full_name = _get_fully_qualified_type_name(cls)
        component_type_name = cls._get_component_type().__name__
        args = _get_obj_init_args(cls)
        required_args = list()
        optional_args = dict()

        for arg, default in args.items():
            if default:
                optional_args[arg] = default
            else:
                required_args.append(arg)

        return dedent(
            f"""
            {component_type_name}: {full_name}
            Required Parameters: {required_args}
            Optional Parameters: {optional_args}
            """)


def _info(component: Component):
    if getattr(component, "__info__"):
        print(component.__info__())


def _get_fully_qualified_type_name(type_obj: type):

    try:
        module_name = type_obj.__module__ + "."
    finally:
        pass

    if not module_name:
        return type_obj.__name__

    return module_name + type_obj.__name__


def _get_obj_init_args(type_obj: type) -> dict[str, Any]:
    """Return constructor parameter names and their declared defaults.

    Parameters
    ----------
    type_obj : type
        Component class whose public constructor signature should be inspected.

    Returns
    -------
    dict[str, Any]
        Mapping from constructor parameter names to their defaults. Required
        parameters are represented by ``None``.
    """
    signature = inspect.signature(type_obj)

    return {
        name: None if parameter.default is inspect.Parameter.empty else parameter.default
        for name, parameter in signature.parameters.items()
    }
