"""Reusable runtime type checking for ``pyspoc`` classes.

The module provides :class:`RuntimeTypeCheckedMixin`, which instruments
selected subclass callables at class-creation time and validates their
annotations with :mod:`typeguard`.
"""

from __future__ import annotations

import inspect
import logging

from collections.abc import Callable
from functools import wraps
from typing import Any, Optional, Literal, ParamSpec, TypeVar, cast, get_type_hints
from numbers import Real
from typeguard import CollectionCheckStrategy, TypeCheckError, check_type

from pyspoc import _base
from pyspoc.settings import settings

_P = ParamSpec("_P")
_R = TypeVar("_R")
_TRealNumber = TypeVar("_TRealNumber", bound=Real)
_TYPE_CHECKED_ATTRIBUTE = "_pyspoc_type_checked"
_MISSING_TYPE_HINT = object()
_LOGGER = logging.getLogger(__name__)


class RuntimeTypeCheckedMixin:
    """Automatically validate annotated constructor and method calls.

    Constructors declared directly by a subclass are always type checked.
    Additional methods can be selected by name or by configuring all public
    or all non-dunder methods. Property accessors can be checked independently
    of ordinary methods.

    At call time, each wrapper binds the supplied positional and keyword
    arguments to the original signature. Annotated values are then validated
    with :func:`typeguard.check_type`; unannotated values are left unchanged.
    Variadic positional and keyword arguments are checked item by item. When
    enabled, an annotated return value is also checked after the callable
    completes.

    Attributes
    ----------
    _type_check_methods : set[str] or {"public", "all"}, optional
        Additional methods to check. An explicit set checks only those method
        names, ``"public"`` checks methods whose names do not begin with an
        underscore, and ``"all"`` checks public and private non-dunder methods.
        The default is ``"public"``.
    _type_check_returns : bool, optional
        Whether to validate annotated return values. The default is ``True``.
    _type_check_properties : bool, optional
        Whether to check annotated property getters, setters, and deleters.
        The default is ``True``.
    _type_check_collections : CollectionCheckStrategy, optional
        Typeguard strategy controlling collection item validation. The default
        is :attr:`CollectionCheckStrategy.ALL_ITEMS`.

    Notes
    -----
    Only members declared directly on a subclass are wrapped. Inherited
    members were checked when their defining class was created, avoiding
    repeated wrapping throughout an inheritance hierarchy.

    Type hints are resolved lazily on the first invocation of a wrapped
    callable. This permits annotations that refer to the subclass itself,
    which may not yet be available in its module namespace while
    ``__init_subclass__`` is running.

    This mixin performs structural validation only. Semantic requirements,
    such as permitted numerical ranges or relationships between arguments,
    must still be enforced by the implementing class. Ordinary attribute
    assignment is not intercepted unless it passes through a checked property
    setter.

    Examples
    --------
    Configure a class to check all of its public methods:

    >>> class Record(RuntimeTypeCheckedMixin):
    ...     _type_check_methods = "public"
    ...
    ...     def __init__(self, name: str) -> None:
    ...         self.name = name
    ...
    ...     def rename(self, name: str) -> None:
    ...         self.name = name
    ...
    >>> record = Record("original")
    >>> record.rename("updated")
    """

    _type_check_methods: set[str] | Literal["public", "all"] = "public"
    _type_check_returns = True
    _type_check_properties = True
    _type_check_collections = CollectionCheckStrategy.ALL_ITEMS

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Apply configured runtime checks to a newly created subclass.

        Parameters
        ----------
        **kwargs : Any, optional
            Keyword arguments forwarded to the next ``__init_subclass__``
            implementation in the method resolution order.

        Returns
        -------
        None
            The subclass is modified in place.

        Notes
        -----
        Only attributes in the subclass's own namespace are processed.
        ``staticmethod`` and ``classmethod`` descriptors are unwrapped before
        instrumentation and reconstructed afterward so their binding behavior
        is retained.
        """
        super().__init_subclass__(**kwargs)

        # Work from a snapshot because wrapped descriptors are assigned back
        # onto the class while this loop is running. This also prevents newly
        # installed wrappers from being encountered during the same pass.
        declared_members = tuple(cls.__dict__.items())

        for member_name, member in declared_members:
            # Constructors are the primary use case and are always checked,
            # independently of the additional method-selection policy.
            if member_name == "__init__" and inspect.isfunction(member):
                setattr(
                    cls,
                    member_name,
                    cls._wrap_type_checked_callable(member, member_name),
                )
                continue

            if isinstance(member, property):
                # A property owns up to three separate callables, so rebuild
                # the descriptor with individually wrapped accessors.
                if cls._type_check_properties:
                    setattr(
                        cls,
                        member_name,
                        cls._wrap_type_checked_property(member, member_name),
                    )
                continue

            if not cls._should_type_check_method(member_name):
                continue

            if isinstance(member, staticmethod):
                # Preserve static binding after wrapping the stored function.
                wrapped = cls._wrap_type_checked_callable(
                    member.__func__,
                    member_name,
                )
                setattr(cls, member_name, staticmethod(wrapped))
                continue

            if isinstance(member, classmethod):
                # Preserve class binding after wrapping the stored function.
                wrapped = cls._wrap_type_checked_callable(
                    member.__func__,
                    member_name,
                )
                setattr(cls, member_name, classmethod(wrapped))
                continue

            if inspect.isfunction(member):
                # Plain functions retain normal instance-method binding when
                # the wrapper is assigned back onto the class.
                setattr(
                    cls,
                    member_name,
                    cls._wrap_type_checked_callable(member, member_name),
                )

    @classmethod
    def _should_type_check_method(cls, method_name: str) -> bool:
        """Return whether a declared method is selected for checking.

        Parameters
        ----------
        method_name : str
            Name of the method being considered.

        Returns
        -------
        bool
            ``True`` when the configured method policy selects
            ``method_name``; otherwise ``False``.

        Notes
        -----
        The constructor is excluded here because it is handled unconditionally
        by :meth:`__init_subclass__`. The ``"all"`` policy excludes dunder
        methods but includes private, non-dunder methods.
        """
        if method_name == "__init__":
            return False

        configured_methods = cls._type_check_methods

        if configured_methods == "public":
            return not method_name.startswith("_")

        if configured_methods == "all":
            return not (method_name.startswith("__") and method_name.endswith("__"))

        return method_name in configured_methods

    @classmethod
    def _wrap_type_checked_property(cls, prop: property, property_name: str) -> property:
        """Return a property with each available accessor type checked.

        Parameters
        ----------
        prop : property
            Property descriptor whose accessors should be wrapped.
        property_name : str
            Name of the property, used to identify its accessors in validation
            errors.

        Returns
        -------
        property
            A new descriptor containing wrapped versions of the available
            getter, setter, and deleter. Missing accessors remain ``None`` and
            the original property docstring is preserved.
        """
        # Each accessor has its own signature and annotations. Wrap them
        # separately before reconstructing the descriptor.
        getter = (
            cls._wrap_type_checked_callable(
                prop.fget,
                f"{property_name}.getter",
            )
            if prop.fget is not None
            else None
        )
        setter = (
            cls._wrap_type_checked_callable(
                prop.fset,
                f"{property_name}.setter",
            )
            if prop.fset is not None
            else None
        )
        deleter = (
            cls._wrap_type_checked_callable(
                prop.fdel,
                f"{property_name}.deleter",
            )
            if prop.fdel is not None
            else None
        )

        return property(getter, setter, deleter, prop.__doc__)

    @classmethod
    def _wrap_type_checked_callable(
        cls, func: Callable[_P, _R], callable_name: str
    ) -> Callable[_P, _R]:
        """Wrap a callable with configured argument and return checks.

        Parameters
        ----------
        func : Callable[_P, _R]
            Callable to instrument. For static and class methods, this is the
            function extracted from the original descriptor.
        callable_name : str
            Name used to identify the callable in validation errors.

        Returns
        -------
        Callable[_P, _R]
            A wrapper that preserves the callable's metadata and static
            signature. A callable already wrapped by this mixin is returned
            unchanged.

        Raises
        ------
        TypeError
            Raised when the returned wrapper is invoked with an incompatible
            annotated argument, or produces an incompatible annotated return
            value while return checking is enabled.

        Notes
        -----
        Annotations are resolved with :func:`typing.get_type_hints` on first
        invocation and cached for subsequent calls. Consequently, errors
        caused by unresolved forward references also occur on first use rather
        than during class creation.
        """
        # Avoid wrapping the same callable repeatedly in inheritance or
        # descriptor-reconstruction scenarios.
        if getattr(func, _TYPE_CHECKED_ATTRIBUTE, False):
            return func

        signature = inspect.signature(func)
        resolved_hints: dict[str, Any] | None = None

        def get_hints() -> dict[str, Any]:
            """Resolve and cache the callable's evaluated type hints."""
            nonlocal resolved_hints

            if resolved_hints is None:
                # Resolve lazily: the subclass name may not yet be present in
                # its module while ``__init_subclass__`` is executing.
                resolved_hints = get_type_hints(func)

            return resolved_hints

        def check_arguments(args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            """Validate arguments supplied to a single invocation.

            Parameters
            ----------
            args : tuple[Any, ...]
                Positional arguments supplied to the wrapped callable.
            kwargs : dict[str, Any]
                Keyword arguments supplied to the wrapped callable.

            Returns
            -------
            None

            Raises
            ------
            TypeError
                If signature binding fails or an annotated argument does not
                satisfy its resolved annotation.
            """
            # Binding normalizes positional and keyword calls into the same
            # parameter mapping. Applying defaults ensures explicitly supplied
            # and defaulted values follow identical validation paths.
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            type_hints = get_hints()

            for argument_name, argument_value in bound.arguments.items():
                expected_type = type_hints.get(
                    argument_name,
                    _MISSING_TYPE_HINT,
                )

                if expected_type is _MISSING_TYPE_HINT:
                    # Unannotated arguments are intentionally ignored.
                    continue

                parameter = signature.parameters[argument_name]

                if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
                    # ``*args: T`` annotates each collected positional value.
                    values = argument_value
                elif parameter.kind is inspect.Parameter.VAR_KEYWORD:
                    # ``**kwargs: T`` annotates each collected keyword value.
                    values = argument_value.values()
                else:
                    values = (argument_value,)

                for value in values:
                    cls._check_value_type(
                        value,
                        expected_type,
                        value_description=(
                            f"argument {argument_name!r} for {cls.__qualname__}.{callable_name}"
                        ),
                    )

        def check_return(return_value: Any) -> None:
            """Validate a return value when return checking is enabled.

            Parameters
            ----------
            return_value : Any
                Value produced by the wrapped callable.

            Returns
            -------
            None

            Raises
            ------
            TypeError
                If the value does not satisfy an available return annotation.
            """
            if not cls._type_check_returns:
                return

            return_type = get_hints().get("return", _MISSING_TYPE_HINT)

            if return_type is _MISSING_TYPE_HINT:
                return

            cls._check_value_type(
                return_value,
                return_type,
                value_description=(f"return value from {cls.__qualname__}.{callable_name}"),
            )

        # The branches necessarily have different runtime callable shapes:
        # calling the async wrapper produces a coroutine, whereas calling the
        # sync wrapper produces its result directly. Keep the temporary broad
        # and restore the original generic contract at the return boundary.
        wrapper: Callable[..., Any]

        if inspect.iscoroutinefunction(func):
            # Await coroutine completion before validating its resolved result.

            @wraps(func)
            async def async_wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
                check_arguments(args, kwargs)
                return_value = await func(*args, **kwargs)
                check_return(return_value)
                return return_value

            wrapper = async_wrapper

        else:
            # Synchronous results are available for checking immediately after
            # the underlying callable returns.

            @wraps(func)
            def sync_wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
                check_arguments(args, kwargs)
                return_value = func(*args, **kwargs)
                check_return(return_value)
                return return_value

            wrapper = sync_wrapper

        # Mark the wrapper so later class instrumentation can recognize it.
        setattr(wrapper, _TYPE_CHECKED_ATTRIBUTE, True)

        # ``inspect.iscoroutinefunction()`` determines the correct wrapper at
        # runtime, but static type checkers cannot use that result to relate
        # the async branch's coroutine type back to the original callable's
        # generic signature. Both wrappers preserve that signature through
        # ``ParamSpec`` and ``functools.wraps``, so make the common contract
        # explicit at this boundary.
        return cast(Callable[_P, _R], wrapper)

    @classmethod
    def _check_value_type(cls, value: Any, expected_type: Any, value_description: str) -> None:
        """Validate one value and translate Typeguard errors consistently.

        Parameters
        ----------
        value : Any
            Runtime value to validate.
        expected_type : Any
            Resolved annotation accepted by :func:`typeguard.check_type`.
        value_description : str
            Description of the value included in any resulting error message.

        Returns
        -------
        None
            The method returns normally when ``value`` satisfies
            ``expected_type``.

        Raises
        ------
        TypeError
            If Typeguard reports that ``value`` does not satisfy
            ``expected_type``. The original
            :class:`typeguard.TypeCheckError` is retained as the exception
            cause.
        """
        try:
            check_type(
                value,
                expected_type,
                collection_check_strategy=cls._type_check_collections,
            )
        except TypeCheckError as error:
            # Expose the conventional exception type expected by callers while
            # retaining Typeguard's detailed diagnostic through chaining.
            raise TypeError(f"Invalid {value_description}: {error}") from error


def check_natural_number(arg_value: int, arg_name: Optional[str] = None) -> int:

    return check_integer(arg_value=arg_value, minimum=1, arg_name=arg_name)


def check_integer(
    arg_value: int,
    minimum: float | None = None,
    exclusive_minimum: float | None = None,
    maximum: float | None = None,
    exclusive_maximum: float | None = None,
    arg_name: str | None = None,
) -> int:
    """Check integer argument corresponds to required bounds."""

    name = arg_name or _base.retrieve_arg_name(arg_value) or "value"

    if isinstance(arg_value, bool) or not isinstance(arg_value, int):
        raise TypeError(f"{name} must be an integer, not {type(arg_value).__name__}")

    check_real(
        arg_value=arg_value,
        minimum=minimum,
        exclusive_minimum=exclusive_minimum,
        maximum=maximum,
        exclusive_maximum=exclusive_maximum,
        arg_name=arg_name,
    )

    return arg_value


def try_check_natural_number(arg_value: int, arg_name: str) -> bool:

    try:
        check_natural_number(arg_value, arg_name)
        return True
    except (TypeError, ValueError):
        return False


def check_real(
    arg_value: _TRealNumber,
    minimum: float | None = None,
    exclusive_minimum: float | None = None,
    maximum: float | None = None,
    exclusive_maximum: float | None = None,
    arg_name: Optional[str] = None,
) -> _TRealNumber:

    name = arg_name or _base.retrieve_arg_name(arg_value) or "value"

    if isinstance(arg_value, bool) or not isinstance(arg_value, Real):
        raise TypeError(f"{name} must be a real number, not {type(arg_value).__name__}.")

    if minimum is not None and exclusive_minimum is not None:
        raise ValueError("Only one of minimum and exclusive_minimum may be provided.")

    if maximum is not None and exclusive_maximum is not None:
        raise ValueError("Only one of maximum and exclusive_maximum may be provided.")

    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError("minimum cannot be greater than maximum")

    if minimum is not None and exclusive_maximum is not None and minimum >= exclusive_maximum:
        raise ValueError("minimum cannot be greater than or equal to exclusive maximum")

    if exclusive_minimum is not None and maximum is not None and exclusive_minimum >= maximum:
        raise ValueError("exclusive minimum cannot be greater than or equal to maximum")

    if (
        exclusive_minimum is not None
        and exclusive_maximum is not None
        and exclusive_minimum >= exclusive_maximum
    ):
        raise ValueError("exclusive minimum cannot be greater than or equal to exclusive maximum")

    within_lower = True
    within_upper = True

    if minimum is not None:
        within_lower = arg_value >= minimum
        interval = f"[{minimum},"

    elif exclusive_minimum is not None:
        within_lower = arg_value > exclusive_minimum
        interval = f"({exclusive_minimum},"

    else:
        interval = "(-inf,"

    if maximum is not None:
        within_upper = arg_value <= maximum
        interval += f"{maximum}]"

    elif exclusive_maximum is not None:
        within_upper = arg_value < exclusive_maximum
        interval += f"{exclusive_maximum})"

    else:
        interval += "inf)"

    if not within_lower or not within_upper:
        raise ValueError(
            f"{name} should be a value within bounds {interval}, received {arg_value}."
        )

    return arg_value


def check_float(
    arg_value: Real,
    minimum: float | None = None,
    exclusive_minimum: float | None = None,
    maximum: float | None = None,
    exclusive_maximum: float | None = None,
    arg_name: Optional[str] = None,
) -> float:
    """Validate and normalize a continuous numeric argument.

    Parameters
    ----------
    arg_value : numbers.Real
        Value to validate and convert to ``float``.
    minimum : float | None, default=None
        Inclusive lower bound accepted for ``arg_value``.
    exclusive_minimum : float | None, default=None
        Exclusive lower bound accepted for ``arg_value``.
    maximum : float | None, default=None
        Inclusive upper bound accepted for ``arg_value``.
    exclusive_maximum : float | None, default=None
        Exclusive upper bound accepted for ``arg_value``.
    arg_name : str, optional
        Name included in validation errors. When omitted, the helper attempts
        to recover the caller's argument name.

    Returns
    -------
    float
        Validated value normalized to the built-in ``float`` type.

    Raises
    ------
    TypeError
        If ``arg_value`` is not a non-Boolean real number.
    ValueError
        If ``arg_value`` is not strictly within the supplied bounds.
    """
    return float(
        check_real(
            arg_value,
            minimum=minimum,
            exclusive_minimum=exclusive_minimum,
            maximum=maximum,
            exclusive_maximum=exclusive_maximum,
            arg_name=arg_name,
        )
    )


def try_check_real(
    arg_value: Real,
    minimum: float | None = None,
    exclusive_minimum: float | None = None,
    maximum: float | None = None,
    exclusive_maximum: float | None = None,
    arg_name: Optional[str] = None,
) -> bool:

    try:
        check_real(
            arg_value,
            minimum=minimum,
            exclusive_minimum=exclusive_minimum,
            maximum=maximum,
            exclusive_maximum=exclusive_maximum,
            arg_name=arg_name,
        )
        return True
    except (TypeError, ValueError):
        return False


def clip_integer(
    arg_value: int,
    minimum: int | None = None,
    maximum: int | None = None,
    arg_name: str | None = None,
) -> int:
    """Clip and normalize an integer argument."""

    name = arg_name or _base.retrieve_arg_name(arg_value) or "value"

    if isinstance(arg_value, bool):
        raise TypeError(f"{name} must be an integer, not bool")

    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError("lower_bound cannot be greater than upper_bound")

    clipped = arg_value

    if minimum is not None:
        clipped = max(clipped, minimum)

    if maximum is not None:
        clipped = min(clipped, maximum)

    if clipped != arg_value and settings.current.verbose:
        _LOGGER.info(
            "%s was clipped from %r to %r; accepted bounds are [%r, %r].",
            name,
            arg_value,
            clipped,
            minimum,
            maximum,
        )

    return clipped


def clip_float(
    arg_value: Real,
    minimum: Real | None = None,
    maximum: Real | None = None,
    arg_name: str | None = None,
) -> float:
    """Clip and normalize a continuous numeric argument."""

    name = arg_name or _base.retrieve_arg_name(arg_value) or "value"

    if isinstance(arg_value, bool):
        raise TypeError(f"{name} must be a real number, not bool")

    value = float(arg_value)
    lower = float(minimum) if minimum is not None else None
    upper = float(maximum) if maximum is not None else None

    if lower is not None and upper is not None and lower > upper:
        raise ValueError("lower_bound cannot be greater than upper_bound")

    clipped = value

    if lower is not None:
        clipped = max(clipped, lower)

    if upper is not None:
        clipped = min(clipped, upper)

    if clipped != value and settings.current.verbose:
        _LOGGER.info(
            "%s was clipped from %r to %r; accepted bounds are [%r, %r].",
            name,
            value,
            clipped,
            lower,
            upper,
        )

    return clipped
