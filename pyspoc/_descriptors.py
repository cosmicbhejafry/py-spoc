"""Descriptors shared by components and runtime type checking."""

from __future__ import annotations

import inspect

from collections.abc import Callable
from functools import wraps
from typing import Concatenate, Generic, ParamSpec, TypeVar, cast


_Owner = TypeVar("_Owner")
_P = ParamSpec("_P")
_R = TypeVar("_R")


class hybridmethod(Generic[_Owner, _P, _R]):
    """Bind a method's first argument to its accessing instance or class.

    Parameters
    ----------
    method : callable
        Function whose first parameter accepts either an instance or its class.

    Notes
    -----
    Unlike a class method, instance access retains the instance, allowing its
    configuration to override class defaults. Binding does not itself require
    any particular metadata fields on the receiver.
    """

    def __init__(
        self,
        method: Callable[Concatenate[_Owner | type[_Owner], _P], _R],
    ) -> None:
        """Store the function and prepare its user-facing signature.

        Parameters
        ----------
        method : callable
            Unbound function, including its instance-or-class first parameter.
        """
        self.method = method
        # Keep documentation and introspection available on the descriptor too.
        wraps(method)(self)
        signature = inspect.signature(method)
        # Binding supplies the receiver automatically; callers should not see
        # it as an argument they need to provide in completion or inspection.
        self._bound_signature = signature.replace(
            parameters=tuple(signature.parameters.values())[1:],
        )

    @property
    def __isabstractmethod__(self) -> bool:
        """Expose the underlying function's abstract-method status.

        Returns
        -------
        bool
            Whether the function is marked abstract for ABC enforcement.
        """
        # Forward the flag rather than hiding an abstract method inside a
        # descriptor, which would otherwise permit incomplete implementations.
        return bool(getattr(self.method, "__isabstractmethod__", False))

    def __get__(
        self,
        instance: _Owner | None,
        owner: type[_Owner],
    ) -> Callable[_P, _R]:
        """Bind the function to the receiver used for this attribute access.

        Parameters
        ----------
        instance : object or None
            Accessing instance, or ``None`` for access through a class.
        owner : type
            Class through which the descriptor is accessed.

        Returns
        -------
        callable
            Function accepting only the original non-receiver arguments.
        """
        # Test against None explicitly: a false-valued instance is still a
        # valid receiver and must not accidentally be replaced by its class.
        target = instance if instance is not None else owner

        # Each access gets its own closure. No shared descriptor state is
        # changed when different instances or subclasses use the method.
        @wraps(self.method)
        def bound(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            """Call the stored function with the captured receiver.

            Parameters
            ----------
            *args : Any
                Positional arguments following the receiver.
            **kwargs : Any
                Keyword arguments forwarded unchanged to the function.

            Returns
            -------
            _R
                Result of the underlying function.
            """
            return self.method(target, *args, **kwargs)

        # wraps preserves the original signature, which includes the receiver.
        # Runtime wrappers need that signature, while users of the bound method
        # should see only the parameters they actually supply.
        setattr(bound, "__signature__", self._bound_signature)
        return bound


class hybridproperty(hybridmethod[_Owner, [], _R]):
    """Evaluate a read-only getter against its accessing instance or class.

    Parameters
    ----------
    method : callable
        Getter accepting exactly one positional receiver parameter. The
        receiver is the instance for instance access and the accessing class
        for class access. No additional arguments are accepted.

    Notes
    -----
    Values are evaluated on every access, not cached. Instance assignment and
    deletion are prohibited; replacing the attribute on the class remains
    possible, as with a normal Python property.

    Class access to an abstract getter returns the descriptor itself rather
    than evaluating the getter. This preserves ABC abstractness checks in
    subclasses that have not implemented the property.
    """

    def __init__(self, method: Callable[[_Owner | type[_Owner]], _R]) -> None:
        """Store a getter after validating its receiver-only signature.

        Parameters
        ----------
        method : callable
            Unbound getter with exactly one positional receiver parameter.

        Raises
        ------
        TypeError
            If the getter does not accept exactly one positional parameter.
        """
        parameters = tuple(inspect.signature(method).parameters.values())
        if len(parameters) != 1 or parameters[0].kind not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            raise TypeError(
                "hybridproperty requires exactly one positional receiver parameter."
            )
        super().__init__(method)

    # This intentional descriptor specialization returns the result, not the
    # callable promised by hybridmethod. Retain precise value typing for users.
    def __get__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, instance: _Owner | None, owner: type[_Owner],
    ) -> _R:
        """Return the getter's value for the accessing receiver.

        Parameters
        ----------
        instance : object or None
            Accessing instance, or ``None`` for access through a class.
        owner : type
            Class through which the descriptor is accessed.

        Returns
        -------
        _R
            Getter result. For an abstract getter accessed through a class,
            the descriptor itself is returned for ABC introspection instead.
        """
        if instance is None and self.__isabstractmethod__:
            # ABCMeta inspects inherited abstract attributes using getattr().
            # Evaluating here could either raise or make an unfinished getter
            # look concrete merely because its result is not marked abstract.
            return cast(_R, self)
        return super().__get__(instance, owner)()

    def __set__(self, instance: _Owner, value: object) -> None:
        """Reject instance assignment to this read-only property.

        Parameters
        ----------
        instance : object
            Instance receiving the attempted assignment.
        value : object
            Proposed replacement value.

        Raises
        ------
        AttributeError
            Always; no setter is provided.
        """
        raise AttributeError("hybridproperty is read-only")

    def __delete__(self, instance: _Owner) -> None:
        """Reject instance deletion of this read-only property.

        Parameters
        ----------
        instance : object
            Instance receiving the attempted deletion.

        Raises
        ------
        AttributeError
            Always; no deleter is provided.
        """
        raise AttributeError("hybridproperty is read-only")
