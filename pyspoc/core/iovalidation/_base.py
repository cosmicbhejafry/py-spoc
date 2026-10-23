from __future__ import annotations

from abc import ABC
from types import get_original_bases
from typing import Generic, TypeVar, ClassVar, Any, get_args, get_origin
from typeguard import check_type, TypeCheckError

from pyspoc.core.types import NumpyRealTensorUpTo2D
from pyspoc.core._component import Component


TNumpyFloatTensor = TypeVar("TNumpyFloatTensor", bound=NumpyRealTensorUpTo2D)


class Validator(
    Generic[TNumpyFloatTensor],
    ABC):

    _value_type = ClassVar[Any]
    _registry: ClassVar[
        dict[object, type[Validator[NumpyRealTensorUpTo2D]]]
    ] = {}


    def __init_subclass__(
            cls,
            *,
            validates: object | None = None,
            **kwargs: Any) -> None:
        
        super().__init_subclass__(**kwargs)

        for declared_base in get_original_bases(cls):
            if get_origin(declared_base) is Validator:
                cls._value_type = get_args(declared_base)[0]
                break

        # Intermediate abstract validator classes can omit `validates`.
        if validates is None:
            return

        existing = Validator._registry.get(validates)

        if existing is not None and existing is not cls:
            raise TypeError(
                f"A validator for {validates!r} is already registered: "
                f"{existing.__qualname__}."
            )

        Validator._registry[validates] = cls


    @classmethod
    def _get_validator(
        cls,
        validated_type: type[NumpyRealTensorUpTo2D],
    ) -> type[Validator[NumpyRealTensorUpTo2D]]:
        try:
            return cls._registry[validated_type]
        except KeyError:
            raise LookupError(
                f"No validator is registered for {validated_type!r}.")


    def _validate(
            self,
            data: TNumpyFloatTensor,
            owner: Component
    ):
        """
            Internal validation method for a candidate :class:`Component` input or output data
            object handled by pySPoC's :class:`Calculator`. Provides additional context
            for any errors raised by data validation method :meth:`validate`.

            Parameters
            ----------
            data : :class:`NumpyFloatTensorUpTo2D`
                A floating point tensor of up to 2 dimensions inclusive,
                hence the following results are accepted:
                - an :class:`np.ndarray` of float dtype with 2 dimensions.
                - an :class:`np.ndarray` of float dtype with 1 dimensions.
                - an :class:`np.ndarray` of float dtype with 0 dimensions.
                - an :class:`np.floating` scalar.
                - a `float` scalar.

            owner : :class:`Component`
                The Component object that produced the data object.

            Raises
            ------
            TypeCheckError
                If ``data`` does not meet typing requirements.

            ValueError
                If ``data`` does not meet semantic requirements.
        """

        try:
            self.validate(data)

        except (TypeCheckError, ValueError) as e:
            
            if isinstance(self, Component):
                error_detail = (f"{self.designation} produced an invalid output" if self is owner
                    else f"{self.designation} received an invalid input from {owner.designation}"
                )
                             
                e.add_note(error_detail)
                raise

    
    def validate(
            self,
            data: TNumpyFloatTensor,
    ):
        """Validate a candidate :class:`Component` input or output data object

            Parameters
            ----------
            data : :class:`NumpyFloatTensorUpTo2D`
                A floating point tensor of up to 2 dimensions inclusive,
                hence the following results are accepted:
                - an :class:`np.ndarray` of float dtype with 2 dimensions.
                - an :class:`np.ndarray` of float dtype with 1 dimensions.
                - an :class:`np.ndarray` of float dtype with 0 dimensions.
                - an :class:`np.floating` scalar.
                - a `float` scalar.

            Raises
            ------
            TypeCheckError
                If ``data`` does not meet typing requirements.

            ValueError
                If ``data`` does not meet semantic requirements.

            Notes
            -----
            Subclasses can override this method with narrower jaxtyping annotations
            and/or additional semantic checks then call ``super().validate(data)``
            for basic type checking requirements.
        """
        
        check_type(data, type(self)._value_type)
