from __future__ import annotations

import asyncio
import inspect

import pytest

from pyspoc._argchecking import RuntimeTypeCheckedMixin


class ConstructorChecked(RuntimeTypeCheckedMixin):
    def __init__(self, count: int, labels: tuple[str, ...] = ()) -> None:
        self.count = count
        self.labels = labels


class SelectedMethodsChecked(RuntimeTypeCheckedMixin):
    _type_check_methods = {"update", "make"}

    def update(self, value: int, *extras: str, **metadata: float) -> str:
        return str(value)

    def unchecked(self, value: int) -> int:
        return value

    @staticmethod
    def make(value: int) -> str:
        return str(value)


class PublicMethodsChecked(RuntimeTypeCheckedMixin):
    _type_check_methods = "public"

    def public(self, value: int) -> int:
        return value

    def _private(self, value: int) -> int:
        return value

    @classmethod
    def construct(cls, value: int) -> PublicMethodsChecked:
        return cls()


class PropertyChecked(RuntimeTypeCheckedMixin):
    def __init__(self) -> None:
        self._value = 0

    @property
    def value(self) -> int:
        return self._value

    @value.setter
    def value(self, value: int) -> None:
        self._value = value


class BadReturnChecked(RuntimeTypeCheckedMixin):
    _type_check_methods = "public"

    def value(self) -> int:
        return "not an integer"  # type: ignore[return-value]


class ReturnCheckingDisabled(RuntimeTypeCheckedMixin):
    _type_check_methods = "public"
    _type_check_returns = False

    def value(self) -> int:
        return "not an integer"  # type: ignore[return-value]


class AsyncChecked(RuntimeTypeCheckedMixin):
    _type_check_methods = {"double"}

    async def double(self, value: int) -> int:
        return value * 2


class AllMethodsChecked(RuntimeTypeCheckedMixin):
    _type_check_methods = "all"

    def _private(self, value: int) -> int:
        return value


class PropertyCheckingDisabled(RuntimeTypeCheckedMixin):
    _type_check_properties = False

    def __init__(self) -> None:
        self._value = 0

    @property
    def value(self) -> int:
        return self._value

    @value.setter
    def value(self, value: int) -> None:
        self._value = value


def test_constructor_checks_arguments_and_collection_contents():
    instance = ConstructorChecked(2, ("a", "b"))

    assert instance.count == 2
    assert instance.labels == ("a", "b")

    with pytest.raises(TypeError, match="argument 'count'"):
        ConstructorChecked("2")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="item 1"):
        ConstructorChecked(2, ("a", 1))  # type: ignore[arg-type]


def test_named_methods_check_arguments_including_variadics():
    instance = SelectedMethodsChecked()

    assert instance.update(1, "extra", source=1.0) == "1"

    with pytest.raises(TypeError, match="argument 'extras'"):
        instance.update(1, 2)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="argument 'metadata'"):
        instance.update(1, source="invalid")  # type: ignore[arg-type]


def test_unselected_methods_are_not_checked():
    instance = SelectedMethodsChecked()

    assert instance.unchecked("not an integer") == "not an integer"  # type: ignore[arg-type, comparison-overlap]


def test_public_mode_checks_public_instance_static_and_class_methods():
    instance = PublicMethodsChecked()

    with pytest.raises(TypeError, match="PublicMethodsChecked.public"):
        instance.public("invalid")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="PublicMethodsChecked.construct"):
        PublicMethodsChecked.construct("invalid")  # type: ignore[arg-type]

    assert SelectedMethodsChecked.make(1) == "1"

    with pytest.raises(TypeError, match="SelectedMethodsChecked.make"):
        SelectedMethodsChecked.make("invalid")  # type: ignore[arg-type]


def test_public_mode_does_not_check_private_methods():
    instance = PublicMethodsChecked()

    assert instance._private("invalid") == "invalid"  # type: ignore[arg-type, comparison-overlap]


def test_all_mode_checks_private_methods():
    with pytest.raises(TypeError, match="AllMethodsChecked._private"):
        AllMethodsChecked()._private("invalid")  # type: ignore[arg-type]


def test_property_setter_is_checked():
    instance = PropertyChecked()
    instance.value = 2

    assert instance.value == 2

    with pytest.raises(TypeError, match="value.setter"):
        instance.value = "invalid"  # type: ignore[assignment]


def test_property_checking_can_be_disabled():
    instance = PropertyCheckingDisabled()
    instance.value = "unchecked"  # type: ignore[assignment]

    assert instance.value == "unchecked"


def test_annotated_return_values_are_checked():
    with pytest.raises(TypeError, match="return value"):
        BadReturnChecked().value()

    assert ReturnCheckingDisabled().value() == "not an integer"


def test_async_methods_are_checked_and_preserve_coroutine_behavior():
    instance = AsyncChecked()

    assert asyncio.run(instance.double(2)) == 4

    with pytest.raises(TypeError, match="argument 'value'"):
        asyncio.run(instance.double("invalid"))  # type: ignore[arg-type]


def test_wrapped_signatures_are_preserved():
    assert str(inspect.signature(ConstructorChecked)) == (
        "(count: 'int', labels: 'tuple[str, ...]' = ()) -> 'None'"
    )
    assert str(inspect.signature(SelectedMethodsChecked.update)) == (
        "(self, value: 'int', *extras: 'str', "
        "**metadata: 'float') -> 'str'"
    )
