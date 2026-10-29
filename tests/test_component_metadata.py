"""Regression coverage for component metadata and receiver-aware checks."""

from __future__ import annotations

import asyncio
import inspect
from typing import Self

import pytest

from pyspoc._argchecking import RuntimeTypeCheckedMixin
from pyspoc.component._base import Component as LibraryComponent, get_obj_init_args, hybridmethod


class Component(LibraryComponent):
    @classmethod
    def _get_properties(cls, reload=False):
        return {}

    @classmethod
    def _get_property_schema(cls):
        return ()

    @classmethod
    def _get_properties_yaml_path(cls):
        return None


class MetadataComponent(Component):
    def __init__(
        self, required: int, /, zero: int = 0, *,
        disabled: bool = False, empty: str = "", optional: str | None = None,
    ) -> None:
        super().__init__()
        self.identifier = str(required)

    @classmethod
    def get_long_name(cls) -> str:
        return "Metadata component"

    @classmethod
    def get_signatures(cls):
        return ()

    @staticmethod
    def _get_identifier(obj: MetadataComponent | type[MetadataComponent]) -> str:
        return "default" if isinstance(obj, type) else obj.identifier

    @staticmethod
    def _get_type(obj):
        return ()

    @staticmethod
    def _get_tags(obj):
        return ()


def test_wrapped_constructor_defaults_and_required_parameters() -> None:
    assert get_obj_init_args(MetadataComponent) == {
        "required": None, "zero": 0, "disabled": False,
        "empty": "", "optional": None,
    }
    information = MetadataComponent.__info__()
    assert "Required Parameters: ['required']" in information
    assert "'zero': 0" in information
    assert "'disabled': False" in information
    assert "'empty': ''" in information
    assert "'optional': None" in information


def test_constructor_defaults_do_not_change_captured_parameters() -> None:
    component = MetadataComponent(3)
    assert component._active_calculator is None
    assert component.params == {
        "required": 3, "zero": 0, "disabled": False,
        "empty": "", "optional": None,
    }


def test_hybrid_accessor_binding_documentation_and_signature() -> None:
    assert MetadataComponent.get_identifier() == "default"
    assert MetadataComponent(3).get_identifier() == "3"
    for accessor in (
        MetadataComponent.get_identifier,
        MetadataComponent(3).get_identifier,
    ):
        assert accessor.__doc__ == Component.get_identifier.__doc__
        assert len(inspect.signature(accessor).parameters) == 0


class CheckedReceiver(RuntimeTypeCheckedMixin):
    def peer(self, value: Self | type[Self]) -> Self | type[Self]:
        return value

    def nested(self, values: list[Self]) -> list[Self]:
        return values

    def incorrect(self) -> Self:
        return CheckedReceiver()

    @classmethod
    def class_peer(cls, value: Self | type[Self]) -> Self | type[Self]:
        return value

    @hybridmethod
    def hybrid_peer(receiver: CheckedReceiver | type[CheckedReceiver], value: Self) -> Self:
        """Return a checked peer through class or instance access."""
        return value

    @hybridmethod
    def invalid_hybrid_return(receiver: CheckedReceiver | type[CheckedReceiver]) -> int:
        return "wrong"

    async def async_peer(self, value: Self) -> Self:
        return value


class CheckedChild(CheckedReceiver):
    pass


@pytest.mark.parametrize("access", [
    lambda child: child.peer,
    lambda child: child.class_peer,
    lambda child: CheckedChild.class_peer,
    lambda child: child.hybrid_peer,
    lambda child: CheckedChild.hybrid_peer,
])
def test_inherited_self_checks_use_actual_receiver(access) -> None:
    child = CheckedChild()
    method = access(child)
    assert method(child) is child
    with pytest.raises(TypeError, match="Invalid argument"):
        method(CheckedReceiver())


def test_self_class_objects_and_nested_containers() -> None:
    child = CheckedChild()
    assert child.peer(CheckedChild) is CheckedChild
    assert child.nested([child]) == [child]
    with pytest.raises(TypeError):
        child.peer(CheckedReceiver)
    with pytest.raises(TypeError):
        child.nested([CheckedReceiver()])


def test_self_and_hybrid_return_validation() -> None:
    with pytest.raises(TypeError, match="Invalid return value"):
        CheckedChild().incorrect()
    with pytest.raises(TypeError, match="Invalid return value"):
        CheckedChild.invalid_hybrid_return()


def test_async_self_validation() -> None:
    child = CheckedChild()
    assert asyncio.run(child.async_peer(child)) is child
    with pytest.raises(TypeError):
        asyncio.run(child.async_peer(CheckedReceiver()))


def test_static_self_requires_a_receiver() -> None:
    class InvalidStatic(RuntimeTypeCheckedMixin):
        @staticmethod
        def method(value: Self) -> None:
            pass

    with pytest.raises(TypeError, match="Self requires"):
        InvalidStatic.method(InvalidStatic())
