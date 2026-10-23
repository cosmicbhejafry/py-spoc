from __future__ import annotations

import inspect

import pytest
import yaml

from pyspoc._base import Component as LibraryComponent


class Component(LibraryComponent):
    """Isolate constructor capture from unrelated YAML metadata discovery."""

    @classmethod
    def _get_properties(cls, reload=False):
        return {}

    @classmethod
    def _get_property_schema(cls):
        return ()

    @classmethod
    def _get_properties_yaml_path(cls):
        return None


class ExampleComponent(Component):
    def __init__(
        self,
        count: int,
        factor: float = 2.0,
        *,
        enabled: bool = True,
    ) -> None:
        super().__init__()

    @property
    def identifier(self) -> str:
        return "example"

    @property
    def tags(self) -> list[str]:
        return ["test"]


class IntermediateComponent(Component):
    def __init__(self, internal_value: str) -> None:
        super().__init__()

    @property
    def identifier(self) -> str:
        return "intermediate"

    @property
    def tags(self) -> list[str]:
        return ["test"]


class ConcreteComponent(IntermediateComponent):
    def __init__(self, value: int, *, mode: str = "default") -> None:
        super().__init__("parent-only")


class VariadicComponent(Component):
    def __init__(self, name: str, /, *values: int, **options: str) -> None:
        super().__init__()

    @property
    def identifier(self) -> str:
        return "variadic"

    @property
    def tags(self) -> list[str]:
        return ["test"]


class FailingComponent(ExampleComponent):
    def __init__(self, count: int) -> None:
        super().__init__(count)
        raise RuntimeError("initialization failed")


def test_constructor_arguments_are_normalized_with_defaults() -> None:
    component = ExampleComponent(3)

    assert component.params == {
        "count": 3,
        "factor": 2.0,
        "enabled": True,
    }


def test_positional_and_keyword_calls_have_equal_parameters() -> None:
    positional = ExampleComponent(3, 4.0, enabled=False)
    keyword = ExampleComponent(count=3, factor=4.0, enabled=False)

    assert positional.params == keyword.params


def test_only_outermost_constructor_arguments_are_recorded() -> None:
    component = ConcreteComponent(7, mode="custom")

    assert component.params == {"value": 7, "mode": "custom"}


def test_variadic_and_positional_only_arguments_are_preserved() -> None:
    component = VariadicComponent("sample", 1, 2, output="full")

    assert component.params == {
        "name": "sample",
        "values": (1, 2),
        "options": {"output": "full"},
    }


def test_returned_parameter_mapping_cannot_mutate_component_state() -> None:
    component = ExampleComponent(3)
    parameters = component.params

    parameters["count"] = 9

    assert component.params["count"] == 3


def test_constructor_signature_is_preserved() -> None:
    signature = inspect.signature(ExampleComponent)

    assert tuple(signature.parameters) == ("count", "factor", "enabled")
    assert signature.parameters["factor"].default == 2.0
    assert signature.parameters["enabled"].default is True


def test_failed_initialization_does_not_publish_outer_arguments() -> None:
    component = FailingComponent.__new__(FailingComponent)

    with pytest.raises(RuntimeError, match="initialization failed"):
        component.__init__(5)

    assert component.params == {}
    assert "_component_init_capture_depth" not in vars(component)


def test_parameter_mapping_remains_yaml_serializable() -> None:
    component = ExampleComponent(3)

    serialized = yaml.safe_dump(component.params)

    assert yaml.safe_load(serialized) == component.params
