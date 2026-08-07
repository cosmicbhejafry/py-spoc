from __future__ import annotations

import inspect

import pytest

from pyspoc._core.component import Component


class ExampleComponent(Component):
    def __init__(
        self,
        count: int,
        factor: float = 2.0,
        *,
        enabled: bool = True,
    ) -> None:
        super().__init__("example", ("test",))
        self.count = count
        self.factor = factor
        self.enabled = enabled

    @classmethod
    def _get_component_type(cls) -> type:
        return Component


class IntermediateComponent(Component):
    def __init__(self, short_name: str) -> None:
        super().__init__(short_name, ("intermediate",))

    @classmethod
    def _get_component_type(cls) -> type:
        return Component


class ConcreteComponent(IntermediateComponent):
    def __init__(self, value: int, *, mode: str = "default") -> None:
        super().__init__("concrete")
        self.value = value
        self.mode = mode


class VariadicComponent(Component):
    def __init__(self, name: str, *values: int, **options: str) -> None:
        super().__init__(name, ("variadic",))

    @classmethod
    def _get_component_type(cls) -> type:
        return Component


class FailingComponent(Component):
    def __init__(self, value: int) -> None:
        super().__init__("failing", ("test",))
        raise RuntimeError("initialization failed")

    @classmethod
    def _get_component_type(cls) -> type:
        return Component


def test_initialization_arguments_are_normalized_with_defaults() -> None:
    component = ExampleComponent(3)

    assert component.params == {
        "count": 3,
        "factor": 2.0,
        "enabled": True,
    }


def test_positional_and_keyword_calls_have_same_normalized_parameters() -> None:
    positional = ExampleComponent(3, 4.0, enabled=False)
    keyword = ExampleComponent(count=3, factor=4.0, enabled=False)

    assert positional.params == keyword.params


def test_only_outermost_constructor_arguments_are_recorded() -> None:
    component = ConcreteComponent(7, mode="custom")

    assert component.params == {"value": 7, "mode": "custom"}


def test_variadic_arguments_retain_bound_tuple_and_dictionary() -> None:
    component = VariadicComponent("sample", 1, 2, output="full")

    assert component.params == {
        "name": "sample",
        "values": (1, 2),
        "options": {"output": "full"},
    }


def test_initialization_argument_mapping_is_read_only() -> None:
    component = ExampleComponent(3)

    with pytest.raises(TypeError):
        component.params["count"] = 4  # type: ignore[index]


def test_constructor_signature_is_preserved() -> None:
    signature = inspect.signature(ExampleComponent)

    assert tuple(signature.parameters) == ("count", "factor", "enabled")
    assert signature.parameters["factor"].default == 2.0
    assert signature.parameters["enabled"].default is True


def test_failed_initialization_does_not_publish_call_arguments() -> None:
    component = FailingComponent.__new__(FailingComponent)

    with pytest.raises(RuntimeError, match="initialization failed"):
        component.__init__(5)

    assert component.params == {}
    assert not hasattr(component, "_component_init_capture_depth")


def test_short_name_property_returns_stored_value() -> None:
    component = ExampleComponent(3)

    assert component.short_name == "example"
