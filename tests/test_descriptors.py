"""Class/instance binding and runtime-checking contracts for hybrid properties."""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Self

import pytest

from pyspoc._argchecking import RuntimeTypeCheckedMixin
from pyspoc._descriptors import hybridmethod, hybridproperty


class Example:
    default = "class default"

    def __init__(self, value="instance value"):
        self._value = value

    @hybridproperty
    def value(receiver: Example | type[Example]) -> str:
        """Return the receiver-specific value."""
        return receiver.default if isinstance(receiver, type) else receiver._value


def test_class_and_instance_access_return_values_without_parentheses():
    assert Example.value == "class default"
    assert Example("custom").value == "custom"
    assert not callable(Example.value)


def test_subclass_access_binds_the_subclass_not_the_declaring_class():
    class Child(Example):
        default = "child default"

    assert Child.value == "child default"
    assert Child("child instance").value == "child instance"
    assert Example.value == "class default"


def test_false_valued_instance_is_not_replaced_by_its_class():
    class FalseExample(Example):
        def __bool__(self):
            return False

    assert FalseExample("still an instance").value == "still an instance"


def test_values_are_recomputed_and_receiver_state_is_not_shared():
    first, second = Example("first"), Example("second")
    assert first.value == "first"
    first._value = "changed"
    assert first.value == "changed"
    assert second.value == "second"

    # Concurrent accesses must not overwrite a receiver stored on the shared
    # descriptor. Each access should evaluate against its own instance.
    instances = [Example(str(index)) for index in range(20)]
    with ThreadPoolExecutor(max_workers=2) as executor:
        values = list(executor.map(lambda instance: instance.value, instances))
    assert values == [str(index) for index in range(20)]


def test_descriptor_preserves_getter_documentation_and_subclass_relationship():
    descriptor = inspect.getattr_static(Example, "value")
    assert isinstance(descriptor, hybridmethod)
    assert descriptor.__doc__ == "Return the receiver-specific value."
    assert descriptor.__wrapped__ is descriptor.method


def test_instance_assignment_and_deletion_are_rejected():
    instance = Example()
    with pytest.raises(AttributeError, match="read-only"):
        setattr(instance, "value", "replacement")
    with pytest.raises(AttributeError, match="read-only"):
        delattr(instance, "value")

    # A data descriptor must also take precedence over an instance-dict entry.
    instance.__dict__["value"] = "shadow"
    assert instance.value == "instance value"


@pytest.mark.parametrize("getter", [
    lambda: 1,
    lambda receiver, extra: 1,
    lambda receiver, extra=None: 1,
    lambda *receivers: 1,
    lambda **receivers: 1,
    lambda *, receiver: 1,
])
def test_getter_requires_exactly_one_positional_receiver(getter):
    with pytest.raises(TypeError, match="exactly one positional receiver"):
        hybridproperty(getter)


def test_abstract_getter_remains_abstract_until_overridden():
    class Abstract(ABC):
        @hybridproperty
        @abstractmethod
        def value(receiver):
            raise NotImplementedError

    class Incomplete(Abstract):
        pass

    # ABCMeta inspects the inherited attribute through class access. Neither
    # running the abstract getter nor losing its abstract flag is acceptable.
    assert inspect.isabstract(Abstract)
    assert inspect.isabstract(Incomplete)
    with pytest.raises(TypeError, match="abstract"):
        Incomplete()

    class Concrete(Abstract):
        @hybridproperty
        def value(receiver):
            return "implemented"

    assert not inspect.isabstract(Concrete)
    assert Concrete.value == Concrete().value == "implemented"


class Checked(RuntimeTypeCheckedMixin):
    # Properties follow their own policy even when method checking is disabled.
    _type_check_methods = set()

    @hybridproperty
    def peer(receiver: Self | type[Self]) -> Self | type[Self]:
        return receiver

    @hybridproperty
    def bad_value(receiver) -> int:
        return "not an integer"


def test_runtime_wrapper_preserves_value_access_and_self_resolution():
    class Child(Checked):
        pass

    instance = Child()
    assert Child.peer is Child
    assert instance.peer is instance
    assert isinstance(inspect.getattr_static(Checked, "peer"), hybridproperty)


@pytest.mark.parametrize("receiver", [Checked, Checked()])
def test_runtime_wrapper_checks_getter_return_values(receiver):
    with pytest.raises(TypeError, match="Invalid return value"):
        _ = receiver.bad_value


def test_property_checks_can_be_disabled_without_changing_binding():
    class Unchecked(RuntimeTypeCheckedMixin):
        _type_check_properties = False

        @hybridproperty
        def value(receiver) -> int:
            return "unchecked"

    assert Unchecked.value == Unchecked().value == "unchecked"


def test_checked_abstract_property_keeps_its_abstract_status():
    class Abstract(RuntimeTypeCheckedMixin, ABC):
        @hybridproperty
        @abstractmethod
        def value(receiver) -> int:
            raise NotImplementedError

    class Incomplete(Abstract):
        pass

    assert inspect.isabstract(Incomplete)
