"""Tests for depth-aware automatic initialization hooks."""

from __future__ import annotations

import pytest

from pyspoc._initialization import AutoInitializedMixin


def test_nested_constructor_chain_runs_hooks_once_with_concrete_arguments() -> None:
    """A wrapped base constructor must not replace the outer constructor state."""
    events: list[tuple[str, dict[str, object]]] = []

    class Parent(AutoInitializedMixin):
        def __init__(self, shared: int) -> None:
            self.shared = shared

        def _before_component_init(self, init_args: dict[str, object]) -> None:
            events.append(("before", init_args.copy()))

        def _after_component_init(self, init_args: dict[str, object]) -> None:
            events.append(("after", init_args.copy()))

    class Child(Parent):
        def __init__(self, shared: int, specific: str = "default") -> None:
            super().__init__(shared=shared)
            self.specific = specific

    child = Child(shared=3, specific="child-only")

    expected_arguments = {"shared": 3, "specific": "child-only"}
    assert events == [
        ("before", expected_arguments),
        ("after", expected_arguments),
    ]
    assert child.shared == 3
    assert child.specific == "child-only"
    assert "_pyspoc_auto_initialization_depth" not in child.__dict__


def test_depth_state_is_removed_when_construction_raises() -> None:
    """A failed nested constructor must not leave later initialization nested."""
    events: list[str] = []

    class Parent(AutoInitializedMixin):
        def __init__(self) -> None:
            pass

        def _before_component_init(self, init_args: dict[str, object]) -> None:
            events.append("before")

        def _after_component_init(self, init_args: dict[str, object]) -> None:
            events.append("after")

    class Child(Parent):
        def __init__(self, fail: bool) -> None:
            super().__init__()
            if fail:
                raise RuntimeError("construction failed")

    child = Child.__new__(Child)

    with pytest.raises(RuntimeError, match="construction failed"):
        Child.__init__(child, fail=True)

    assert "_pyspoc_auto_initialization_depth" not in child.__dict__

    Child.__init__(child, fail=False)

    assert events == ["before", "before", "after"]
    assert "_pyspoc_auto_initialization_depth" not in child.__dict__
