from __future__ import annotations

import numpy as np
import pytest
import re

from pyspoc.base import Component


class DummyComponent(Component):
    def __init__(self, alpha, beta=2, *, gamma=3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    @property
    def name(self) -> str:
        return "dummy"

    @property
    def identifier(self) -> str:
        return "dummy-id"

    @property
    def labels(self) -> list[str]:
        return ["dummy"]

    @staticmethod
    def _get_component_type() -> type:
        return Component

    def compute(self, data: np.ndarray) -> np.ndarray:
        return np.asarray(data) * self.alpha

    def calculate(self, data: np.ndarray) -> np.ndarray:
        return self.compute(data)


class BadNameComponent(DummyComponent):
    
    def __init__(self, alpha=1, beta=2, *, gamma=3):
        super().__init__(alpha, beta, gamma=gamma)
    
    @property
    def name(self) -> int:
        return 123


class BadLabelsComponent(DummyComponent):
    
    def __init__(self, alpha=1, beta=2, *, gamma=3):
        super().__init__(alpha, beta, gamma=gamma)
    
    @property
    def labels(self):
        return 123


class DummyConfig:
    def __init__(self, name: str):
        self.name = name


def test_component_params_capture_default_and_kwonly_args():
    component = DummyComponent(alpha=1)

    assert component.params == {"alpha": 1, "beta": 2, "gamma": 3}
    assert component.alpha == 1
    assert component.beta == 2
    assert component.gamma == 3


def test_component_params_preserve_positional_and_keyword_args():
    component = DummyComponent(5, beta=8, gamma=9)

    assert component.params == {"alpha": 5, "beta": 8, "gamma": 9}
    assert component.alpha == 5
    assert component.beta == 8
    assert component.gamma == 9


def test_component_cfg_and_scheme_setter_getter():
    component = DummyComponent(alpha=1)
    config = DummyConfig("my-config")

    component.set_config(config)
    component.set_scheme("test-scheme")

    assert component.cfg is config
    assert component.scheme == "test-scheme"


def test_component_string_representation_includes_metadata():
    component = DummyComponent(alpha=1)
    config = DummyConfig("config-name")
    component.set_config(config)

    string = str(component)

    assert "Component: test_component.DummyComponent" in string
    assert "Name: dummy" in string
    assert bool(re.search("Active Parameters: {.*'alpha': 1.*}", string))
    assert bool(re.search("Active Parameters: {.*'beta': 2.*}", string))
    assert bool(re.search("Active Parameters: {.*'gamma': 3.*}", string))
    assert "Associated Configuration: config-name" in string


def test_component_info_shows_required_and_optional_arguments():
    info_text = DummyComponent.__info__()

    assert "Component: test_component.DummyComponent" in info_text
    assert "Required Parameters: ['alpha']" in info_text
    assert bool(re.search("Optional Parameters: {.*'beta': 2.*}", info_text))
    assert bool(re.search("Optional Parameters: {.*'gamma': 3.*}", info_text))


def test_component_init_raises_for_invalid_name_type():
    with pytest.raises(TypeError, match="name should be one of.*types"):
        BadNameComponent()


def test_component_init_raises_for_invalid_labels_type():
    with pytest.raises(TypeError, match="labels should be an iterable type"):
        BadLabelsComponent()
