from __future__ import annotations

import numpy as np
import pytest
import re

from pyspoc._base import Component as LibraryComponent
from pyspoc.config import Config
from pyspoc.settings import settings
from pyspoc._exceptions import MissingComponentPropertyError
from pyspoc._tag import Tag


class Component(LibraryComponent):
    """Keep constructor unit tests independent of external metadata files."""

    @classmethod
    def _get_properties(cls, reload=False):
        return {}

    @classmethod
    def _get_property_schema(cls):
        return ()

    @classmethod
    def _get_properties_yaml_path(cls):
        return None


class DummyComponent(Component):
    def __init__(self, alpha, beta=2, *, gamma=3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    _identifier = "dummy-id"
    _tags = (Tag.ASSOCIATION,)

    @property
    def name(self) -> str:
        return "dummy"

    @property
    def identifier(self) -> str:
        return "dummy-id"

    @property
    def tags(self) -> list[str]:
        return ["dummy"]

    @staticmethod
    def _get_component_type() -> type:
        return Component

    def compute(self, data: np.ndarray) -> np.ndarray:
        return np.asarray(data) * self.alpha

    def calculate(self, data: np.ndarray) -> np.ndarray:
        return self.compute(data)


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


def test_component_config_is_attached_through_internal_hook():
    component = DummyComponent(alpha=1)
    config = object.__new__(Config)
    config._name = "my-config"
    component._set_config(config)
    assert component.cfg is config
    assert component.scheme is None


def test_component_string_representation_includes_metadata():
    component = DummyComponent(alpha=1)
    config = object.__new__(Config)
    config._name = "config-name"
    component._set_config(config)

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


def test_component_metadata_fields_are_available_on_class_and_instance():
    component = DummyComponent(1)
    assert DummyComponent.get_identifier() == component.get_identifier() == "dummy-id"
    assert component.get_tags() == (Tag.ASSOCIATION,)


def test_missing_metadata_obeys_strict_policy():
    class MissingMetadata(Component):
        pass

    # Metadata is validated when requested, not as obsolete name/tags properties
    # during construction. Strict mode must not silently provide an empty value.
    with settings.override(properties_missing_response="throw"):
        with pytest.raises(MissingComponentPropertyError):
            MissingMetadata.get_identifier()
