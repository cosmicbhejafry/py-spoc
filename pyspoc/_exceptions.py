"""Public exception types raised by pySPoC."""
from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
     from pyspoc._base import Component



class InvalidStandardisedCausalGraphError(ValueError):

    def __init__(
            self,
            message: str,
            conventions_doc : str):
        
        message += ("\n" +
            "See below for more information on pySPoC's causal graph conventions.\n" +
            conventions_doc)
        
        super().__init__(message)



class MissingComponentPropertyError(NotImplementedError):
    def __init__(
            self,
            component: Component,
            property_name: str,
            property_type: type,
            has_private_method: bool = False):

        component_name = component.get_name()
        yaml_path = component._get_properties_yaml_path()
        property_method = f"get_{property_name}"

        if has_private_method:
            property_method = "_" + property_method
        
        message = (
            f"{component_name} must provide {property_type.__qualname__} "
            f"property {property_name} by either:\n"
                f"\t- Implementing class-level field _{property_name}.\n"
                f"\t- Implementing class method {property_method}().")

        if yaml_path:
            message += f"\n\t- Adding the property to YAML property file: pyspoc\\{yaml_path}"
        
        super().__init__(message)
        self._message = message


    @property
    def message(self) -> str:
        return self._message
        


class OptionalDependencyMissingError(ImportError):
    """Indicate that a requested feature requires an unavailable dependency.

    Parameters
    ----------
    dependency : str
        Import or distribution name of the missing optional dependency.
    feature : str or None, default=None
        Feature that requires the dependency, if a more specific description
        is useful to the caller.
    install_hint : str or None, default=None
        Optional installation guidance appended to the exception message.

    Attributes
    ----------
    dependency : str
        Import or distribution name supplied by the raising code.
    feature : str or None
        Feature that could not be used.
    install_hint : str or None
        Installation guidance supplied by the raising code.

    Notes
    -----
    This exception subclasses :class:`ImportError`, so existing handlers for
    failed imports remain compatible. Callers that want to distinguish an
    optional dependency from other import failures can catch this type more
    specifically.
    """

    def __init__(
            self,
            dependency: str,
            *,
            feature: str | None = None,
            install_hint: str | None = None):
        self.dependency = dependency
        self.feature = feature
        self.install_hint = install_hint

        message = f"Optional dependency {dependency!r} is required"
        if feature is not None:
            message += f" to use {feature}"
        message += "."

        if install_hint is not None:
            message += f" {install_hint}"

        super().__init__(message, name=dependency)
