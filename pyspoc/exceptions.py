"""Public exception types raised by pySPoC."""
from __future__ import annotations

from typing import Sequence, Iterable, TYPE_CHECKING

from ._utils.const import PYSPOC_YAML_GUIDE_URL

if TYPE_CHECKING:
     from pyspoc.component._base import Component


class InvalidComponentConfigurationError(RuntimeError):

    def __init__(self, msg: str):
        super().__init__(msg)


class ComponentNotFoundError(NameError):

    def __init__(self, name: str | None):
        msg = ("The component {name} could not be found within the pySPoC component "
            "hierarchy.").format(name=name)
        super().__init__(msg, name=name)



class ComponentModuleNotFoundError(BaseExceptionGroup):

    def __init__(self, module_reference: str, exceptions: Sequence[Exception]):
        msg = f"Failed to import module {module_reference} due to a number of exceptions"
        super().__init__(msg, exceptions)



class InternalModuleNotFoundError(ModuleNotFoundError):

    def __init__(self, name: str | None):
        msg = ("The module {name} could not be found within the pySPoC hierarchy."
               ).format(name=name)
        super().__init__(msg, name=name)



class InternalModuleImportError(ImportError):

    def __init__(self, name: str | None):
        msg = ("The module {name} could not be imported from within the pySPoC hierarchy."
               ).format(name=name)
        super().__init__(msg, name=name)



class ExternalModuleNotFoundError(FileNotFoundError):

    def __init__(self, path: str | None = None):
        msg = ("The external module {path} could not be found. Please check the path "
            "of the module is absolute. If you are attempting to use a custom Component "
            "class, consider using Component.register() to register the class with pySPoC "
            "prior to loading your configuration.").format(path=path)
        super().__init__(msg)



class ExternalModuleImportError(ImportError):

    def __init__(self, path: str | None):
        msg = ("The external module {path} could not be imported. Please check the file "
            "is a valid Python module. If you are attempting to use a custom Component "
            "class, consider using Component.register() to register the class with pySPoC "
            "prior to loading your configuration.").format(path=path)
        super().__init__(msg, path=path)


class ExternalComponentNotFoundError(ModuleNotFoundError):

    def __init__(self, name: str | None, path: str | None = None):
        msg = ("The external compoentn {name} could not be found in module {path}. "
            "Please check the module path is correct or consider using Component.register() "
            "to register the class with pySPoC prior to loading your configuration."
            ).format(name=name, path=path)
        super().__init__(msg, name=name, path=path)


class DuplicateComponentReferenceError(KeyError):

    def __init__(self, name: str, references: Iterable[type[Component]]):

        # TODO: Replace this with _get_fully_qualified_name()
        reference_list = "\n\t".join(
            ref.__module__ + "." + ref.__name__ for ref in references)

        msg = ("The component name {name} is assigned to  distinct registered components:\n\t"
            "{reference_list}\n"
            "\n"
            "To produce a valid config, either:\n"
            "- Specify the fully qualified component name.\n"
            "- Nest the component configuration underneath the module.\n"
            "\n"
            "See {url} for more information."
        ).format(
            name=name,
            reference_list=reference_list,
            url=PYSPOC_YAML_GUIDE_URL)

        super().__init__(msg)



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
