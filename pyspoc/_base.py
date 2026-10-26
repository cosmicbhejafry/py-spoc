from __future__ import annotations

import numpy as np
import pandas as pd
import os
import inspect
import pkgutil
import re
import importlib
import warnings

from runpy import run_path
from collections.abc import Callable, Collection
from functools import wraps
from types import FunctionType, ModuleType, MethodType, MappingProxyType
from typing import (
    Self, Literal, Iterable, Any, Optional, Dict, TypeVar, cast, overload, TYPE_CHECKING)
from abc import ABC, abstractmethod
from textwrap import dedent
from numbers import Number
from argparse import Namespace
from typeguard import typechecked
from yaml import safe_load
from warnings import warn
from enum import Enum
from pathlib import Path
from importlib import import_module
from importlib.resources.abc import Traversable

from pyspoc._exceptions import OptionalDependencyMissingError
from pyspoc._descriptors import hybridmethod
from pyspoc._argchecking import RuntimeTypeCheckedMixin
from pyspoc._random import RandomSeedMixin
from pyspoc._utils import lib
from pyspoc.settings import settings
from pyspoc._exceptions import MissingComponentPropertyError
from pyspoc._tag import Tag, Type, Signature

if TYPE_CHECKING:
    from pyspoc.calculator import Calculator
    from pyspoc.config import Config


_INITIALIZATION_CAPTURED_ATTRIBUTE = "_pyspoc_initialization_captured"
TInit = TypeVar("TInit", bound=Callable[..., None])
TTag = TypeVar("TTag", bound=Tag | Type | Signature)


# ---------------------------------------------------------------------------
# Implementation note
# ---------------------------------------------------------------------------
#
# `Component` is the base object for all main
#
# All core numerical algorithms have been moved into standalone functions in
# `func`. This separation follows the common scientific-computing pattern of
# isolating pure computational kernels from the object-oriented API layer.
#
# The benefits of this design are:
#
#   * Stateless kernels are easier to unit test in isolation.
#   * Computational code can be accelerated independently (e.g. via Numba).
#   * The class remains responsible only for configuration and orchestration.
#   * Algorithmic logic is not tightly coupled to object state.
#
# In practice this class is responsible for:
#
#   * storing configuration parameters (e.g. q, tolerances)
#   * validating inputs and managing object state
#   * dispatching calls to the appropriate function in `func`
#
# The `func` module therefore contains the functional core of the algorithm,
# while this class provides the public stateful interface.
# ---------------------------------------------------------------------------

class Component(
    RuntimeTypeCheckedMixin,
    RandomSeedMixin,
    ABC):

    """
    Base object for both Statistic and Reducer components.

    Provides functionality for constructing configuration files.
    """

    _TICK_CHAR = u'\u2714'
    _CROSS_CHAR = u'\u2716'

    # Constructors temporarily shadow this default on each instance. Declaring
    # it here makes the internal lifecycle attribute visible to static type
    # checkers without retaining per-instance state after initialization.
    _component_init_capture_depth: int = 0
    _warned_properties_missing = False
    _warned_properties_error = False
    _properties_mapping: MappingProxyType[str, dict[str, str | list]] = \
        MappingProxyType({})
    _cached_modules = dict()
    _cached_components: MappingProxyType[str, dict[str, type["Component"]]] = \
        MappingProxyType({})


    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Instrument directly declared constructors for argument capture.

        A constructor inherited unchanged from a parent has already been
        instrumented where it was declared. Wrapping only constructors present
        in the new subclass namespace avoids rebinding a concrete call against
        an unrelated parent signature.
        """
        super().__init_subclass__(**kwargs)

        declared_init = cls.__dict__.get("__init__")

        if not inspect.isfunction(declared_init):
            return

        if getattr(declared_init, _INITIALIZATION_CAPTURED_ATTRIBUTE, False):
            return

        cls.__init__ = cls._wrap_initialization_argument_capture(declared_init)
        cls._load_properties()

    @classmethod
    def _wrap_initialization_argument_capture(cls, init: TInit) -> TInit:
        """Wrap a constructor and record its normalized call arguments.

        Arguments are bound using Python's own signature machinery, so
        positional-only, keyword-only, variadic, and defaulted parameters are
        represented consistently. In a cooperative ``super()`` chain, only
        the outermost constructor call is retained as the public Component
        configuration.
        """
        signature = inspect.signature(init)

        @wraps(init)
        def wrapped_init(
            self: Component,
            *args: Any,
            **kwargs: Any,
        ) -> None:
            # Bind before initialization. Invalid calls therefore raise the
            # normal Python TypeError without publishing partial parameters.
            bound = signature.bind(self, *args, **kwargs)
            bound.apply_defaults()

            normalized_arguments = dict(bound.arguments)
            normalized_arguments.pop("self", None)

            # Wrapped parent constructors are entered through ``super()``.
            # Track their nesting so they cannot overwrite the concrete
            # constructor arguments supplied by the user.
            depth = self._component_init_capture_depth
            self._component_init_capture_depth = depth + 1

            try:
                init(self, *args, **kwargs)

                # Publish configuration only after successful construction.
                if depth == 0:
                    self._params = normalized_arguments
            finally:
                remaining_depth = self._component_init_capture_depth - 1

                if remaining_depth == 0:
                    del self._component_init_capture_depth
                else:
                    self._component_init_capture_depth = remaining_depth

        setattr(wrapped_init, _INITIALIZATION_CAPTURED_ATTRIBUTE, True)
        return cast(TInit, wrapped_init)
    

    def __init__(
            self,
            internal_parallelism: Literal["serial", "parallel"] = "serial",
            thread_safety: Literal["safe", "unsafe"] = "unsafe"):

        self._cfg: Optional[Config] = None
        self._active_calculator: Calculator | None = None
        self._scheme: Optional[str] = None
        self._params: Dict = {}
        self._cache = dict()
        self._thread_safety = thread_safety
        self._internal_parallelism = internal_parallelism
      

    @property
    def cfg(self):
        return self._cfg
    

    @property
    def scheme(self):
        return self._scheme
    

    @property
    def params(self) -> dict:
        """Return a shallow copy of the normalized constructor arguments."""
        return self._params.copy()
    

    # NOTE: Change this to get_name() soon. Will require refactoring a lot of areas.
    @property
    def name(self) -> str:
        return self.__class__.__name__


    @classmethod
    def get_name(cls) -> str:
        return cls.__name__


    @classmethod
    def get_long_name(cls) -> str:
        """Return the long, human-readable name for the Component class.
        
        A valid Component class must either:
        - Implement _long_name: A `str` class-level private field.
        - Implement get_long_name(): A public class method overriding this method, returning `str`.
        - Add the 'long_name' string property under your Component in the relevant YAML property
            file.
        """
        error_response = settings.current.properties_missing_response

        try:
            return getattr(cls, "_yaml_long_name")
        except AttributeError:
            try:
                return getattr(cls, "_long_name")
            except AttributeError as e:
                if error_response == "ignore":
                    return ""

                missing_err = MissingComponentPropertyError(
                    cls, "long_name", str)
                
                if error_response == "throw":
                    raise missing_err from e

                warn(missing_err.message)
                return ""


    @classmethod
    def get_signatures(cls) -> tuple[Signature, ...]:
        """Return the signatures for the Component class.
        
        A valid Component class must either:
        - Implement _signatures: A `tuple[Signature, ...]` class-level private field.
        - Implement get_signatures(): A public class method overriding this method,
            returning `tuple[Signature, ...]`.
        - Add the 'signatures' list[str] property under your Component in the relevant YAML property
            file, containing only valid `Signature` strings.

        See :class:`Signature` for a list of available signatures.
        """
        error_response = settings.current.properties_missing_response

        try:
            return getattr(cls, "_yaml_signatures")
        except AttributeError:
            try:
                return getattr(cls, "_signatures")
            except AttributeError as e:
                if error_response == "ignore":
                    return tuple()

                missing_err = MissingComponentPropertyError(
                    cls, "signatures", tuple[Signature, ...])
                
                if error_response == "throw":
                    raise missing_err from e

                warn(missing_err.message)
                return tuple()


    @hybridmethod
    def get_identifier(cls_or_self: Self | type[Self]) -> str:
        """Return the identifier for a Component class or instance."""
        owner = cls_or_self if isinstance(cls_or_self, type) else type(cls_or_self)
        error_response = settings.current.properties_missing_response

        try:
            return owner._get_identifier(cls_or_self)
        except AttributeError as e:
            if error_response == "ignore":
                return ""

            missing_err = MissingComponentPropertyError(
                owner, "identifier", str, True)
            
            if error_response == "throw":
                raise missing_err from e

            warn(missing_err.message)
            return ""

    
    @staticmethod
    def _get_identifier(
        cls_or_self: Component | type[Component]) -> str:
        """Return the short identifier for the Component class.
        
        A valid Component class must either:
        - Implement _identifier: A `str` class-level private field.
        - Implement _get_identifier(): A private static method overriding this method,
            returning `str`.
        - Add the 'identifier' string property under your Component in the relevant YAML
            property file.
        """
        try:
            return getattr(cls_or_self, "_yaml_identifier")
        except AttributeError:
            return getattr(cls_or_self, "_identifier")


    @hybridmethod
    def get_type(cls_or_self: Self | type[Self]) -> Type:
        """Return the type of a Component class or instance."""
        owner = cls_or_self if isinstance(cls_or_self, type) else type(cls_or_self)
        error_response = settings.current.properties_missing_response
        
        try:
            return owner._get_type(cls_or_self)
        except AttributeError as e:
            if error_response == "ignore":
                return Type.DESCRIPTIVE

            missing_err = MissingComponentPropertyError(
                owner, "type", Type, True)
            
            if error_response == "throw":
                raise missing_err from e

            warn(missing_err.message)
            return Type.DESCRIPTIVE

    
    @staticmethod
    def _get_type(cls_or_self: Component | type[Component]) -> Type:
        """
        Return the types of a Component class or instance.

        A valid Component class must either:
        - Implement _type: A `Type` class-level private field.
        - Implement _get_type(): A private static method overriding this method, returning `Type`.
        - Add the 'type' string property under your Component in the relevant YAML property file,
            resricted to valid `Type` strings.

        See :class:`Type` for a list of available types.
        """
        try:
            return getattr(cls_or_self, "_yaml_type")
        except AttributeError:
            return getattr(cls_or_self, "_type")


    @hybridmethod
    def get_tags(cls_or_self: Self | type[Self]) -> tuple[Tag, ...]:
        """Return the tags of a Component class or instance."""
        owner = cls_or_self if isinstance(cls_or_self, type) else type(cls_or_self)
        error_response = settings.current.properties_missing_response
        
        try:
            return owner._get_tags(cls_or_self)
        except AttributeError as e:
            if error_response == "ignore":
                return tuple()
            
            missing_err = MissingComponentPropertyError(
                owner, "tags", tuple[Tag, ...], True)
            
            if error_response == "throw":
                raise missing_err from e

            warn(missing_err.message)
            return tuple()


    @staticmethod
    def _get_tags(cls_or_self: Component | type[Component]) -> tuple[Tag, ...]:
        """
        Return the tags of a Component class or instance.

        A valid Component class must either:
        - Implement _tags: A `tuple[Tag, ...]` class-level private field.
        - Implement _get_tags(): A private static method overriding this method,
            returning `tuple[Tag, ...]`.
        - Add the 'tags' list[str] property under your Component in the relevant YAML property
            file, containing only valid `Tag` strings.

        See :class:`Tag` for a list of available tags.
        """
        try:
            return getattr(cls_or_self, "_yaml_tags")
        except AttributeError:
            return getattr(cls_or_self, "_tags")


    @staticmethod
    def _get_modified_tags(
            current_tags: Collection[TTag],
            add_tags: Collection[TTag] | TTag | None = None,
            remove_tags: Collection[TTag] | TTag | None = None) -> tuple[TTag, ...]:
        
        tags_difference = (set() if remove_tags is None
            else set(remove_tags) if isinstance(remove_tags, Collection)
            else {remove_tags})

        tags_union: set[TTag] = (set() if add_tags is None
            else set(add_tags) if isinstance(add_tags, Collection)
            else {add_tags})

        new_tags = set(current_tags).difference(tags_difference).union(tags_union)
        return tuple(new_tags)


    def set_scheme(self, scheme: str):
        self._scheme = scheme


    @classmethod
    def get_available(
            cls,
            refresh: bool = False,
            raise_errors: bool = False
    ) -> tuple[
            tuple[type[Component], ...],
            dict[str, Exception]]:
        
        discovered: set[type[Component]] = set()
        failures: dict[str, Exception] = {}

        if cls._cached_components and not refresh:
            for module_reference, component_dict in cls._cached_components.items():
                for component_name, component in component_dict.items():
                    discovered.add(component)

        else:
            component_type = cls._get_component_type()
            base_package_name = component_type.__module__
            module = import_module(base_package_name)
            package = cls._get_containing_package(module)

            if package is None:
                error = RuntimeError(
                    "Unable to find associated package to search "
                    f"for {cls.__qualname__}s")

                if raise_errors:
                    raise error
                else:
                    failures[cls.__qualname__] = error
                    return (), failures

            cached_components = dict()
            
            for module_info in pkgutil.walk_packages(
                package.__path__,
                prefix=f"{package.__name__}."):

                try:
                    module = import_module(module_info.name)
                except Exception as error:
                    if raise_errors:
                        raise error

                    failures[module_info.name] = error
                    continue

                for _, candidate in inspect.getmembers(module, inspect.isclass):
                    try:
                        if (
                            candidate is not component_type
                            and issubclass(candidate, component_type)
                            and not inspect.isabstract(candidate)
                            and candidate.__module__ == module.__name__):
                            
                            discovered.add(candidate)
                            
                            if not cached_components.get(module.__name__):
                                cached_components[module.__name__] = {}

                            module_components = cached_components[module.__name__]
                            module_components[candidate.__qualname__] = candidate

                    except Exception as error:
                        if raise_errors:
                            raise error

                        failures[module_info.name] = error
                        continue

                cls._cached_components = MappingProxyType(cached_components)
                
        classes = tuple(
            sorted(discovered, key=lambda cls: cls.__module__ + cls.__qualname__)
        )
        return classes, failures

    @overload
    @classmethod
    def make_component(
            cls,
            component_name: str) -> Component | None: ...


    @overload
    @classmethod
    def make_component(
            cls,
            component_name: str,
            module_reference: str) -> Component | None: ...

    @overload
    @classmethod
    def make_component(
            cls,
            component_name: str,
            module_reference: str,
            component_archetype: type[Component]) -> Component | None: ...
    

    @classmethod
    def make_component(
            cls,
            component_name: str,
            module_reference: str | None = None,
            component_archetype: type[Component] | None = None) -> Component | None:
        
        if module_reference is None:
            available = cls.get_available()
            available

        full_component_name = _get_full_component_name(module_reference, component_name)

        # Return cached object if present.
        if component_archetype:
            component_type_name = component_archetype.__name__
            component_classes = cls._cached_components.get(component_type_name)
        
            if component_classes:
                component_class = component_classes.get(full_component_name)

                if component_class:
                    return component_class()

        # Otherwise check the module directory - will only have it by class name not component name.
        # TODO: this needs a fix
        is_component_exists = component_name in module_reference.__dir__()

        # Skip if not present.
        if not is_component_exists:
            print(f"  {cls._CROSS_CHAR} {component_type_name} {component_name} could not be found. "
                  "Skipping.")
            return

        # Get object from module
        component_class = getattr(module_reference, component_name)

        # Check object is the expected type
        if not issubclass(component_class, component_archetype):
            type_name = component_class.__name__
            print(f"  {cls._CROSS_CHAR} {component_type_name} {component_name} is of type "
                  f"{type_name} which is not a {component_type_name} object. Skipping.")
            return


        if False:
            dependency_delimited = ",".join(dependency_set)
            print(f"  {cls._CROSS_CHAR} {component_type_name} {component_name} is missing dependencies. "
                    f"Skipping")
            print(f"    (Missing dependencies: {dependency_delimited})")
            return

        return component_class

    @classmethod
    def _find_component(
            cls,
            component_name: str) -> Component | None:


        

    @classmethod
    def _get_module(
            cls,
            module_reference: str,
            global_modules: dict | None = None,
            refresh_module: bool = False) -> ModuleType | Namespace | None:

        # Check if module is cached.
        module = cls._get_cached_module(module_reference)

        if module:
            return module

        # Check package modules.
        module = cls._get_package_module(module_reference)

        if module:
            return module

        # Check modules loaded in memory.
        module = cls._get_loaded_module(module_reference, global_modules)
        
        if module:
            return module

        # Check module files.
        module = cls._load_module_file(module_reference,
                                        refresh_module=refresh_module,
                                        suppress_warning=True)

        if module:
            return module
        
    @classmethod
    def _get_cached_module(
            cls,
            module_reference: str) -> ModuleType | Namespace | None:

        # Check if module is already loaded.
        module = cls._cached_modules.get(module_reference)

        if module:
            print(f"  {cls._TICK_CHAR} Module {module_reference} already loaded.")
            return module
        
    @classmethod
    def _get_package_module(
            cls,
            module_reference: str) -> ModuleType | None:
        
        # Get package modules.
        try:
            module = importlib.import_module(module_reference, __package__)
            print(f"  {cls._TICK_CHAR} Module {module_reference} loaded successfully.")
            
            if hasattr(module, "IMPORT_WARNINGS") and not settings.current.ignore_import_warnings:
                print("    (The following warnings were raised: "
                        f"{[str(w.message) for w in module.IMPORT_WARNINGS]})")

            return module
        except ModuleNotFoundError:
            pass
        except Exception as e:
            raise e
        
    @classmethod
    def _get_loaded_module(
            cls,
            module_reference: str,
            global_modules: dict | None) -> ModuleType | None:
    
        if not global_modules:
            global_modules = {obj.__name__.lower(): obj for obj
                in globals().values() if inspect.ismodule(obj)}
        
        module = global_modules.get(module_reference)

        if module:
            print(f"  {cls._TICK_CHAR} Module {module_reference} loaded from global environment.")
            return module

    @classmethod
    def _load_module_file(
            cls,
            module_path: str,
            refresh_module: bool,
            suppress_warning: bool = False) -> ModuleType | Namespace | None:

        # Get cached module if available
        module = cls._cached_modules.get(module_path)

        if module and not refresh_module:
            return module

        # Otherwise check if module path exists
        if not os.path.isfile(module_path) and not suppress_warning:
            warnings.warn(f"No file could be found at the following path: {module_path}")
            return

        # Load and return module
        module_dict = run_path(module_path, run_name=module_path)
        module = Namespace(**module_dict)
        cls._cached_modules[module_path] = module
        return module

    
    @classmethod
    def reload_properties(cls):
        cls._load_properties(reload=True)
        
    
    @classmethod
    def _load_properties(cls, reload: bool = False):
        property_schema = cls._get_property_schema()
        properties = cls._get_properties(reload)
        throw_load_errors = settings.current.properties_load_error_response

        if not properties:
            return

        error_msg_template = (f"{cls.__name__} encountered an Enum conversion error " +
            "attempting to load property '{property_name}':\n" +
            "{property_value}\n" +
            f"Check {cls.__name__} properties " + "in {yaml_path} for errors.")
        
        for property_name, property_type, _ in property_schema:
            result = None
            standardised_name = re.sub(r"[\W_]", "", property_name).lower()
            property_name.lower().strip()
            result = properties.get(standardised_name)
            formatted_name = "_yaml_" + property_name

            if isinstance(result, Collection) and not isinstance(result, str):
                values = []
                
                for element in result:
                    if element is None:
                        continue

                    try:
                        value = (property_type(element)
                            if issubclass(property_type, Enum)
                            else element)
                        values.append(value)
                    except ValueError as e:
                        yaml_path = "pyspoc" / cls._get_properties_yaml_path()
                        error_msg = error_msg_template.format(
                            property_name=property_name,
                            property_value=e.args[0],
                            yaml_path=yaml_path)

                        if throw_load_errors == "throw":
                            raise ValueError(error_msg) from e
                        elif throw_load_errors == "warn":
                            warn(error_msg)

                        continue

                setattr(cls, formatted_name, tuple(values))

            else:
                if result is None:
                    continue

                try:
                    value = (property_type(result)
                        if issubclass(property_type, Enum)
                        else result)
                except ValueError as e:
                    yaml_path = "pyspoc" / cls._get_properties_yaml_path()
                    error_msg = error_msg_template.format(
                        property_name=property_name,
                        property_value=e.args[0],
                        yaml_path=yaml_path)
                    
                    if throw_load_errors == "throw":
                        raise ValueError(error_msg) from e
                    elif throw_load_errors == "warn":
                        warn(error_msg)

                    continue
                    
                setattr(cls, formatted_name, value)
    
    
    @classmethod
    def _get_properties(cls, reload: bool = False):
        if (reload
            or not isinstance(cls._properties_mapping, MappingProxyType)
            or not cls._properties_mapping):
            cls._load_properties_yaml()
                
        found = cls._properties_mapping.get(cls.__module__ + "." + cls.__qualname__)

        if not found:
            found = cls._properties_mapping.get(cls.__qualname__)

        return found


    @classmethod
    def _load_properties_yaml(cls):
        resource = cls._get_properties_yaml_resource()

        if resource is None:
            return
        
        try:
            with resource.open(mode="r") as f:
                properties_dict = safe_load(f)
                formatted_dict, exceptions = cls._standardise_properties_dict(properties_dict)
                cls._properties_mapping = MappingProxyType(formatted_dict)

                if exceptions:
                    raise ExceptionGroup(
                        "A number of errors were encountered loading properties YAML",
                        exceptions)

        except Exception as err:
            yaml_path = cls._get_properties_yaml_path()
            error_prefix = ("Loading properties YAML file threw an error:\n\t" +
                f"File: {yaml_path}")

            match settings.current.properties_load_error_response:
                case "error":
                    raise err

                case "warn":
                    if cls._warned_properties_error:
                        return
                    
                    cls._warned_properties_error = True

                    if isinstance(err, ExceptionGroup):
                        msg = err.message + "\n\t" + "\n\t".join(
                            [str(e) + (f" ({e.__notes__[0]})" if hasattr(e, "__notes__") else "")
                                for e in err.exceptions])
                    else:
                        msg = str(err)

                    warn(error_prefix + f"\n\tException: {msg}")


    @classmethod
    def _get_properties_yaml_resource(cls) -> Traversable | None:
        yaml_path = cls._get_properties_yaml_path()
        resource = None
        
        try:
            resource = lib.get_library_resource(yaml_path)
        except FileNotFoundError as e:
            new_error_msg = f"Properties YAML file could not be found at: {yaml_path}"
            cls._handle_properties_missing(new_error_msg, e)
                
        return resource
    

    @classmethod
    def _handle_properties_missing(
        cls,
        new_error_msg: str,
        base_error: FileNotFoundError):

        match settings.current.properties_missing_response:
            case "error":
                raise FileNotFoundError(new_error_msg) from base_error

            case "warn":
                if not cls._warned_properties_missing:
                    cls._warned_properties_missing = True
                    warn(new_error_msg)


    @classmethod
    def _standardise_properties_dict(
        cls,
        current_dict: dict[str, Any]) -> tuple[dict[str, Any], list[Exception]]:
        
        standardised_dict = {}
        exceptions = []
        
        for key, properties in current_dict.items():
            if not isinstance(properties, dict):
                continue

            standardised_dict[key] = {}

            for property_key, val in properties.items():

                try:
                    formatted_property = re.sub(r"[\W_]", "", property_key).lower()

                    if val is None:
                        continue

                    prop_dict = standardised_dict[key]

                    match formatted_property:
                        case "identifier" | "type":
                            prop_dict[formatted_property] = (
                                val.lower() if isinstance(val, str) else val)

                        case "tags" | "signatures":
                            prop_dict[formatted_property] = tuple(
                                v.lower() for v in val if isinstance(v, str))

                        case _:
                            prop_dict[formatted_property] = val

                except Exception as e:
                    e.add_note(f"Component: {key}, Property: {property_key}, Value: {val}")
                    exceptions.append(e)

        return standardised_dict, exceptions


    @classmethod
    @abstractmethod
    def _get_property_schema(cls) -> tuple[
        tuple[str, type, MethodType | FunctionType], ...]:
        pass


    @classmethod
    @abstractmethod
    def _get_properties_yaml_path(cls) -> Path:
        pass
 

    @staticmethod
    def _get_containing_package(module: ModuleType) -> ModuleType | None:
        """
        Return the module itself if it is a package,
        otherwise its parent package.
        """
        
        if hasattr(module, "__path__"):
            return module

        package_name = module.__package__

        if not package_name:
            return None

        return import_module(package_name)
    
    
    def _set_config(self, cfg: Config):
        self._cfg = cfg


    def _set_active_calculator(self, calculator: Calculator):
        self._active_calculator = calculator


    @staticmethod
    def _get_property_val(prop: Any) -> Any:
        while callable(prop):
            prop = prop()

        return prop

    @staticmethod
    def _get_component_type() -> type:
        return Component


    def __str__(self):
        self_type = type(self)
        full_name = get_fully_qualified_type_name(self_type)
        cfg_name = self.cfg.name if self.cfg is not None else "None"
        component_type_name = self._get_component_type().__name__

        return dedent(
            f"""
            {component_type_name}: {full_name}
            Name: {self.name}
            Active Parameters: {self.params}
            Associated Configuration: {cfg_name}
            """)

    @classmethod
    def __info__(cls):
        """Describe the component class and its constructor parameters.

        Returns
        -------
        str
            Formatted component identity, required parameter names, and
            optional parameters with their declared defaults.

        Notes
        -----
        Constructor inspection follows decorated signatures. Variadic
        collectors are omitted because they do not declare individual options.
        """
        full_name = get_fully_qualified_type_name(cls)
        component_type_name = cls._get_component_type().__name__
        required_args = list()
        optional_args = dict()

        # signature follows __wrapped__, exposing the constructor contract
        # rather than the *args/**kwargs used by instrumentation wrappers.
        for parameter in inspect.signature(cls.__init__).parameters.values():
            if parameter.name == "self" or parameter.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            # Missing defaults use a sentinel, not truthiness: None, False,
            # zero, and empty containers are all legitimate optional defaults.
            if parameter.default is inspect.Parameter.empty:
                required_args.append(parameter.name)
            else:
                optional_args[parameter.name] = parameter.default

        return dedent(
            f"""
            {component_type_name}: {full_name}
            Required Parameters: {required_args}
            Optional Parameters: {optional_args}
            """)


def info(component: Component):
    if getattr(component, "__info__"):
        print(component.__info__())


def get_fully_qualified_type_name(type_obj: type):

    try:
        module_name = type_obj.__module__ + "."
    finally:
        pass

    if not module_name:
        return type_obj.__name__

    return module_name + type_obj.__name__


def has_required_func_args(func: FunctionType) -> bool:
    pars = inspect.signature(func).parameters

    for arg_name, par in pars.items():
        if arg_name == "self":
            continue

        if par.default is inspect._empty:
            return True

    return False


def get_obj_init_args(class_obj: type) -> dict[str, Any]:
    """Return named constructor defaults, following decorated signatures.

    Parameters
    ----------
    class_obj : type
        Class whose constructor parameters should be inspected.

    Returns
    -------
    dict[str, Any]
        Named constructor parameters mapped to their defaults. Required
        parameters are mapped to ``None`` for compatibility with Config.
        The receiver and variadic collectors are omitted.

    Notes
    -----
    Positional and keyword-only parameters are included. The legacy ``None``
    marker cannot distinguish a required parameter from an optional parameter
    whose actual default is ``None``; inspect the signature when that
    distinction is needed, as :meth:`Component.__info__` does.
    """
    # inspect.signature follows wrapper metadata, unlike inspection of the
    # wrapper's own argument list, so instrumentation does not hide defaults.
    return {
        parameter.name: (
            None if parameter.default is inspect.Parameter.empty
            else parameter.default
        )
        for parameter in inspect.signature(class_obj.__init__).parameters.values()
        if parameter.name != "self" and parameter.kind not in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        )
    }


def retrieve_arg_name(var, max_steps: int = 2):
    frame = inspect.currentframe()

    if not frame:
        return

    steps = 0
    var_name = None

    while steps < max_steps:
        frame = frame.f_back # type: ignore
        local_vars = frame.f_locals.items() # type: ignore
        var_names = [var_name for var_name, var_val in local_vars if var_val is var]
        steps += 1

        if var_names:
            var_name = var_names[0]
        else:
            break

    return var_name



def _get_module_name(module: ModuleType | Namespace):
    module_name = module.__name__
    is_internal = _is_internal_module(module_name)

    if is_internal:
        return module_name

    return module.__file__

def _get_component_module_name(component: Component):
    module = inspect.getmodule(component)

    if module:
        return _get_module_name(module)

    return component.__module__

def _is_internal_module(module_reference: str) -> bool:

    try:
        importlib.import_module(module_reference)
        return True
    except ModuleNotFoundError:
        return False
    except Exception as e:
        raise e

def _get_full_component_name(module_name: str, component_type_name: str) -> str:
    return module_name + "." + component_type_name

def _get_full_instantiated_name(module_name: str, component_type_name: str, scheme_name: str) -> str:
    return ".".join([module_name, component_type_name, scheme_name])

def _get_component_name_parts(full_component_name) -> tuple[str, str, str]:
    component_parts = full_component_name.split(".")
    return ".".join(component_parts[:-2]), component_parts[-2], component_parts[-1]


def convert_mdf_to_ddf(df):
    ddf = pd.pivot_table(data=df.stack(dropna=False).reset_index(),
                         index='Dataset',
                         columns=['SPI-1', 'SPI-2'],
                         dropna=False).T.droplevel(0)
    return ddf


def is_jpype_jvm_available():
    """Check whether a JVM is accessible via Jpype"""
    try:
        import jpype as jp # pyright: ignore[reportMissingImports] # pylint: disable=E0401
        if not jp.isJVMStarted():
            jarloc = (os.path.dirname(os.path.abspath(__file__)) + "/lib/jidt/infodynamics.jar")
            # if JVM not started, start a session
            print(f"Starting JVM with java class {jarloc}.")
            jp.startJVM(jp.getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarloc)
        return True
    except Exception as e:
        print(f"Jpype JVM not available: {e}")
        return False


def is_octave_available():
    """Check whether octave is available"""
    try:
        from oct2py import Oct2Py # pyright: ignore[reportMissingImports] # pylint: disable=E0401
        oc = Oct2Py()
        oc.exit()
        return True
    except Exception as e: # pyright: ignore
        print(f"Octave not available: {e}")
        return False


def get_available_optional_deps():
    """Bundle all the optional dependency checks together."""

    optional_deps = list()
    if is_octave_available():
        optional_deps.append("octave")

    if is_jpype_jvm_available():
        optional_deps.append("java")

    return optional_deps


@typechecked
def copy_array(array: np.ndarray | Iterable[Number]) -> np.ndarray:
    nparray = np.array(array) if not isinstance(array, np.ndarray) \
        else array

    result = nparray.copy()
    return result
