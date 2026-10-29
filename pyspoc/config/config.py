from __future__ import annotations

import yaml
import warnings
import os
import re
import inspect
import importlib
import json

from typing import Literal, Any, TypeVar, TypeAlias, TypeGuard, Iterable, Generator, cast, get_args
from types import ModuleType
from runpy import run_path
from argparse import Namespace
from typeguard import TypeCheckError
from pathlib import Path
from pydantic import ValidationError
from warnings import warn

import pyspoc._argchecking as argchk
import pyspoc.component._base as pyb
from pyspoc.settings import settings
from pyspoc._utils import lib
from pyspoc.component._base import Component, ComponentModule
from pyspoc.component.rstatistics.base import ReducedStatistic
from pyspoc.component.statistics.base import Statistic
from pyspoc.component.reducers.base import Reducer
from pyspoc.exceptions import OptionalDependencyMissingError, InvalidComponentConfigurationError
from ._yaml_schemata import LongConfig, ShortConfig, SchemesConfig
from ._config_schema import ComponentStore, MappingRule, MappingTrunk, MappingTree


_TICK_CHAR = u'\u2714'
_CROSS_CHAR = u'\u2716'
TComponent = TypeVar("TComponent", bound=Component)
ComponentGroup: TypeAlias = Literal[
    "Statistics",
    "ReducedStatistics",
    "Reducers"]

class Config:

    """
    Provides a construct for run configurations.

    Existing configurations can be imported via:
        - a YAML configuration file.
        - a JSON configuration file.
        - an in-memory dict object.

    Configurations can be modified or new configurations constructed using provided methods.

    Current configuration state can be exported to YAML or JSON files for later use.

    Arguments:
        name (string): Unique name for the configuration. This allows a single Calculator class to serve
        multiple configurations.
    """


    class LongKeyDumper(yaml.SafeDumper):
        def check_simple_key(self):
            return True

        def ignore_aliases(self, data):
            return True


    def __init__(self, name: str):
        self._name = name
        self._config_dict = dict()
        self._reducer_filtered_stats = dict()
        self._reducer_filters = dict()
        self._config_store = ComponentStore(
            Statistics={},
            ReducedStatistics={},
            Reducers={})
        

    @property
    def name(self) -> str:
        """
        Unique name of the configuration.
        """
        return self._name
    

    @property
    def scheme(self):
        """
        A Python dictionary representation of the configuration scheme.
        """
        return self._config_store
    

    @property
    def statistics(self) -> dict:
        """
        A Python dictionary representation of the Statistics configuration.
        """
        return self._config_store.Statistics.copy()
    

    @property
    def reducers(self) -> dict:
        """
        A Python dictionary representation of the Reducers configuration.
        """
        return self._config_store.Reducers.copy()
    
    
    @property
    def reduced_statistics(self) -> dict:
        """
        A Python dictionary representation of the ReducedStatistics configuration.
        """
        return self._config_store.ReducedStatistics.copy()
    

    def get_reducer_filtered_statistics(self, reducer: Reducer) -> dict:
        """
        A list of Statistics that a given Reducer is applied to.
        """
        return self._reducer_filtered_stats.get(reducer, {})
    

    def get_reducer_filters(self, reducer: Reducer) -> dict:
        """
        Text-based filter limiting the Statistics a Reducer is applied to.
        """
        return self._reducer_filters.get(reducer, {})
    
    
    @classmethod
    def from_yaml(cls, name: str, yaml_string: str) -> Config:
        """
        Creates and returns a new Config object from the provided YAML string.

        Parameters
        ----------
            name : str
                A reference name for the configuration.
            yaml_string : str
                A string in YAML format representing the configuration.
        """
        instance = cls(name)
        print("Registering YAML string.")
        instance._config_dict = yaml.load(yaml_string, Loader=yaml.FullLoader)
        instance._process_config_dict()
        return instance
    
    
    @classmethod
    def from_yaml_file(cls, name: str, yaml_file_path: str | Path) -> Config:
        """
        Creates and returns a new Config object from a YAML file.

        Parameters
        ----------
            name : str
                A reference name for the configuration.
            yaml_file_path : str
                A file path pointing to a valid configuration in YAML format.
        """
        instance = cls(name)
        print("Registering YAML configuration file: {}.".format(yaml_file_path))

        with open(yaml_file_path) as yaml_io:
            config_dict = yaml.load(yaml_io, Loader=yaml.FullLoader)

        instance._config_dict = config_dict
        instance._process_config_dict()
        return instance
    

    @classmethod
    def from_internal(cls, name: str) -> Config:
        """
        Creates and returns a new Config object from a pre-defined internal configuration provided
        by the framework.

        Parameters
        ----------
            name : str
                Name of the pre-defined configuration. Options available are:
                    fabfour
                    fast
                    basic
                    causal
                    distance
                    misc
        """
        
        yaml_path = Path(f"run_config/{name}.yaml")
        yaml_resource = lib.get_library_resource(yaml_path)
        return cls.from_yaml_file(name, yaml_resource)


    @classmethod
    def from_dict(
        cls,
        name: str,
        config_dict: dict[str, dict[str, dict[str, dict[str, Any]]]]) -> Config:
        """
        Creates and returns a new Config object from a Python dictionary.

        Parameters
        ----------
            name : str
                A reference name for the configuration.
            config_dict : dict[str, dict[str, dict[str, dict[str, Any]]]]
                A Python nested dictionary object representing a valid configuration.
        """
        instance = cls(name)
        print("Registering configuration dictionary object.")
        instance._config_dict = config_dict.copy()
        instance._process_config_dict()
        return instance
    

    @classmethod
    def from_json_file(cls, name: str, json_file_path: str) -> Config:
        """
        Creates and returns a new Config object from a JSON file.

        Parameters
        ----------
            name : str
                A reference name for the configuration.
            json_file_path : str
                A file path pointing to a valid configuration in JSON format.
        """
        instance = cls(name)
        print("Registering YAML configuration file: {}.".format(json_file_path))
        with open(json_file_path, "r") as f:
            instance._config_dict = json.load(f)

        instance._process_config_dict()
        return instance
    

    def to_yaml(self):
        statistics = self.statistics.values()
        reducers = self.reducers.values()
        reduced_statistics = self.reduced_statistics.values()

        if not statistics and (not reduced_statistics):
            warnings.warn("No Statistics or ReducedStatistics have been loaded. Skipping.")
            return

        if statistics and (not reducers):
            warnings.warn("No Reducers have been loaded for the loaded Statistics."
                            " Skipping.")
            return
        
        if reduced_statistics and reducers and (not statistics):
            warnings.warn("Reducers were loaded without any loaded Statistics."
                            " These will be ignored.")

        yaml_dict = {
            "Statistics": dict(),
            "Reducers": dict(),
            "ReducedStatistics": dict()
        }

        for stat in statistics:
            self._add_export_component(yaml_dict, stat, Statistic)

        for reducer in reducers:
            self._add_export_component(yaml_dict, reducer, Reducer)
            self._add_export_reducer_statistic_filters(yaml_dict, reducer)

        for reduced_statistic in reduced_statistics:
            self._add_export_component(yaml_dict, reduced_statistic, ReducedStatistic)

        yaml_text = yaml.dump(yaml_dict,
            Dumper=Config.LongKeyDumper,
            sort_keys=False,
            width=10**6)
        return yaml_text
    
    
    def export_yaml(self, export_path: str = None):
        yaml_text = self.to_yaml()

        if not yaml_text:
            return

        if export_path is None:
            export_path = os.getcwd() + f"\\{self.name}.yaml"

        with open(export_path, "w") as f:
            f.write(yaml_text)


    def add_statistic(
            self,
            statistic: Statistic,
            scheme_name: str):
        """
        Adds a new Statistic to the configuration.

        Parameters
        ------------
        statistic : Statistic
            A Statistic object to add to the configuration.
        scheme_name : str
            A unique name for the scheme attached to the object.
        """

        component_archetype = (ReducedStatistic
            if isinstance(statistic, ReducedStatistic) else Statistic)
        self._get_config_section(statistic, component_archetype, scheme_name)


    def add_statistic_by_name(
            self,
            module_reference: str,
            statistic_name: str,
            statistic_params: dict[str, Any],
            scheme_name: str,
            refresh_module: bool = False):
        """
        Adds a new Statistic to the configuration based on the specified name and
        configuration parameters.

        Parameters
        ------------
        module_reference : str
            A reference to the module containing the Statistic class.
            The following formats are supported:
                - Dot format for internal PySS modules (ie. "pyspoc.statistics.basic")
                - File path (ie. "/path/to/my/module.py")
        statistic_name : str
            Name of the Statistic class.
        statistic_params: dict:
            A dictionary of parameters for instantiating the Statistic object.
        scheme_name : str
            A unique name for the scheme attached to the object.
        refresh_module : bool, default=False
            If set to true, PySS will reimport the module if already loaded.
        """

        component = self._get_component_by_name(module_reference,
                                                 Statistic,
                                                 statistic_name,
                                                 statistic_params,
                                                 scheme_name,
                                                 refresh_module)
        
        if not component:
            return
        
        statistic = cast(Statistic, component)
        self.add_statistic(statistic, scheme_name)


    def add_reduced_statistic(
            self,
            reduced_statistic: ReducedStatistic,
            scheme_name: str):
        """
        Adds a new ReducedStatistic to the configuration.

        Parameters
        ----------
        reduced_statistic : ReducedStatistic
            A ReducedStatistic object to add to the configuration.
        scheme_name : str
            A unique name for the scheme attached to the object.
        """
        
        self._get_config_section(reduced_statistic, ReducedStatistic, scheme_name)


    def add_reduced_statistic_by_name(
            self,
            module_reference: str,
            reduced_statistic_name: str,
            reduced_statistic_params: dict[str, Any],
            scheme_name: str,
            refresh_module: bool = False):
        
        """
        Adds a new ReducedStatistic to the configuration based on the specified name and
        configuration parameters.

        Parameters
        ----------
        module_reference : str
            A reference to the module containing the ReducedStatistic class.
            The following formats are supported:
                - Dot format for internal PySS modules (ie. "pyspoc.rstatistics.basic")
                - File path (ie. "/path/to/my/module.py")
        reduced_statistic_name : str
            Name of the ReducedStatistic class.
        reduced_statistic_params : dict[str, Any]
            A dictionary of parameters for instantiating the ReducedStatistic object.
        scheme_name : str
            A unique name for the scheme attached to the object.
        refresh_module : bool, default=False
            If set to true, PySS will reimport the module if already loaded.
        """
        
        component = self._get_component_by_name(
            module_reference,
            Statistic,
            reduced_statistic_name,
            reduced_statistic_params,
            scheme_name,
            refresh_module)
        
        if not component:
            return
            
        rstatistic = cast(ReducedStatistic, component)
        self.add_reduced_statistic(rstatistic, scheme_name)


    def add_reducer(
            self,
            reducer: Reducer,
            scheme_name: str,
            statistic_filters: Iterable[str] | str | None = None):
        """
        Adds a new Reducer to the configuration.

        Parameters
        ----------
        reducer : Reducer
            A Reducer object to add to the configuration.
        scheme_name : str
            A unique name for the scheme attached to the object.
        statistic_filters : str or Iterable[str], optional
            A set of filters, restricting the application of the Reducer to specific
            Statistics within the configuration. Please see  for more information.
        """

        self._get_config_section(reducer, Reducer, scheme_name)

        if statistic_filters:
            self._add_reducer_statistic_filters(reducer, statistic_filters)


    def add_reducer_by_name(
            self,
            module_reference: str,
            reducer_name: str,
            reducer_params: dict[str, Any],
            scheme_name: str,
            statistic_filters: Iterable[str] | str | None = None,
            refresh_module: bool = False):
        """
        Adds a new Reducer to the configuration based on the specified name and
        configuration parameters.

        Parameters
        ----------
        module_reference : str
            A reference to the module containing the Reducer class.
            The following formats are supported:
                - Dot format for internal PySS modules (ie. "pyspoc.reducers.basic")
                - File path (ie. "/path/to/my/module.py")
        reducer_name : str
            Name of the Reducer class.
        reducer_params : dict[str, Any]
            A dictionary of parameters for instantiating the Reducer object.
        scheme_name : str
            A unique name for the scheme attached to the object.
        refresh_module : bool, default=False
            If set to true, PySS will reimport the module if already loaded.
        """

        component = self._get_component_by_name(
            module_reference,
            Reducer,
            reducer_name,
            reducer_params,
            scheme_name,
            refresh_module)
        
        if not component:
            return
        
        reducer = cast(Reducer, component)
        self.add_reducer(reducer, scheme_name, statistic_filters)


    def _get_component_by_name(
            self,
            module_reference: str,
            component_archetype: type,
            component_name: str,
            component_params: dict,
            scheme_name: str,
            refresh_module: bool = False) -> Component:

        module = self._get_module(module_reference, refresh_module=refresh_module)

        if not module:
            raise ValueError(
                f"The module {module_reference} specified could not be found.\n"
                "- If a module name was specified, please check if the module is accessible from "
                "your current environment.\n"
                "- If a relative path was specified, check the path is valid from the current "
                "working directory (os.getcwd()).\n"
                "- If an absolute path was specified, check the file exists and is a valid python "
                "module.")

        component_archetype_name = component_archetype.__name__
        statistic_class = self._get_component_class(
            Statistic, component_name, component_params, module)

        component = self._instantiate_component(
            module_reference,
            component_name,
            component_archetype_name,
            statistic_class,
            scheme_name,
            component_params)
        return component
    

    def remove_statistic(self, statistic: Statistic):
        self._remove_component(statistic, Statistic)


    def remove_statistic_by_name(
            self,
            module_reference: str,
            statistic_name: str,
            scheme_name: str):
        
        self._remove_component_by_name(
            Statistic,
            module_reference,
            statistic_name,
            scheme_name)


    def remove_reduced_statistic(self, reduced_statistic: ReducedStatistic):
        self._remove_component(reduced_statistic, ReducedStatistic)


    def remove_reduced_statistic_by_name(self,
                                         module_reference: str,
                                         reduced_statistic_name: str,
                                         scheme_name: str):
        
        self._remove_component_by_name(ReducedStatistic,
                                        module_reference,
                                        reduced_statistic_name,
                                        scheme_name)
        

    def remove_reducer(self, reducer: Reducer):
        reducer = self._remove_component(reducer, Reducer)
        self._reducer_filtered_stats.pop(reducer, None)


    def remove_reducer_by_name(
            self,
            module_reference: str,
            reducer_name: str,
            scheme_name: str):
        
        reducer = self._remove_component_by_name(
            Reducer, module_reference, reducer_name, scheme_name)

        if reducer:
            self._reducer_filtered_stats.pop(reducer, None)


    def _remove_component(self,
                           component: Component,
                           component_archetype: type):

        module_reference = self._get_component_module_name(component)
        component_type = type(component)
        component_type_name = component_type.__name__
        scheme_name = component.scheme
        return self._remove_component_by_name(
            component_archetype,
            module_reference,
            component_type_name,
            scheme_name)
    

    def _remove_component_by_name(
            self,
            component_archetype: type,
            module_reference: str,
            component_type_name: str,
            scheme_name: str):

        component_archetype_name = component_archetype.__name__
        full_instance_name = self._get_full_instantiated_name(
            module_reference, component_type_name, scheme_name)

        result = self._config_store[component_archetype_name].pop(
            full_instance_name, None)

        if not result:
            warnings.warn(f"The {component_archetype_name} {component_type_name} was not found in the configuration.")
            return
        
        component_type_name = type(result).__name__
        scheme_name = result.scheme
        print(f"  {component_archetype_name} {component_type_name} scheme '{scheme_name}' "
                f"removed successfully.")
        return result


    def _add_export_component(
            self,
            yaml_dict: dict,
            component: Component,
            component_archetype: type):
        
        component_archetype_name = component_archetype.__name__ + "s"
        module_name = self._get_component_module_name(component)
        scheme_name = component.scheme
        params = component.params

        if module_name not in yaml_dict[component_archetype_name]:
            yaml_dict[component_archetype_name][module_name] = dict()

        module_dict = yaml_dict[component_archetype_name][module_name]
        component_name = type(component).__name__

        if component_name not in module_dict:
            module_dict[component_name] = dict()
            module_dict[component_name]["schemes"] = dict()

        schemes_dict = module_dict[component_name]["schemes"]
        schemes_dict[scheme_name] = None if not params else params

    
    def _process_config_dict(self, config_dict: dict):
        print("Building internal configuration.")
        self._component_store = self._build_component_store(config_dict)


    @classmethod
    def _build_component_store(
            cls,
            config_dict: dict) -> ComponentStore:

        try:
            config = LongConfig(**config_dict)
        except ValidationError as long_e:
            try:
                config = ShortConfig(**config_dict)
            except ValidationError as short_e:
                raise short_e from long_e

        component_store = ComponentStore(
            Statistics = {},
            ReducedStatistics = {},
            Reducers = {})

        for component_type_key, scheme_name, component_group, component in cls._yield_built_components(config):
            component_group_store = getattr(component_store, component_group)

            if not component_group_store:
                continue

            component_type_dict = component_group_store.get(component_type_key)

            if not component_type_dict:
                component_group_store[component_type_key] = {}
            
            component_group_store[component_type_key][scheme_name] = component

        return component_store


    @classmethod
    def _yield_built_components(
            cls,
            config: LongConfig | ShortConfig
    ) -> Generator[tuple[tuple[str | None, str], str, ComponentGroup, Component]]:

        if isinstance(config, LongConfig):
            return cls._yield_built_components_from_long(config)

        if isinstance(config, ShortConfig):
            return cls._yield_built_components_from_short(config)

        raise TypeError(
            f"Invalid configuration type {type(config)}.")
    
    
    @classmethod
    def _yield_built_components_from_long(
            cls,
            long_config: LongConfig
    ) -> Generator[tuple[tuple[str | None, str], str, ComponentGroup, Component]]:

        long_config_dict = long_config.model_dump()
        for component_group, module_dict in long_config_dict.items():

            if not cls._is_component_group(component_group):
                continue

            component_archetype = cls._component_group_to_archetype(component_group)

            for module_reference, component_dict in module_dict.items():
                for component_name, scheme_group in component_dict.items():
                    schemes_dict = scheme_group["schemes"]

                    for scheme_name, scheme_params in schemes_dict.items():
                        component_triple = cls._make_component(
                            component_archetype=component_archetype,
                            module_reference=module_reference,
                            component_name=component_name,
                            scheme_name=scheme_name,
                            scheme_params=scheme_params)

                        if component_triple:
                            component_type_key, scheme, component = component_triple
                            yield component_type_key, scheme, component_group, component

    @classmethod
    def _yield_built_components_from_short(
            cls,
            short_config: ShortConfig
    ) -> Generator[tuple[tuple[str | None, str], str, ComponentGroup, Component]]:

        long_config_dict = short_config.model_dump()
        for component_group, component_dict in long_config_dict.items():
            if not cls._is_component_group(component_group):
                continue

            component_archetype = cls._component_group_to_archetype(component_group)

            for component_name, scheme_group in component_dict.items():
                name_parts = component_name.split(".")
                module_reference = None
                
                if len(name_parts) > 0:
                    component_name = name_parts[-1]
                    module_reference = ".".join(name_parts[:-1])
                    
                schemes_dict = scheme_group["schemes"]
                for scheme_name, scheme_params in schemes_dict.items():
                    component_triple = cls._make_component(
                        component_archetype=component_archetype,
                        module_reference=module_reference,
                        component_name=component_name,
                        scheme_name=scheme_name,
                        scheme_params=scheme_params)

                    if component_triple:
                        component_type_key, scheme, component = component_triple
                        yield component_type_key, scheme, component_group, component

    @staticmethod
    def _make_component(
            component_archetype: type[Component],
            module_reference: str | None,
            component_name: str,
            scheme_name: str,
            scheme_params: dict[str, Any],
            error_handling: Literal["raise", "warn", "ignore"] = "raise"
    ) -> tuple[tuple[str | None, str], str, Component] | None:

        try:
            component = component_archetype.make(
                name=component_name,
                module_reference=module_reference,
                **scheme_params)

            component_type_key = (module_reference, component_name)
            return component_type_key, scheme_name, component
            
        except OptionalDependencyMissingError as e:
            print(f"  {_CROSS_CHAR} {module_reference}.{component_name} is missing optional "
                    f"dependency {e.dependency}. Skipping")
        except Exception as e:
            if error_handling == "raise":
                raise e

            if error_handling == "warn":
                warn(f"  {_CROSS_CHAR} {component_name} failed to load due to the "
                    f"following exception: {str(e)}")

    @staticmethod
    def _is_component_group(
            value: str) -> TypeGuard[ComponentGroup]:
        component_group_values = get_args(ComponentGroup)
        return value in component_group_values


    @staticmethod
    def _component_group_to_archetype(
            component_group: ComponentGroup
    ) -> type[Component]:

        match component_group:
            case "Statistics":
                return Statistic

            case "ReducedStatistics":
                return ReducedStatistic

            case "Reducers":
                return Reducer

            case _:
                raise ValueError(
                    f"Invalid component group {component_group}. "
                )

    
    def _process_mappings(self, config: LongConfig | ShortConfig):
        config_scheme = MappingTrunk(Branches={})
        
        for pos_mapping in config.Mappings.Positive:
            stat_reference = pos_mapping.statistic
            stat_key = stat_reference.module, stat_reference.name
            stat_type_dict = self._config_store.Statistics.get(stat_key)

            if not stat_type_dict:
                # Raise here.
                continue
            
            stat_scheme = stat_reference.scheme
            stats = []

            if stat_scheme:
                stat = stat_type_dict.get(stat_scheme)

                if not stat:
                    # Raise here
                    continue

                stats.append(stat)
            else:
                for _, stat in stat_type_dict.items():
                    stats.append(stat)

            for stat in stats:
                prev_tree = MappingTree(Branches={})
                config_scheme.Branches[stat] = prev_tree
                prev_trees = [prev_tree]

                for reducer_stage in pos_mapping.reducer_stages:
                    new_trees = []
                    
                    for reducer_reference in reducer_stage:
                        reducer_key = reducer_reference.module, reducer_reference.name
                        reducer_type_dict = self._config_store.Reducers.get(reducer_key)

                        if not reducer_type_dict:
                            # Raise here.
                            continue

                        reducer_scheme = reducer_reference.scheme
                        reducers = []

                        if reducer_scheme:
                            reducer = reducer_type_dict.get(reducer_scheme)

                            if not reducer:
                                # Raise here
                                continue

                            reducers.append(reducer)
                        else:
                            for _, reducer in reducer_type_dict.items():
                                reducers.append(reducer)

                        new_tree = MappingTree(Branches={reducer : None})
                        new_trees.append(new_tree)

                        for prev_tree in prev_trees:
                            for reducer in prev_tree.Branches.keys():
                                prev_tree.Branches[reducer] = new_tree

                    prev_trees = new_trees
    
    
    def _build_config_scheme(self,
                              stats_spec: dict,
                              reducers_spec: dict,
                              rstats_spec: dict):

        # Get instantiated Statistics based on configuration
        stats_generator = self._yield_instantiated_components(Statistic, stats_spec)

        # Store each Statistic
        for stat, scheme_name, _ in stats_generator:
            self._get_config_section(stat, Statistic, scheme_name)

        # Get instantiated ReducedStatistics based on configuration
        rstats_generator = self._yield_instantiated_components(ReducedStatistic, rstats_spec)

        # Store each ReducedStatistic
        for rstat, scheme_name, _ in rstats_generator:
            self._get_config_section(rstat, ReducedStatistic, scheme_name)

        # Get instantiated Reducers based on configuration
        reducers_generator = self._yield_instantiated_components(Reducer, reducers_spec)

        # Store each Reducer and all viable Statistic names after filtering
        for reducer, scheme_name, reducer_params in reducers_generator:
            self._get_config_section(reducer, Reducer, scheme_name)
            statistics = reducer_params.get("Statistics")
            self._add_reducer_statistic_filters(reducer, statistics)

    def _yield_instantiated_components(self,
                                        component_archetype: type,
                                        component_spec: dict) -> Generator[tuple[Component, str, dict]]:

        component_archetype_name = component_archetype.__name__

        # Cycle through modules and components
        for module_reference, module_components in component_spec.items():

            # Check module components is a dictionary
            try:
                argchk.check_type(module_components, dict)
            except TypeCheckError as e:
                raise RuntimeError(
                    f"Incorrect format for {component_archetype_name}s definition.") from e

            module = self._get_module(module_reference)

            # Cycle through components and component parameters
            for component_name, component_params in module_components.items():

                component_class = self._get_component_class(component_archetype,
                                                             component_name,
                                                             component_params,
                                                             module)

                if not component_class:
                    continue

                # Check component params is a dictionary

                try:
                    argchk.check_type(component_params, dict)
                except TypeCheckError as e:
                    raise RuntimeError(
                        f"Incorrect format for {component_archetype_name} {component_name} "
                        f"definition under module {module_reference}.") from e

                # Get the component configurations
                schemes = component_params.get("schemes")

                if not schemes:
                    raise ValueError(f"{component_archetype_name} {component_name} under module {module_reference} "
                                     f"is missing 'schemes' definition.")

                # Check configs is a dictionary
                try:
                    argchk.check_type(schemes, dict)
                except TypeCheckError as e:
                    raise RuntimeError(
                        f"Incorrect format for {component_archetype_name} {component_name} "
                        f"'scheme' definition under module {module_reference}.") from e

                # Cycle config and config parameters
                for scheme_name, scheme_args in schemes.items():

                    if scheme_args is None:
                        scheme_args = dict()
                    
                    # Check config params is a dictionary
                    try:
                        argchk.check_type(scheme_args, dict)
                    except TypeCheckError as e:
                        raise RuntimeError(
                            f"Incorrect format for {component_archetype_name} {component_name} "
                            f"'schemes' definition under module {module_reference}.") from e

                    # Instantiate the component
                    component = self._instantiate_component(module_reference,
                                                             component_name,
                                                             component_archetype_name,
                                                             component_class,
                                                             scheme_name,
                                                             scheme_args)

                    # Return the component instance, scheme name and parameters.
                    yield component, scheme_name, component_params

    @staticmethod
    def _instantiate_component(
        module_reference: str,
        component_name: str,
        component_archetype_name: str,
        component_class: type,
        scheme_name: str,
        scheme_args: dict) -> Component:

        # Get required constructor arguments
        args = pyb.get_obj_init_args(component_class)
        missing_args = list()

        # Check all required arguments are provided in the configuration
        for arg, default in args.items():
            if arg not in scheme_args and default is None:
                missing_args.append(arg)

        # If required arguments are missing, throw error.
        if missing_args:
            raise ValueError(f"{component_archetype_name} {component_name} configuration '{scheme_name}' "
                             + f"under module '{module_reference}' is missing the arguments:"
                             + f"\n\t"
                             + "\n\t".join(missing_args))

        # Instantiate the component class with configured arguments.
        component: Component = component_class(**scheme_args)
        return component

    def _add_reducer_statistic_filters(self,
                                        reducer: Reducer,
                                        statistic_filters: Iterable[str] | str | None):

        reducer_name = type(reducer).__name__
        filtered_stats = set()

        if not statistic_filters:
            return

        statistic_list = self._yaml_str_list_to_list(statistic_filters)
        module_name = self._get_component_module_name(reducer)

        try:
            argchk.check_type(statistic_list, Iterable[str])
        except TypeCheckError as e:
            raise RuntimeError(
                f"Incorrect format for Reducer {reducer_name} 'Statistics' "
                f"definition under module {module_name}.") from e

        available_stat_names = self._config_store["Statistic"].keys()

        for stat_name in statistic_list:
            esc_stat_name = "^" + re.escape(stat_name).replace("\\*", ".*")
            viable_stat_names = [stat for stat in available_stat_names if re.match(esc_stat_name, stat)]

            for viable_stat_name in viable_stat_names:
                filtered_stats.add(viable_stat_name)

        self._reducer_filtered_stats[reducer] = filtered_stats
        self._reducer_filters[reducer] = statistic_filters


    @classmethod
    def _get_module_name(cls, module: ModuleType | Namespace):
        module_name = module.__name__
        is_internal = cls._is_internal_module(module_name)

        if is_internal:
            return module_name

        return module.__file__

    @classmethod
    def _get_component_module_name(cls, component: Component):
        module = inspect.getmodule(component)

        if module:
            return cls._get_module_name(module)

        return component.__module__

    @staticmethod
    def _is_internal_module(module_reference: str) -> bool:

        try:
            importlib.import_module(module_reference)
            return True
        except ModuleNotFoundError:
            return False
        except Exception as e:
            raise e

    @staticmethod
    def _get_full_component_name(module_name: str, component_type_name: str) -> str:
        return module_name + "." + component_type_name

    @staticmethod
    def _get_full_instantiated_name(module_name: str, component_type_name: str, scheme_name: str) -> str:
        return ".".join([module_name, component_type_name, scheme_name])

    @staticmethod
    def _get_component_name_parts(full_component_name) -> tuple[str, str, str]:
        component_parts = full_component_name.split(".")
        return ".".join(component_parts[:-2]), component_parts[-2], component_parts[-1]

    @staticmethod
    def _yaml_str_list_to_list(str_list: str | list) -> list:
        return re.split(r"\s+", str_list) if isinstance(str_list, str) else str_list
