from __future__ import annotations

import yaml
import warnings
import os
import re
import inspect
import importlib
import json
import pyspoc._base as pyb

from typing import Iterable, Generator, cast
from types import ModuleType
from runpy import run_path
from argparse import Namespace

from pyspoc._base import Component
from pyspoc.statistic import Statistic, ReducedStatistic
from pyspoc.reducer import Reducer


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

    _TICK_CHAR = u'\u2714'
    _CROSS_CHAR = u'\u2716'

    _cached_modules = dict()
    _cached_module_classes = dict()
    _available_dependencies = pyb.get_available_optional_deps()

    def __init__(self, name: str):
        self._name = name
        self._config_dict = dict()
        self._reducer_filtered_stats = dict()
        self._reducer_filters = dict()

        component_types = ["Statistic", "Reducer", "ReducedStatistic"]
        self._config_scheme = dict()
        self._cached_module_classes = dict()

        for component_type in component_types:
            self._config_scheme[component_type] = dict()
            self._cached_module_classes[component_type] = dict()

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
        return self._config_scheme

    @property
    def statistics(self) -> dict:
        """
        A Python dictionary representation of the Statistics configuration.
        """
        return self._config_scheme.get("Statistic", {})

    @property
    def reducers(self) -> dict:
        """
        A Python dictionary representation of the Reducers configuration.
        """
        return self._config_scheme.get("Reducer", {})
    
    @property
    def reduced_statistics(self) -> dict:
        """
        A Python dictionary representation of the ReducedStatistics configuration.
        """
        return self._config_scheme.get("ReducedStatistic", {})

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

        Arguments:
            name (string): A reference name for the configuration.
            yaml_string (string): A string in YAML format representing the configuration.
        """
        instance = cls(name)
        print("Registering YAML string.")
        instance._config_dict = yaml.load(yaml_string, Loader=yaml.FullLoader)
        instance._process_config_file()
        return instance
    
    @classmethod
    def from_yaml_file(cls, name: str, yaml_file_path: str) -> Config:
        """
        Creates and returns a new Config object from a YAML file.

        Arguments:
            name (string): A reference name for the configuration.
            yaml_file_path (string): A file path pointing to a valid configuration in YAML format.
        """
        instance = cls(name)
        print("Registering YAML configuration file: {}.".format(yaml_file_path))

        with open(yaml_file_path) as yaml_io:
            config_dict = yaml.load(yaml_io, Loader=yaml.FullLoader)

        instance._config_dict = config_dict
        instance._process_config_file()
        return instance

    @classmethod
    def from_internal(cls, name: str) -> Config:
        """
        Creates and returns a new Config object from a pre-defined internal configuration provided by the framework.

        Arguments:
            name (string): Name of the pre-defined configuration. Options available are:
                fabfour
                fast
                basic
                causal
                distance
                misc
        """
        yaml_path = f"../run_config/{name}.yaml"
        return cls.from_yaml_file(name, yaml_path)

    @classmethod
    def from_archetypes(cls,
                        name: str,
                        statistic_archetypes: list[str] | None = None,
                        reducer_archetypes: list[str] | None = None,
                        scalar_statistic_archetypes: list[str] | None = None) -> Config:
                
        if not scalar_statistic_archetypes:
            scalar_statistic_archetypes = list()

        if not statistic_archetypes:
            statistic_archetypes = list()

        if not reducer_archetypes:
            reducer_archetypes = list()
        
        is_valid_config = any([
           scalar_statistic_archetypes,
           statistic_archetypes and reducer_archetypes
        ])
        
        if not is_valid_config:
            raise ValueError("At least one of the following argument inputs are required to produce a valid configuration\n"
                             " - statistic_archetypes and reducer_archetypes must be provided.\n"
                             " - reduced_statistic_archetypes must be provided.")
        
        instance = cls(name)
        component_dirs = ["statistics", "reducers", "rstatistics"]
        component_classes = [Statistic, Reducer, ReducedStatistic]
        all_archetypes = [statistic_archetypes, reducer_archetypes, scalar_statistic_archetypes]
        archetype_triples = zip(component_dirs, component_classes, all_archetypes)
        
        for dir, component_class, archetypes in archetype_triples:
            for archetype in archetypes:
                module_reference = f"pyspoc.{dir}.{archetype}"
                module = cls._get_module(module_reference)
                instance._cached_modules[module_reference] = module

                for module_obj in cls._get_components_from_module(module):
                    instance._add_component(module_obj, component_class, "std")

        if instance.statistics and instance.reducers:
            return instance
        
        if instance.reduced_statistics:
            return instance

        raise ValueError("The selected archetypes did not result in a valid configuration. "
                         "Please report the issue to " + pyb.PYSS_GITHUB_ISSUES_URL)

    @staticmethod
    def _get_components_from_module(module: ModuleType | Namespace) -> Generator[Component]:

        for name, module_obj in module.__dict__.items():

            if not inspect.isclass(module_obj):
                continue
            
            if not issubclass(module_obj, Component):
                continue

            if hasattr(module_obj, "__abstractmethods__"):
                abstract_methods = getattr(module_obj, "__abstractmethods__")
                
                if abstract_methods:
                    continue

            if not pyb.has_required_func_args(module_obj.__init__):
                yield module_obj()

    @classmethod
    def from_dict(cls, name: str, config_dict: dict) -> Config:
        """
        Creates and returns a new Config object from a Python dictionary.

        Arguments:
            name (string): A reference name for the configuration.
            config_dict (dictionary): A Python dictionary object representing a valid configuration.
        """
        instance = cls(name)
        print("Registering configuration dictionary object.")
        instance._config_dict = config_dict
        instance._process_config_file()
        return instance

    @classmethod
    def from_json_file(cls, name: str, json_file_path: str) -> Config:
        """
        Creates and returns a new Config object from a JSON file.

        Arguments:
            name (string): A reference name for the configuration.
            json_file_path (string): A file path pointing to a valid configuration in JSON format.
        """
        instance = cls(name)
        print("Registering YAML configuration file: {}.".format(json_file_path))
        with open(json_file_path, "r") as f:
            instance._config_dict = json.load(f)

        instance._process_config_file()
        return instance

    def add_statistic(self,
                      statistic: Statistic,
                      scheme_name: str):
        """
        Adds a new Statistic to the configuration.

        Arguments:
            statistic (Statistic): A Statistic object to add to the configuration.
            scheme_name (string): A unique name for the scheme attached to the object.
        """

        component_archetype = ReducedStatistic if isinstance(statistic, ReducedStatistic) else Statistic
        self._add_component(statistic, component_archetype, scheme_name)

    def add_statistic_by_name(self,
                              module_reference: str,
                              statistic_name: str,
                              statistic_params: dict,
                              scheme_name: str,
                              refresh_module: bool = False):
        """
        Adds a new Statistic to the configuration based on the specified name and configuration parameters.

        Arguments:
            module_reference (string): A reference to the module containing the Statistic class.
                The following formats are supported:
                    - Dot format for internal PySS modules (ie. "pyspoc.statistics.basic")
                    - File path (ie. "/path/to/my/module.py")
            statistic_name (string): Name of the Statistic class.
            statistic_params (dictionary): A dictionary of parameters for instantiating the Statistic object.
            scheme_name (string): A unique name for the scheme attached to the object.
            refresh_module (bool): If set to true, PySS will reimport the module if already loaded (default: False).
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

    def add_reduced_statistic(self,
                              reduced_statistic: ReducedStatistic,
                              scheme_name: str):
        """
        Adds a new ReducedStatistic to the configuration.

        Arguments:
            reduced_statistic (ReducedStatistic): A ReducedStatistic object to add to the configuration.
            scheme_name (string): A unique name for the scheme attached to the object.
        """
        
        self._add_component(reduced_statistic, ReducedStatistic, scheme_name)

    def add_reduced_statistic_by_name(self,
                                      module_reference: str,
                                      reduced_statistic_name: str,
                                      reduced_statistic_params: dict,
                                      scheme_name: str,
                                      refresh_module: bool = False):
        
        """
        Adds a new ReducedStatistic to the configuration based on the specified name and configuration parameters.

        Arguments:
            module_reference (string): A reference to the module containing the ReducedStatistic class.
                The following formats are supported:
                    - Dot format for internal PySS modules (ie. "pyspoc.rstatistics.basic")
                    - File path (ie. "/path/to/my/module.py")
            reduced_statistic_name (string): Name of the ReducedStatistic class.
            reduced_statistic_params (dictionary): A dictionary of parameters for instantiating the ReducedStatistic object.
            scheme_name (string): A unique name for the scheme attached to the object.
            refresh_module (bool): If set to true, PySS will reimport the module if already loaded (default: False).
        """        
        
        component = self._get_component_by_name(module_reference,
                                                 Statistic,
                                                 reduced_statistic_name,
                                                 reduced_statistic_params,
                                                 scheme_name,
                                                 refresh_module)
        
        if not component:
            return
            
        rstatistic = cast(ReducedStatistic, component)
        self.add_reduced_statistic(rstatistic, scheme_name)

    def add_reducer(self,
                    reducer: Reducer,
                    scheme_name: str,
                    statistic_filters: Iterable[str] | str | None = None):
        """
        Adds a new Reducer to the configuration.

        Arguments:
            reducer (Reducer): A Reducer object to add to the configuration.
            scheme_name (string): A unique name for the scheme attached to the object.
            statistic_filters (str, Iterable(str)) (default: None): A set of filters, restricting the application of the Reducer
                to specific Statistics within the configuration. Please see  for more information.
        """

        self._add_component(reducer, Reducer, scheme_name)

        if statistic_filters:
            self._add_reducer_statistic_filters(reducer, statistic_filters)

    def add_reducer_by_name(self,
                            module_reference: str,
                            reducer_name: str,
                            reducer_params: dict,
                            scheme_name: str,
                            statistic_filters: Iterable[str] | str | None = None,
                            refresh_module: bool = False):
        """
        Adds a new Reducer to the configuration based on the specified name and configuration parameters.

        Arguments:
            module_reference (string): A reference to the module containing the Reducer class.
                The following formats are supported:
                    - Dot format for internal PySS modules (ie. "pyspoc.reducers.basic")
                    - File path (ie. "/path/to/my/module.py")
            reducer_name (string): Name of the Reducer class.
            reducer_params (dictionary): A dictionary of parameters for instantiating the Reducer object.
            scheme_name (string): A unique name for the scheme attached to the object.
            refresh_module (bool): If set to true, PySS will reimport the module if already loaded (default: False).
        """

        component = self._get_component_by_name(module_reference,
                                                 Reducer,
                                                 reducer_name,
                                                 reducer_params,
                                                 scheme_name,
                                                 refresh_module)
        
        if not component:
            return
        
        reducer = cast(Reducer, component)
        self.add_reducer(reducer, scheme_name, statistic_filters)

    def _get_component_by_name(self,
                               module_reference: str,
                               component_archetype: type,
                               component_name: str,
                               component_params: dict,
                               scheme_name: str,
                               refresh_module: bool = False) -> Component:

        module = self._get_module(module_reference, refresh_module=refresh_module)

        if not module:
            raise ValueError(f"The module {module_reference} specified could not be found.\n"
                             "- If a module name was specified, please check if the module is accessible from "
                             "your current environment.\n"
                             "- If a relative path was specified, check the path is valid from the current working "
                             "directory (os.getcwd()).\n"
                             "- If an absolute path was specified, check the file exists and is a valid python module.")

        component_archetype_name = component_archetype.__name__
        statistic_class = self._get_component_class(Statistic, component_name, component_params, module)

        component = self._instantiate_component(module_reference,
                                                 component_name,
                                                 component_archetype_name,
                                                 statistic_class,
                                                 scheme_name,
                                                 component_params)
        return component

    def remove_statistic(self, statistic: Statistic):
        self._remove_component(statistic, Statistic)

    def remove_statistic_by_name(self, module_reference: str, statistic_name: str, scheme_name: str):
        self._remove_component_by_name(Statistic, module_reference, statistic_name, scheme_name)

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

    def remove_reducer_by_name(self, module_reference: str, reducer_name: str, scheme_name: str):
        reducer = self._remove_component_by_name(Reducer, module_reference, reducer_name, scheme_name)

        if reducer:
            self._reducer_filtered_stats.pop(reducer, None)

    def _add_component(self,
                        component: Component,
                        component_archetype: type,
                        scheme_name: str):

        module = inspect.getmodule(component)
        module_reference = self._get_component_module_name(component)
        component_type = type(component)
        component_type_name = component_type.__name__

        if module_reference not in self._cached_modules:
            self._cached_modules[module_reference] = module

        component.set_scheme(scheme_name)
        component_archetype_name = component_archetype.__name__
        full_component_name = self._get_full_component_name(module_reference, component_type_name)
        self._cached_module_classes[component_archetype_name][full_component_name] = component_type

        full_instance_name = self._get_full_instantiated_name(module_reference,
                                                               component_type_name,
                                                               scheme_name)

        if full_instance_name in self._config_scheme[component_archetype_name]:
            raise ValueError(f"{component_archetype_name} {full_instance_name} already exists.")

        self._config_scheme[component_archetype_name][full_instance_name] = component
        print(f"    {self._TICK_CHAR} {component_archetype_name} {component_type_name} scheme '{scheme_name}' "
              f"added successfully.")

    def _remove_component(self,
                           component: Component,
                           component_archetype: type):

        module_reference = self._get_component_module_name(component)
        component_type = type(component)
        component_type_name = component_type.__name__
        scheme_name = component.scheme
        return self._remove_component_by_name(component_archetype,
                                               module_reference,
                                               component_type_name,
                                               scheme_name)

    def _remove_component_by_name(self,
                                   component_archetype: type,
                                   module_reference: str,
                                   component_type_name: str,
                                   scheme_name: str):

        component_archetype_name = component_archetype.__name__
        full_instance_name = self._get_full_instantiated_name(module_reference,
                                                               component_type_name,
                                                               scheme_name)

        result = self._config_scheme[component_archetype_name].pop(full_instance_name, None)

        if not result:
            warnings.warn(f"The {component_archetype_name} {component_type_name} was not found in the configuration.")
            return
        
        component_type_name = type(result).__name__
        scheme_name = result.scheme
        print(f"  {component_archetype_name} {component_type_name} scheme '{scheme_name}' "
                f"removed successfully.")
        return result

    def to_yaml(self):
        statistics = self.statistics.values()

        if not statistics:
            warnings.warn("No Statistics have been loaded. Skipping.")
            return

        reducers = self.reducers.values()

        if not reducers:
            warnings.warn("No Reducers have been loaded. Skipping.")
            return
        
        reduced_statistics = self.reduced_statistics.values()

        if not reduced_statistics:
            warnings.warn("No ReducedStatistics have been loaded. Skipping.")
            return

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

        print(yaml_dict)
        yaml_text = yaml.dump(yaml_dict, sort_keys=False)
        return yaml_text

    def export_yaml(self, export_path: str = None):
        yaml_text = self.to_yaml()

        if not yaml_text:
            return

        if export_path is None:
            export_path = os.getcwd() + f"\\{self.name}.yaml"

        with open(export_path, "w") as f:
            f.write(yaml_text)

    def _add_export_component(self, yaml_dict: dict, component: Component, component_archetype: type):
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

    def _add_export_reducer_statistic_filters(self, yaml_dict: dict, reducer: Reducer):
        reducer_dict = yaml_dict["Reducers"]
        module_name = self._get_component_module_name(reducer)
        module_dict = reducer_dict[module_name]
        reducer_name = type(reducer).__name__
        applicable_stats = self.get_reducer_filters(reducer)

        if applicable_stats:
            module_dict[reducer_name]["Statistics"] = list(applicable_stats)

    def _process_config_file(self):
        print(f"Building internal configuration.")
        stats_spec = self._get_config_top_level("Statistics")
        reducers_spec = self._get_config_top_level("Reducers")
        rstats_spec = self._get_config_top_level("ReducedStatistics")
        
        invalid_config = all(
            [not stats_spec and not reducers_spec,
             not rstats_spec]
        )
        
        if invalid_config:
            raise ValueError(f"Configuration is missing required definitions. One of the following must be fulfilled:"
                            "   - At least one Statistic and one Reducer specified."
                            "   - At least one ReducedStatistic specified.")

        self._build_config_scheme(stats_spec, reducers_spec, rstats_spec)

    def _get_config_top_level(self, level_name: str):
        level_dict = self._config_dict.get(level_name)

        if not level_dict:
            return dict()

        pyb.check_type(level_dict,
                       dict,
                       custom_error_msg=f"Configuration contains incorrect format for {level_name} definition.")
        # edited -- module.lower() wont work on all OS
        return {module: component_config for module, component_config in level_dict.items()}

    def _build_config_scheme(self,
                              stats_spec: dict,
                              reducers_spec: dict,
                              rstats_spec: dict):

        # Get instantiated Statistics based on configuration
        stats_generator = self._yield_instantiated_components(Statistic, stats_spec)

        # Store each Statistic
        for stat, scheme_name, _ in stats_generator:
            self._add_component(stat, Statistic, scheme_name)

        # Get instantiated ReducedStatistics based on configuration
        rstats_generator = self._yield_instantiated_components(ReducedStatistic, rstats_spec)

        # Store each ReducedStatistic
        for rstat, scheme_name, _ in rstats_generator:
            self._add_component(rstat, ReducedStatistic, scheme_name)

        # Get instantiated Reducers based on configuration
        reducers_generator = self._yield_instantiated_components(Reducer, reducers_spec)

        # Store each Reducer and all viable Statistic names after filtering
        for reducer, scheme_name, reducer_params in reducers_generator:
            self._add_component(reducer, Reducer, scheme_name)
            statistics = reducer_params.get("Statistics")
            self._add_reducer_statistic_filters(reducer, statistics)

    def _yield_instantiated_components(self,
                                        component_archetype: type,
                                        component_spec: dict) -> Generator[tuple[Component, str, dict]]:

        component_archetype_name = component_archetype.__name__

        # Cycle through modules and components
        for module_reference, module_components in component_spec.items():

            # Check module components is a dictionary
            pyb.check_type(module_components,
                       dict,
                       custom_error_msg=f"Incorrect format for {component_archetype_name}s definition.")

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
                pyb.check_type(component_params,
                           dict,
                           custom_error_msg=f"Incorrect format for {component_archetype_name} {component_name} "
                                            f"definition under module {module_reference}.")

                # Get the component configurations
                schemes = component_params.get("schemes")

                if not schemes:
                    raise ValueError(f"{component_archetype_name} {component_name} under module {module_reference} "
                                     f"is missing 'schemes' definition.")

                # Check configs is a dictionary
                pyb.check_type(schemes,
                               dict,
                               custom_error_msg=f"Incorrect format for {component_archetype_name} {component_name} "
                                                f"'scheme' definition under module {module_reference}.")

                # Cycle config and config parameters
                for scheme_name, scheme_args in schemes.items():

                    if scheme_args is None:
                        scheme_args = dict()
                    
                    # Check config params is a dictionary
                    pyb.check_type(scheme_args,
                                   dict,
                                   custom_error_msg=f"Incorrect format for {component_archetype_name} {component_name} "
                                                    f"'schemes' definition under module {module_reference}.")

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
    def _instantiate_component(module_reference: str,
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

        pyb.check_iterable(statistic_list,
                       str,
                       custom_error_msg=f"Incorrect format for Reducer {reducer_name} 'Statistics' "
                                        f"definition under module {module_name}.")

        available_stat_names = self._config_scheme["Statistic"].keys()

        for stat_name in statistic_list:
            esc_stat_name = "^" + re.escape(stat_name).replace("\\*", ".*")
            viable_stat_names = [stat for stat in available_stat_names if re.match(esc_stat_name, stat)]

            for viable_stat_name in viable_stat_names:
                filtered_stats.add(viable_stat_name)

        self._reducer_filtered_stats[reducer] = filtered_stats
        self._reducer_filters[reducer] = statistic_filters

    @classmethod
    def _get_module(cls,
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

        # Check user current working directory module files.
        module = cls._load_module_file(module_reference,
                                       refresh_module=refresh_module,
                                       suppress_warning=True)

        if module:
            return module

        raise ModuleNotFoundError(
            f"Unable to find the module {module_reference} within the pyspoc library, "
            "session memory or user current working directory.")
        
    @classmethod
    def _get_cached_module(cls,
                            module_reference: str) -> ModuleType | Namespace | None:

        # Check if module is already loaded.
        module = cls._cached_modules.get(module_reference)

        if module:
            print(f"  {cls._TICK_CHAR} Module {module_reference} already loaded.")
            return module
        
    @classmethod
    def _get_package_module(cls,
                             module_reference: str) -> ModuleType | None:
        # Get package modules.
        try:
            module = importlib.import_module(module_reference, __package__)
            print(f"  {cls._TICK_CHAR} Module {module_reference} loaded successfully.")
            
            if hasattr(module, "IMPORT_WARNINGS") and not pyb.IGNORE_IMPORT_WARNINGS:
                print(f"    (The following warnings were raised: {[str(w.message) for w in module.IMPORT_WARNINGS]})")

            return module
        except ModuleNotFoundError:
            pass
        except Exception as e:
            raise e
        
    @classmethod
    def _get_loaded_module(cls,
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
    def _load_module_file(cls,
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
        try:
            module_dict = run_path(module_path, run_name=module_path)
            module = Namespace(**module_dict)
            cls._cached_modules[module_path] = module

        except FileNotFoundError:
            return None

        return module

    @classmethod
    def _get_component_class(cls,
                              component_archetype: type,
                              component_name: str,
                              component_params: dict,
                              module: Namespace | ModuleType) -> type | None:

        component_type_name = component_archetype.__name__
        module_name = cls._get_module_name(module)
        full_component_name = cls._get_full_component_name(module_name, component_name)

        # Return cached object if present.
        component_classes = cls._cached_module_classes.get(component_type_name)

        if component_classes:
            component_class = component_classes.get(full_component_name)

            if component_class:
                return component_class

        # Otherwise check the module directory. -> will only have it by class name not component name - TODO this needs a fix
        is_component_exists = component_name in module.__dir__()

        # Skip if not present.
        if not is_component_exists:
            # print(module.__dir__())
            # print(component_name)
            print(f"  {cls._CROSS_CHAR} {component_type_name} {component_name} could not be found. Skipping.")
            return

        # Get object from module
        component_class = getattr(module, component_name)

        # Check object is the expected type
        if not issubclass(component_class, component_archetype):
            type_name = component_class.__name__
            print(f"  {cls._CROSS_CHAR} {component_type_name} {component_name} is of type {type_name} which "
                  f"is not a {component_type_name} object. Skipping.")
            return

        # Get the component dependencies
        dependencies = component_params.get("dependencies")

        if dependencies:
            dependency_list = cls._yaml_str_list_to_list(dependencies)
            dependency_set = set(dependency_list)
            missing_dependencies = dependency_set.difference(cls._available_dependencies)

            if missing_dependencies:
                dependency_delimited = ",".join(dependency_set)
                print(f"  {cls._CROSS_CHAR} {component_type_name} {component_name} is missing dependencies. "
                      f"Skipping")
                print(f"    (Missing dependencies: {dependency_delimited})")
                return

        return component_class

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
