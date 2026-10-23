from __future__ import annotations

import numpy as np
import pandas as pd
import traceback
import time

from warnings import warn
from tqdm.auto import tqdm
from typing import Any, Literal, TYPE_CHECKING, overload
from collections.abc import Collection

# From this package
from pyspoc._base import Component
from pyspoc.dataset import Dataset
from pyspoc.config import Config
from pyspoc.rstatistics.base import ReducedStatistic
from pyspoc.statistics.base import Statistic
from pyspoc.reducers.base import Reducer
from pyspoc._argchecking import RuntimeTypeCheckedMixin
from pyspoc.settings import settings

if TYPE_CHECKING:
    from warnings import WarningMessage


class Calculator(RuntimeTypeCheckedMixin):
    """
    The calculator takes in a dataset, computes and stores all selected statistical
    summaries based on a configuration.

    Configurations are provided using the Config class.

    Example:
         import numpy as np
         dataset = np.random.randn(5,500)    # create a random multivariate dataset
         calc = Calculator(dataset=dataset)  # Instantiate the calculator
         calc.compute()                      # Compute all statistical summaries

    Parameters
    ----------

    dataset : Dataset or numpy.ndarray or pandas.DataFrame or str
        A multivariate dataset typically with n realisations of p variables.
    name : str or None, optional
        The name of the calculator. Mainly used for printing the results but can be
        useful if you have multiple instances, defaults to None.
    labels : Iterable[str]
        Any set of strings by which you want to label the calculator. This can be
        useful later for classification purposes, defaults to None.
    normalise : bool or None, optional
        Normalise the dataset across realisations before computing statistical summaries,
        defaults to True.
    """

    _cached_results = dict()
    _max_calc_results = 5  # Change this to a global config setting

    def __init__(self,
                 dataset: Dataset | np.ndarray | pd.DataFrame | str,
                 name: str = None,
                 labels: Collection[str] | None = None,
                 normalise: bool = True):

        self._ss: dict[str, Statistic] = dict()
        self._excluded_ss: list[dict[str, Any]] = list()
        self._normalise: bool = normalise
        self._cached_configs = dict()
        self._dataset: Dataset
        self._current_sample_indices = (0,0)
        self._rng = np.random.default_rng(seed=0)

        self._results_dict = dict()
        self._results = None

        self._loaded_modules = dict()
        self._loaded_stat_config = dict()
        self._loaded_stats = dict()
        self._loaded_reducer_config = dict()
        self._loaded_reducers = dict()
        self._active_component_name : str | None = None
        self._raised_warnings : dict[str, tuple[WarningMessage, ...]] = dict()
        self._untracked_warnings: tuple[WarningMessage, ...] = tuple()
        self._raised_errors : dict[str, Exception] = dict()
        self._untracked_errors: tuple[Exception, ...] = tuple()
        self._names_dict: dict[str, dict[str, str]] = dict()
        self._times: dict[tuple[str, str], float] = dict()

        self._name = name
        self._labels = tuple(labels) if labels is not None else None
        self.set_dataset(dataset)

    @property
    def ss(self):
        """Dict of statistical summaries.

        Keys are the statistical summary identifier and values are their objects.
        """
        return self._ss

    @property
    def n_ss(self):
        """Number of statistical summaries in the calculator."""
        return len(self._ss)

    @property
    def dataset(self):
        """Dataset as a dataset object."""
        return self._dataset
        
    @property
    def name(self) -> str:
        """Name of the calculator."""
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = name

    @property
    def labels(self) -> tuple[str, ...] | None:
        """List of calculator labels."""
        return self._labels

    @labels.setter
    def labels(self, labels: Collection[str]):
        self._labels = tuple(labels)


    def get_errors(self) -> dict[str, Exception | tuple[Exception, ...]]:
        errors = dict()
        errors.update(self._raised_errors)
        errors["Untracked"] = self._untracked_errors
        return errors


    def get_warnings(self) -> dict[str, tuple[WarningMessage, ...]]:
        warnings = dict()
        warnings.update(self._raised_warnings)
        warnings["Untracked"] = self._untracked_warnings
        return warnings
    
    
    def get_error(
            self,
            component_name: str) -> (Exception | None):

        return self._raised_errors.get(component_name)

    def get_warning(
            self,
            component_name: str) -> (tuple[WarningMessage, ...] | None):

        return self._raised_warnings.get(component_name)

    @overload
    def compute(
            self,
            config: Config):
        self.compute(config)


    @overload
    def compute(
            self,
            config: Config,
            subsampling_type: Collection[Literal["random", "cyclical"]] |
                Literal["random", "cyclical"],
            subsamplng_size: tuple[int, int]):
        
        return self.compute(
            config=config,
            subsampling_type=subsampling_type,
            subsamplng_size=subsamplng_size
        )

    @overload
    def compute(
            self,
            config: Config,
            subsampling_type: Collection[Literal["random", "cyclical"]] |
                Literal["random", "cyclical"],
            subsamplng_size: tuple[int, int],
            n_samples: int):

        return self.compute(
            config=config,
            n_samples=n_samples,
            subsampling_type=subsampling_type,
            subsamplng_size=subsamplng_size
        )

    @overload
    def compute(
            self,
            config: Config,
            subsampling_type: Collection[Literal["random", "cyclical"]] |
                Literal["none", "random", "cyclical"] = "none",
            subsamplng_size: tuple[int, int] | None = None,
            n_samples: int | None = None,
            debug: bool = False
            ):
        ...

    # TODO: Restrict the Statistics calculated based on a union of Reducer statistic filters.
    # Stops from computing statistics that won't be used.
    def compute(
            self,
            config: Config,
            subsampling_type: Collection[Literal["random", "cyclical"]] |
                Literal["none", "random", "cyclical"] = "none",
            subsamplng_size: tuple[int, int] | None = None,
            n_samples: int | None = None,
            debug: bool = False
            ):
        
        """Compute the statistical summaries on the dataset."""

        if not hasattr(self, "_dataset"):
            raise AttributeError(
                "Dataset not loaded yet. Please provide dataset to the dataset property.")

        if subsampling_type != "none" and subsamplng_size is None:
            raise ValueError(
                "Subsampling size must be provided when subsampling type is not 'none'.")


        # Initialise for new computation
        self._results_dict.clear()
        self._names_dict.clear()
        stats: dict[str, Statistic] = config.statistics
        reducers: dict[str, Reducer] = config.reducers
        rstats: dict[str, ReducedStatistic] = config.reduced_statistics
        dataset = self.dataset
        elapsed = 0

        self._clear_errors()
        self._clear_warnings()
        dataset._prime_data()

        print("-" * 100)
        print(f"Running calculation [{self._name}]")
        print("-" * 100)

        # Calculate configured Statistics.
        if stats:
            stat_keys = list(stats.keys())
            n_stats = len(stat_keys)

            with tqdm(total=n_stats,
                      position=0,
                      desc="Computing Statistics") as stat_pbar:
                outer_pos = stat_pbar.pos

                with tqdm(
                    bar_format="{desc}",  # Placeholder only.
                    position=outer_pos+1) as label:
                    
                    for stat_name in stat_keys:
                        label.set_description_str(f"Statistic: [{stat_name}]")

                        try:
                            # Get the next statistic.
                            stat = stats[stat_name]

                            # Store the active statistic.
                            self._active_component_name = stat_name

                            # Register this calculator with the statistic for warning propagation.
                            stat._set_active_calculator(self)

                            start = time.perf_counter()

                            # Get result (checks cache first before computation).
                            stat.calculate(dataset, tqdm_pos=outer_pos+2)

                            elapsed = time.perf_counter() - start

                            self._times[stat_name] = elapsed

                            # Store name variations for results display
                            self._add_names(stat_name, stat)

                            # Add the statistic reference to the results dictionary.
                            self._results_dict[stat_name] = dict()

                        except Exception as e:
                            self._add_error(e)

                            if debug:
                                raise e
                            
                        finally:
                            stat_pbar.update()

                    elapsed += stat_pbar.format_dict["elapsed"]

            stat_names = list(self._results_dict.keys())
            reducer_stat_list = list()

            for reducer_name in reducers.keys():
                reducer = reducers[reducer_name]
                applicable_stat_names = config.get_reducer_filters(reducer)

                if not applicable_stat_names:
                    reducer_stat_names = stat_names
                else:
                    reducer_stat_names = [stat_name for stat_name in applicable_stat_names
                                          if stat_name in stat_names]

                for stat_name in reducer_stat_names:
                    reducer_stat_list.append((reducer_name, stat_name))


            n_stat_reducer_pairs = len(reducer_stat_list)

            with tqdm(total=n_stat_reducer_pairs,
                      position=0,
                      desc="Computing Reductions") as reducer_pbar:
                outer_pos = label.pos

                with tqdm(
                    bar_format="{desc}",  # Placeholder only.
                    position=outer_pos+1) as label:

                    for reducer_name, stat_name in reducer_stat_list:
                        combined_name = f"{stat_name} => {reducer_name}"
                        label.set_description_str(
                            f"Reduction: [{combined_name}]")
                        reducer = reducers[reducer_name]

                        # Register this calculator with the reducer for warning propagation.
                        reducer._set_active_calculator(self)

                        try:
                            # Get computed statistic.
                            stat = stats[stat_name]

                            # Store the active statistic-reducer combo.
                            self._active_component_name = combined_name

                            # If the Statistic is a ReducedStatistic (ie. an all-in-one) then
                            # store the result and continue.
                            if isinstance(stat, ReducedStatistic):
                                self._results_dict[stat_name][""] = stat._get_result()
                                continue

                            # Reduce the result.
                            R = reducer.calculate(stat)
                            
                            if isinstance(R, np.ndarray):
                                R = R.squeeze()

                            # Store name variations for results display
                            self._add_names(reducer_name, reducer)

                            # Save results.
                            self._results_dict[stat_name][reducer_name] = R

                        except Exception as e:
                            self._add_error(e)

                            if debug:
                                raise e

                        finally:
                            reducer_pbar.update()

                    elapsed += reducer_pbar.format_dict["elapsed"]

        # Calculate configured ReducedStatistics.
        if rstats:
            rstat_keys = list(rstats.keys())
            n_rstats = len(rstat_keys)

            with tqdm(total=n_rstats,
                      position=0,
                      desc="Computing Reduced Statistics") as rstat_pbar:
                outer_pos = label.pos

                with tqdm(
                    bar_format="{desc}",  # Placeholder only.
                    position=outer_pos+1) as label:

                    for rstat_name in rstat_keys:
                        try:
                            label.set_description_str(
                                f"Reduced Statistic [{rstat_name}]")

                            # Get the next reduced statistical summary.
                            rstat = rstats[rstat_name]

                            # Store the active reduced statistic.
                            self._active_component_name = rstat_name

                            # Register calculator with reduced statistic for warning propagation.
                            rstat._set_active_calculator(self)

                            start = time.perf_counter()

                            # Get result.
                            R = rstat.calculate(dataset)

                            if isinstance(R, np.ndarray):
                                R = R.squeeze()

                            elapsed = time.perf_counter() - start

                            self._times[rstat_name] = elapsed

                            # Save results.
                            self._results_dict[rstat_name] = dict()
                            self._results_dict[rstat_name][""] = R

                            # Store name variations for results display
                            self._add_names(rstat_name, rstat)

                        except Exception as e:
                            self._add_error(e)

                            if debug:
                                raise e

                        finally:
                            rstat_pbar.update()

                    elapsed += rstat_pbar.format_dict["elapsed"]

        print("-" * 100)
        print(f"\nCalculation complete. Time taken: {elapsed:.4f}s")
        print("-" * 100)
        self._active_component_name = None

        try:
            self._results = self.get_results()
        except Exception as e:
            self._add_error(e)

            if debug:
                raise e
        finally:
            self._report_errors()
            self._report_warnings()


    def get_timings(
            self,
            display_names: Literal["full", "class", "short"] = "full") -> pd.DataFrame:

        if display_names == "full":
            return pd.DataFrame.from_dict(
                data=[self._times]).rename({0:self._name})

        timings = dict()

        for stat_full_name, timing in self._times.items():
            stat_name = self._get_display_name(stat_full_name, display_names)
            timings[stat_name] = timing

        return pd.DataFrame.from_dict(
            data=[timings]).rename({0:self._name})
    

    def get_results(
            self,
            display_names: Literal["full", "class", "short"] = "full"
    ) -> pd.DataFrame | None:

        result_row = []
        first_level = []
        second_level = []

        for stat_full_name, reducers in self._results_dict.items():
            if not isinstance(reducers, dict):
                continue

            for reducer_full_name, reduced_result in reducers.items():
                if isinstance(reduced_result, np.ndarray):
                    size = reduced_result.size
                    reduced_result = reduced_result.reshape(-1)
                    result_row.extend(reduced_result)
                else:
                    result_row.append(reduced_result)
                
                stat_name = self._get_display_name(stat_full_name, display_names)
                reducer_name = self._get_display_name(reducer_full_name, display_names)
                first_level.extend([stat_name] * size)
                second_level_names = ([reducer_name] if size == 1
                    else [f"{i+1}" for i in range(size)] if reducer_name == ""
                    else [f"{reducer_name}_{i+1}" for i in range(size)]
                )
                second_level.extend(second_level_names)

        if not first_level or not second_level:
            warn("No results were generated. Check get_errors() or get_warnings() methods "
                "to check if errors occurred during processing.")
            return None
        
        columns = [*zip(first_level, second_level)]
        results_table = pd.DataFrame(
            data = {
                col: datum
                for col, datum in zip(columns, result_row, strict=True)
            },
            index=[self.name])
        results_table.columns.names = ["Statistic", "Reducer"]
        return results_table

    
    # NOTE: This is a hacky method to get the scheme
    # In the future, Config will be rewritten to allow for multiple
    # name types to be selected.
    def _add_names(self, component_full_name: str, component: Component):
        dot_idx = component_full_name.rfind(".")
        cls_name = type(component).__name__
        short_name = component.get_identifier()

        if dot_idx > -1:
            scheme = component_full_name[dot_idx:]
            cls_name += scheme
            short_name += scheme

        self._names_dict[component_full_name] = {
            "class": cls_name,
            "short": short_name
        }

    # TODO: Allow for computation to continue from the current position.
    def resume(self):
        ...

    def set_dataset(
            self,
            dataset: Dataset | np.ndarray | pd.DataFrame | str):
        """Load new dataset into existing instance.

        Args:
            dataset (pyspc.Data, np.ndarray, pd.DataFrame, str)
                New dataset to attach to calculator.
        """
        if isinstance(dataset, Dataset):
            self._dataset = dataset
            return

        self._dataset = Dataset(data=dataset, normalise=self._normalise)
        self._results = None
        self._results_dict = {}

        if settings.current.verbose:
            print(f"Dataset changed for Calculator {self._name}, "
                  "previously results have been cleared.")


    def _clear_errors(self):
        self._raised_errors = dict()
        self._untracked_errors = tuple()


    def _clear_warnings(self):
        self._raised_warnings = dict()
        self._untracked_warnings = tuple()


    def _add_error(self, error: Exception):
        if self._active_component_name is not None:
            self._raised_errors[self._active_component_name] = error
            return

        if self._untracked_warnings is None:
            self._untracked_errors = (error,)
            return
        
        self._untracked_errors = (*self._untracked_errors, error)


    def _add_warnings(self, warnings: Collection[WarningMessage]):
        if self._active_component_name is not None:
            self._raised_warnings[self._active_component_name] = warnings
            return

        if self._untracked_warnings is None:
            self._untracked_warnings = tuple(warnings)
            return
        
        self._untracked_warnings = (*self._untracked_warnings, *warnings)


    def _report_warnings(self):
        if self._raised_warnings or self._untracked_warnings:
            print("-" * 100)
            print("Warnings")
            print("-" * 100)
            print("The following warnings were raised during computation:")

        if self._raised_warnings:
            for component_name, raised_warnings in self._raised_warnings.items():
                for warning in raised_warnings:
                    self._print_warning(warning, component_name)

        if self._untracked_warnings:
            for warning in self._untracked_warnings:
                self._print_warning(warning, "Unattributed to any Component")


    @staticmethod
    def _print_warning(warning: WarningMessage, component_name: str):
        """Print a captured warning and its original source metadata.

        Parameters
        ----------
        warning : warnings.WarningMessage
            Captured warning record containing the warning instance and source
            location supplied by Python's warnings machinery.
        component_name : str
            Component attribution displayed above the warning details.

        Returns
        -------
        None
            The formatted warning is written to standard output immediately.
        """
        print(f"{component_name}:")
        print(f"[{warning.category.__name__}] {warning.message}")
        print(f"Location: {warning.filename}:{warning.lineno}")
        print()

    def _report_errors(self):
        if self._raised_errors or self._untracked_errors:
            print("-" * 100)
            print("Exceptions")
            print("-" * 100)
            print("The following exceptions were raised during computation:")

        if self._raised_errors:
            for component_name, error in self._raised_errors.items():
                self._print_error(error, component_name)

        if self._untracked_errors:
            for error in self._untracked_errors:
                self._print_error(error, "Unattributed to any Component")


    @staticmethod
    def _print_error(error: Exception, component_name: str):
        """Print a caught exception and its originating traceback location.

        Parameters
        ----------
        error : Exception
            Exception captured during Calculator execution.
        component_name : str
            Component attribution displayed above the exception details.

        Returns
        -------
        None
            The formatted exception is written to standard output immediately.

        Notes
        -----
        An exception constructed but never raised has no traceback. In that
        case its filename and line number cannot be recovered.
        """
        print(f"{component_name}:")
        print(f"[{type(error).__name__}] {error}")

        traceback_entries = traceback.extract_tb(error.__traceback__)

        if traceback_entries:
            origin = traceback_entries[-1]
            print(f"Location: {origin.filename}:{origin.lineno}")
        else:
            print("Location: unavailable")

        print()

    def _get_display_name(
            self,
            full_name: str,
            display_names: Literal["full", "class", "short"] = "full") -> str:

        if display_names == "full":
            return full_name

        names = self._names_dict.get(full_name)

        if names is None:
            return full_name

        name = names.get(display_names)

        if name is None:
            return full_name

        return name
