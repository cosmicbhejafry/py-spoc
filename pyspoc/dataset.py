"""Provide dataset structures for multivariate analysis.

Code is adapted from Patricia Wollstadt's IDTxL (https://github.com/pwollstadt/IDTxl)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import os
import math

# from scipy.stats import zscore
# from scipy.signal import detrend
from collections.abc import Collection, Iterator
from sklearn.preprocessing import StandardScaler
from typing import List, Literal, cast
from time import time
from typeguard import check_type

from pyspoc import _base
from pyspoc import _argchecking
from pyspoc.settings import settings


class Dataset:

    """
    Store dataset for dependency analysis.

    Dataset takes a 2-dimensional array representing realisations of random variables.
    Indicate the arrangement of realisations (n) and variables (p) in a two-character string
    e.g. 'np' for an array with rows representing realisations and columns representing variables.

    Example
    -----------

    .. code-block:: python
        # Initialise empty dataset object
        dataset = Dataset()
    
        # Load a prefilled financial dataset
        data_forex = Dataset.load_dataset("forex")
    
        # Create dataset objects with dataset of various sizes
        d = np.arange(3000).reshape((3, 1000)) # 3 procs
        data_2 = Dataset(d, dim_order='ps') # 1000 observations

    Parameters
    ------------

    data : numpy.ndarray or pandas.DataFrame or str
        2-dimensional array with raw dataset.
    dim_order : str
        Order of dimensions, accepts two combinations of the characters 'n' and 'p', defaults to 'np'.
    normalise : bool or None, optional
        If True, dataset is z-scored (normalised) along the realisations dimension, defaults to True.
    name : str or None, optional
        Name of the dataset
    var_names : Collection[str] or None, optional
        List of variable names with length the number of variables, defaults to None.
    n_realisations_subsample : int or None, optional
        Truncates dataset to this many realisations, defaults to None.
    n_variables_subsample : int or None, optional
        Truncates dataset to this many variables, defaults to None.
    """

    def __init__(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.float64]] | pd.DataFrame | str,
            dim_order: str = "np",
            normalise: bool = True,
            name: str | None = None,
            var_names: Collection[str] | None = None):

        self.normalise = normalise
        self.name = name
        self._dim_order: str
        self._n: int
        self._p: int
        self._p_subsample: int | None = None
        self._data_type: type
        self._base_data_copy: np.ndarray[tuple[int, int], np.dtype[np.float64]]
        self._active_data_copy: np.ndarray[tuple[int, int], np.dtype[np.float64]] | None = None

        self._var_names: List[str] = list()
        self._set_data(data=data,
                       dim_order=dim_order,
                       var_names=var_names)
        
        self._instantiation_time = time()

    @property
    def name(self) -> str:
        """Name of the Data object."""
        return self._name

    @name.setter
    def name(self, name: str | None):
        """Set the name of the Data object."""
        if name is None:
            name = ""

        check_type(name, str)
        self._name = name

    @property
    def data_type(self) -> type:
        return self._data_type

    @property
    def dim_order(self) -> str:
        return self._dim_order

    @dim_order.setter
    def dim_order(self, dim_order: str):
        check_type(dim_order, str)

        if len(dim_order) > 2:
            raise RuntimeError("dim_order can not have more than two entries")

        self._dim_order = dim_order

    @property
    def normalise(self) -> bool:
        return self._normalise

    @normalise.setter
    def normalise(self, normalise: bool):
        check_type(normalise, bool)
        self._normalise = normalise

    @property
    def n_realisations(self) -> int:
        return self._n

    @property
    def n_variables(self) -> int:
        return self._p


    @property
    def n_variables_subsample(self) -> int:
        if not self._p_subsample:
            return self._p

        return self._p_subsample

    @n_variables_subsample.setter
    def n_variables_subsample(self, n_variables_subsample: int | None):
        if n_variables_subsample is not None:
            check_type(n_variables_subsample, int)
            _argchecking.check_natural_number(n_variables_subsample)

        self._p_subsample = n_variables_subsample

    @property
    def var_names(self) -> List[str]:
        """List of variable names."""
        return self._var_names

    @property
    def var_names_subsample(self) -> List[str]:
        return self._var_names[:self._p_subsample]
    

    def get_shape(self) -> tuple[int, int]:
        # Cast allowed as runtime checks ensure data copy is 2-dimensional.
        return cast(tuple[int, int], self.get_data().shape)


    # Returns a read-only copy of data with requested transformations
    def get_data(self) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        if self._active_data_copy is not None:
            return self._active_data_copy

        return self._prime_data()
    

    def _prime_data(self):
        active_data = np.array(self._base_data_copy, copy=True)
    
        if self.normalise:
            active_data = self._normalise_data(active_data)
    
        active_data.flags.writeable = False
        self._active_data_copy = active_data
        return active_data


    def reset_sampling_cycle(self):
        self._current_sample_indices = (0,0)


    def get_samples(
            self,
            *,
            sample_shape: tuple[int, int],
            sampling_type: Collection[Literal["random", "cyclical"]] |
                Literal["random", "cyclical"],
            seed: int = 0,
            n_samples: int | None = None
    ) -> list[np.ndarray]:

        self._rng = np.random.default_rng(seed)
        data = self.get_data()

        if isinstance(sampling_type, str):
            sampling_type = [sampling_type]

        use_random_sampling = "random" in sampling_type
        use_cyclical_sampling = "cyclical" in sampling_type

        if n_samples is None:
            n_samples = self._get_computed_n_samples(
                sample_shape=sample_shape)

        mask_generator = self._get_sampling_masks(
            n_samples=n_samples,
            sample_shape=sample_shape,
            use_cyclical_sampling=use_cyclical_sampling,
            use_random_sampling=use_random_sampling)
        samples = list()

        for mask in mask_generator:
            sampled_data = data[mask]
            samples.append(sampled_data)

        return samples

    def _get_computed_n_samples(
            self,
            sample_shape: tuple[int, int]) -> int:
        
        n, p = self.get_shape()
        sample_n, sample_p = sample_shape
        n_coverage = math.ceil(n / sample_n)
        p_coverage = math.ceil(p / sample_p)
        n_samples = n_coverage * p_coverage
        return n_samples

    def _get_sampling_masks(
            self,
            n_samples: int,
            sample_shape: tuple[int, int],
            use_cyclical_sampling: bool,
            use_random_sampling: bool
    ) -> Iterator[tuple[np.ndarray, ...]]:

        if use_cyclical_sampling:
            generator = self._get_cyclical_indices(
                n_samples=n_samples,
                sample_shape=sample_shape,
                use_random_sampling=use_random_sampling)
        else:
            generator = self._get_random_indices(
                n_samples=n_samples,
                sample_shape=sample_shape
            )

        for n_indices, p_indices in generator:
            mask = np.ix_(n_indices, p_indices)
            yield mask


    def _get_random_indices(
            self,
            n_samples: int,
            sample_shape: tuple[int, int]
    ) -> Iterator[tuple[tuple[int, ...], tuple[int, ...]]]:

        data_shape = self.get_shape()
        n, p = data_shape
        sample_n, sample_p = sample_shape

        for _ in range(n_samples):
            n_indices = tuple(self._rng.integers(low=0, high=n-1, size=sample_n))
            p_indices = tuple(self._rng.integers(low=0, high=p-1, size=sample_p))
            yield n_indices, p_indices
    
    
    def _get_cyclical_indices(
            self,
            n_samples: int,
            sample_shape: tuple[int, int],
            use_random_sampling: bool
    ) -> Iterator[tuple[tuple[int, ...], tuple[int, ...]]]:

        current_indices = (0,0)
        data_shape = self.get_shape()
        n_sampled = 0
        break_loop = False
        
        while not break_loop:

            # Outer sampling loop along the n-axis
            axis = 0
            looped = False
            current_idx = current_indices[axis]
            sample_size = sample_shape[axis]
            data_size = data_shape[axis]
            n_indices, next_idx, _ = self._get_cyclical_axis_indices(
                current_idx=current_idx,
                sample_size=sample_size,
                data_size=data_size,
                use_random_sampling=use_random_sampling
            )
            current_indices = (next_idx, current_indices[1])

            while not looped:
                # Inner sampling loop along the p-axis
                axis = 1
                current_idx = current_indices[axis]
                sample_size = sample_shape[axis]
                data_size = data_shape[axis]
                p_indices, next_idx, looped = self._get_cyclical_axis_indices(
                    current_idx=current_idx,
                    sample_size=sample_size,
                    data_size=data_size,
                    use_random_sampling=use_random_sampling
                )

                # Update p position
                current_indices = (current_indices[0], next_idx)

                # Add to the sampling tally.
                n_sampled += 1

                # Yield the sampled indices.
                yield n_indices, p_indices

                # Break loop if we've sampled all requested.
                if n_sampled >= n_samples:
                    break_loop = True
                    break


    def _get_cyclical_axis_indices(
            self,
            current_idx: int,
            sample_size: int,
            data_size: int,
            use_random_sampling: bool):
        
        unconstrained_idx = current_idx + sample_size - 1
        next_idx = unconstrained_idx % data_size
        cycles = unconstrained_idx // data_size
        sample_indices = list()

        if cycles == 0:
            choices = list(range(current_idx, next_idx + 1))
            sample_indices.append(choices)
                
        else:
            choices = list(range(current_idx, data_size))
            sample_indices.append(choices)

            full_cycles = cycles - 1

            for _ in range(full_cycles):
                sample_indices.append(list(range(0, data_size)))

            choices = list(range(0, next_idx + 1))
            sample_indices.append(choices)

        indices = ()

        for sample in sample_indices:
            if use_random_sampling:
                sample = self._rng.choice(sample, size=len(sample), replace=False)

            indices = (*indices, *sample)

        # Store if the sampling looped through the axis
        looped = cycles > 0

        return indices, next_idx, looped


    # TODO: Consolidate with similar methods throughout the library
    # into the _utils module.
    @staticmethod
    def _convert_to_numpy(
        data: np.ndarray[tuple[int, int], np.dtype[np.float64]] | pd.DataFrame | str
    ) -> np.ndarray:
        """Converts other dataset instances to default numpy format."""

        if isinstance(data, np.ndarray):
            return data
        elif isinstance(data, pd.DataFrame):
            return data.to_numpy()
        elif isinstance(data, str):
            return Dataset._load_data(data)
        else:
            raise TypeError(f"Unknown data type: {type(data)}")
        

    @staticmethod
    def _load_data(path: str) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        ext = os.path.splitext(path)[1]

        if ext == ".npy":
            return np.load(path)
        elif ext == ".txt":
            return np.genfromtxt(path)
        elif ext == ".csv":
            return np.genfromtxt(path, ",")
        else:
            raise TypeError(f"Unknown filename extension: {ext}")
        

    def _set_data(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.float64]] | pd.DataFrame | str,
            dim_order: Literal["np", "pn"] = "np",
            var_names: Collection[str] | None = None
    ):

        """Overwrite dataset in an existing instance.

        Args:
            data (ndarray):
                2-dimensional array of realisations

            dim_order (str, Default "np"):
                Two character string indicating the order of dimensions (n) and variables (p).

            n_realisations_subsample (str, Optional):
                Indicates the number of realisations to include in the final dataset. With no value
                specified, all observations will be used.

            n_variables_subsample (str, Optional):
                Indicates the number of variables to include in the final dataset. With no value
                specified, all variables will be used.

            var_names (Collection[str], Optional):
                Provides a set of names for the dataset variables. Must be the same length as the number
                of variables (p).
        """

        # Set initial data transformation and subsampling properties
        self.dim_order = dim_order

        if not self.name:
            self.name = _base.retrieve_arg_name(data, max_steps=3)

        name = self.name

        # Preprocess copy of original data
        data_copy = self._get_formatted_data_copy(data)

        # Store it.
        self._base_data_copy = data_copy

        # Set data properties.
        self._data_type = type(data_copy[0, 0])
        self._set_data_dim(data_copy)

        # Store variables.
        if var_names is not None:
            check_type(var_names, Collection[str])
            var_names_list = list(set(var_names))
            self._var_names = var_names_list
        else:
            # if isinstance(self._base_data, pd.DataFrame):
            #     self._var_names = self._base_data.columns
            # else:
            self._var_names = [f"var-{i}" for i in range(self.n_variables)]

        # Report success if verbosity enabled.

        self._message(
            f'Dataset "{name}" now has properties: {self.n_realisations} realisations, '
            f'{self.n_variables} variables.')
        

    def _get_formatted_data_copy(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.float64]] | pd.DataFrame | str
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:

        # Preprocess copy of original data and store it.
        if isinstance(data, np.ndarray):
            data_copy = np.array(data, copy=True)
        elif isinstance(data, pd.DataFrame):
            data_copy = data.copy(True)
        elif isinstance(data, str):
            data_copy = self._load_data(data)
        else:
            raise TypeError("data must be a numpy.ndarray, pandas.DataFrame, or str")

        if len(self.dim_order) != data_copy.ndim:
            raise RuntimeError(
                "Data array dimension ({0}) and length of "
                "dim_order ({1}) are not equal.".format(data_copy.ndim, len(self.dim_order))
            )

        data_copy = self._convert_to_numpy(data_copy)
        data_copy = self._reorder_data(data_copy, self.dim_order)
        nans = np.isnan(data_copy)

        if nans.any():
            raise ValueError(
                f"Dataset {self.name} contains non-numerics (NaNs) in variables: "
                f"{np.unique(np.where(nans)[0])}."
            )

        return data_copy
    

    @staticmethod
    def _normalise_data(data: np.ndarray) -> np.ndarray:
        # TODO: FIX / CHOOSE
        # print("Normalising the dataset using zscores \n")
        # data = zscore(data, axis=0, nan_policy="omit", ddof=1)
        # try:
        #     data = detrend(data, axis=0)
        #     return data
        # except ValueError as err:
        #     print(f"Could not detrend dataset: {err}")

        # print("Normalising the dataset using RobustScaler \n")
        # try:
        #     data = RobustScaler().fit_transform(data)
        #     return data
        # except ValueError as err:
        #     print(f"Error with RobustScaling: {err}")

        print("Normalising the dataset using StandardScaler \n")
        try:
            std_data = StandardScaler().fit_transform(data)
            return std_data
        except ValueError as err:
            print(f"Error with RobustScaling: {err}")
            return data

    @staticmethod
    def _message(message: str):
        if settings.current.verbose:
            print(message)


    @staticmethod
    def _reorder_data(
        data: np.ndarray,
        dim_order: str):

        """Reorder dataset dimensions n realisations in p variables."""

        # reorder array dims if necessary
        if dim_order[0] != "n":
            return data.swapaxes(0, 1)

        return data

    def _set_data_dim(
            self,
            data: np.ndarray):

        """Set the dataset size."""
        self._n = data.shape[0]
        self._p = data.shape[1]


    def uncache(self, include_gc: bool = False):
        # Import locally so Dataset does not create a module-level dependency
        # cycle with the Statistic class that accepts Dataset instances.
        # TODO: Cascade the uncache() to the Reducers that were computed on the
        # affected statistics also.
        from pyspoc.core.statistic._base import Statistic

        Statistic.clear_cache(self, include_gc)


    @staticmethod
    def load_data(name: str):
        # TODO: MODIFY TO PARSE MY SYNTHETIC DATA NPY'S?
        basedir = os.path.join(os.path.dirname(__file__), "dataset")
        if name == "forex":
            filename = "forex.npy"
            dim_order = "pn"
        elif name == "cml":
            filename = "cml.npy"
            dim_order = "pn"
        else:
            raise NameError(f"Unknown dataset: {name}.")

        path = os.path.join(basedir, filename)
        dataset = np.load(path)
        return Dataset(data=dataset, dim_order=dim_order)
    

    def __getitem__(self, item: int | tuple[int, ...]):
        return self.get_data()[item]
