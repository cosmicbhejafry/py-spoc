"""Causal discovery based graph Statistics.

Back-end algorithm implementations provided by Zheng, Yujia, et al.
"Causal-learn: Causal discovery in python." Journal of Machine Learning Research 25.60 (2024): 1-8.
"""
from __future__ import annotations

import numpy as np

from typing import Literal
from enum import Enum
from abc import ABC, abstractmethod

import pyspoc._argchecking as argchk

from ._base import CausalGraphStatistic
from pyspoc.settings import settings


class UCRule(Enum):
    uc_sepset = 0
    maxP = 1
    definiteMaxP = 2



class UCPriority(Enum):
    uc_rule_default = -1
    overwrite = 0
    bidirected = 1
    existing = 2
    stronger = 3



class CBCDStatistic(CausalGraphStatistic, ABC):
    """
    Base class for constraint-based causal discovery (CBCD) graph statistics performing
    common argument checks.

    The :meth:`_standardise_graph` method must align all causal graph statistics
    to the following conventions:
    
    For an element `X[i,j]` in the graph adjacency matrix, the following DAG-compatible encodings
    should be applied:
    
    - No edge (X_i  X_j) => `X[i,j] = X[j,i] = 0`
    - Unidirectional (X_i -> X_j) => `X[i,j] = 1` and `X[j,i] = 0` (DAG convention)
    - Undirected (X_i -- X_j) => `X[i,j] = X[j,i] = -1`
    - Bidirectional (X_i <-> X_j) => `X[i,j] = X[j,i] = 1`
    - Latent Confounding (X_i <-> X_j) => `X[i,j] = X[j,i] = 2`
    - Conflicted Direction (X_i <-> X_j) => `X[i,j] = X[j,i] = -2`
    """

    _long_name = "Constraint-Based Causal Discovery Graph"
    _identifier = "graph-cbcd"


    def __init__(
            self,
            standardise: bool,
            independence_test: Literal[ # Conditional independence tests.
                "fisherz", # Fishers' Z
                "chisq", # Chi-squared
                "gsq", # G-squared
                "kci", # Kernel-based
                "mv_fisherz" # Missing-value Fisher's Z
            ],
            alpha: float,
            dim: Literal["n", "p"],
            nan_policy: Literal[
                "raise", "row-omit", "col-omit"]):

        self._independence_test = independence_test
        self._alpha = argchk.check_float(alpha, minimum=0, maximum=1, arg_name="alpha")
        super().__init__(
            standardise=standardise,
            dim=dim,
            nan_policy=nan_policy)

    def _compute(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]]
    ) -> np.ndarray[tuple[int, int], np.dtype[np.integer]]:
        verbose = self._compute_kwargs.get("verbose", settings.current.verbose)
        show_progress = self._compute_kwargs.get("show_progress", False)

        graph = self._compute_adjacency(
            data,
            verbose=verbose,
            show_progress=show_progress)

        return graph
       

    @abstractmethod
    def _compute_adjacency(
        self,
        data: np.ndarray[tuple[int, int], np.dtype[np.floating]],
        verbose: bool,
        show_progress: bool
    ) -> np.ndarray[tuple[int, int], np.dtype[np.integer]]:
        pass



class PCCausalGraph(CBCDStatistic):
    """
    Perform Peter-Clark algorithm for causal discovery.
    """
    _long_name = "Peter-Clark Constraint-Based Causal Discovery Graph"
    _identifier = "graph-cbcd-pc"


    def __init__(
            self,
            standardise: bool = True,
            alpha: float = 0.05,
            independence_test: Literal[ # Conditional independence tests.
                "fisherz", # Fishers' Z
                "chisq", # Chi-squared
                "gsq", # G-squared
                "kci", # Kernel-based
                "mv_fisherz" # Missing-value Fisher's Z
            ] = "fisherz",
            stable: bool = True,
            uc_rule: Literal["uc_sepset", "maxP", "definiteMaxP"] = "uc_sepset",
            uc_priority: Literal[
                "uc_rule_default","overwrite","bidirected","existing","stronger"] = "existing",
            mvpc: bool = False,
            correction_name: str = "MV_Crtn_Fisher_Z",
            dim: Literal["n", "p"] = "p",
            nan_policy: Literal[
                "raise", "row-omit", "col-omit"] = settings.current.data_nan_policy):

        super().__init__(
            standardise=standardise,
            independence_test=independence_test,
            alpha=alpha,
            dim=dim,
            nan_policy=nan_policy)
        
        self._stable = stable
        self._uc_rule = UCRule[uc_rule].value
        self._uc_priority = UCPriority[uc_priority].value
        self._mvpc = mvpc
        self._correction_name = correction_name


    def _compute_adjacency(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]],
            verbose: bool,
            show_progress: bool
        ) -> np.ndarray[tuple[int, int], np.dtype[np.integer]]:

        from causallearn.search.ConstraintBased.PC import pc
        cgraph = pc(
            data=data,
            alpha=self._alpha,
            indep_test=self._independence_test,
            stable=self._stable,
            uc_rule=self._uc_rule,
            uc_priority=self._uc_priority,
            mvpc=self._mvpc,
            correction_name=self._correction_name,
            verbose=verbose,
            show_progress=show_progress)
        
        return cgraph.G.graph.astype(np.integer)


    @classmethod
    def _standardise_adjacency(
            cls,
            adjacency: np.ndarray[tuple[int, int], np.dtype[np.integer]]
    ) -> np.ndarray[tuple[int, int], np.dtype[np.integer]]:

        p = adjacency.shape[0]
        
        for i in range(p):
            for j in range(i+1, p):

                # Switch X_i -> X_j directed edge encoding to DAG convention.
                if adjacency[i,j] == -1 and adjacency[j,i] == 1:
                    adjacency[j,i] = 0
                    adjacency[i,j] = 1
                    continue

                if adjacency[i,j] == 1:

                    # Switch X_j -> X_i directed edge encoding to DAG convention.
                    if adjacency[j,i] == -1:
                        adjacency[i,j] = 0
                        adjacency[j,i] = 1
                        continue
                
                    # PC <-> assigned to conflicted direction.
                    if adjacency[j,i] == 1:
                        adjacency[i,j] = -2
                        adjacency[j,i] = -2

        return adjacency


class FCICausalGraph(CBCDStatistic):
    """
    Perform Fast Causal Inference algorithm for causal discovery.
    """
    _long_name = "Fast Causal Inference Constraint-Based Causal Discovery Graph"
    _identifier = "graph-cbcd-fci"


    def __init__(
            self,
            standardise: bool = True,
            independence_test: Literal[ # Conditional independence tests.
                "fisherz", # Fishers' Z
                "chisq", # Chi-squared
                "gsq", # G-squared
                "kci", # Kernel-based
                "mv_fisherz" # Missing-value Fisher's Z
            ] = "fisherz",
            alpha: float = 0.05,
            depth: int = -1,
            max_path_length: int = -1,
            dim: Literal["n", "p"] = "p",
            nan_policy: Literal[
                "raise", "row-omit", "col-omit"] = settings.current.data_nan_policy):

        super().__init__(
            standardise=standardise,
            independence_test=independence_test,
            alpha=alpha,
            dim=dim,
            nan_policy=nan_policy)
        self._depth = depth
        self._max_path_length = max_path_length


    def _compute_adjacency(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]],
            verbose: bool,
            show_progress: bool
        ) -> np.ndarray[tuple[int, int], np.dtype[np.integer]]:

        from causallearn.search.ConstraintBased.FCI import fci

        cgraph_tuple = fci(dataset=data,
            independence_test_method=self._independence_test,
            alpha=self._alpha,
            depth=self._depth,
            max_path_length=self._max_path_length,
            verbose=verbose,
            show_progress=show_progress)

        cgraph = cgraph_tuple[0]
        cgraph_mat: np.ndarray = cgraph.graph # pyright: ignore[reportAttributeAccessIssue]
        return cgraph_mat.astype(np.integer)


    @classmethod
    def _standardise_adjacency(
            cls,
            adjacency: np.ndarray[tuple[int, int], np.dtype[np.integer]]
    ) -> np.ndarray[tuple[int, int], np.dtype[np.integer]]:

        p = adjacency.shape[0]
        
        for i in range(p):
            for j in range(i+1, p):
                
                # FCI no d-separation or no ancestry
                # means no direct edge.
                if adjacency[i,j] == 2:
                    adjacency[i,j] = 0
                    adjacency[j,i] = 0
                    continue

                # Switch X_i -> X_j directed edge encoding to DAG convention.
                if adjacency[i,j] == -1:
                    adjacency[j,i] = 0
                    adjacency[i,j] = 1
                    continue
                
                # Switch X_j -> X_i directed edge encoding to DAG convention.
                if adjacency[j,i] == -1:
                    adjacency[i,j] = 0
                    adjacency[j,i] = 1
                    continue
                
                # Only other changes involve a 1-encoding.
                if adjacency[i,j] != 1:
                    continue

                # FCI <-> assigned to latent confounding.
                if adjacency[j,i] == 1:
                    adjacency[i,j] = 2
                    adjacency[j,i] = 2
                    continue

                # FCI no ancestry means no direct edge.
                if adjacency[j,i] == 2:
                    adjacency[i,j] = 0
                    adjacency[j,i] = 0
                    continue

        return adjacency
