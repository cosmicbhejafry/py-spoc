import numpy as np

from typing import Literal, Any
from causallearn.search.FCMBased.lingam import (
    ICALiNGAM,
    DirectLiNGAM,
    RCD
)

import pyspoc._argchecking as argchk
from pyspoc._utils.lib import OPTIONAL_MODULES
from pyspoc.settings import settings
from pyspoc._tag import Tag
from pyspoc.component.statistics.graph.causal._base import CausalGraphStatistic



class LiNGAMCausalGraphBaseStatistic(CausalGraphStatistic):


    _tags = (Tag.GRAPH_WEIGHTED, Tag.GRAPH_DIRECTED, Tag.GRAPH_ACYCLIC, Tag.SIGNED)
    
    def __init__(
            self,
            standardise: bool,
            dim: Literal["n", "p"] = "p",
            nan_policy: Literal[
                "raise", "row-omit", "col-omit"] = "raise"):

        super().__init__(
            standardise=standardise,
            dim=dim,
            nan_policy=nan_policy)

        if standardise:
            self._tags = self._get_modified_tags(
                self._tags,
                remove_tags=(Tag.GRAPH_WEIGHTED, Tag.SIGNED))


    @classmethod
    def _standardise_adjacency(
            cls,
            graph: np.ndarray[tuple[int, int], np.dtype[Any]]
    ) -> np.ndarray[tuple[int, int], np.dtype[Any]]:
        
        # Coefficients do not indicate direction, hence non-zero indicates cause.
        # Transpose to shift from causal column-wise directions to row-wise.
        graph = graph.T
        non_zero_non_nan_mask = (graph != 0) & (~np.isnan(graph))
        graph[non_zero_non_nan_mask] = 1
        return graph


    def _compute_adjacency_matrix(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]],
            model_type: type[ICALiNGAM | DirectLiNGAM | RCD],
            **model_kwargs: Any
    ) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
        
        model = model_type(**model_kwargs)
        model.fit(data)
        adjacency_mat = model.adjacency_matrix_

        if adjacency_mat is None:
            raise ValueError("No graph found after LiNGAM fit.")

        return adjacency_mat



class ICALiNGAMCausalGraph(LiNGAMCausalGraphBaseStatistic):


    def __init__(
            self,
            standardise: bool = False,
            max_iter: int = 1000,
            random_seed: int | None = None,
            *,
            dim: Literal["n", "p"] = "p",
            nan_policy: Literal[
                "raise", "row-omit", "col-omit"] = settings.current.data_nan_policy):
        
        self._max_iter = argchk.check_natural_number(max_iter, "max_iter")
        super().__init__(
            standardise=standardise,
            dim=dim,
            nan_policy=nan_policy)
        

    def _compute(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]]
    ) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:

        return self._compute_adjacency_matrix(
            data,
            model_type=ICALiNGAM,
            random_seed=self.random_seed,
            max_iter=self._max_iter)



class DirectLiNGAMCausalGraph(LiNGAMCausalGraphBaseStatistic):

    def __init__(
            self,
            standardise: bool = False,
            random_seed: int | None = None,
            measure: Literal["pwling", "kernel"] = "pwling",
            *,
            dim: Literal["n", "p"] = "p",
            nan_policy: Literal[
                "raise", "row-omit", "col-omit"] = settings.current.data_nan_policy):

        self._measure = measure
        super().__init__(
            standardise=standardise,
            dim=dim,
            nan_policy=nan_policy)

    def _compute(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]]
    ) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:

        return self._compute_adjacency_matrix(
            data,
            model_type=ICALiNGAM,
            random_seed=self.random_seed,
            measure=self._measure)
    


class RCDLiNGAMCausalGraph(LiNGAMCausalGraphBaseStatistic):

    def __init__(
            self,
            standardise: bool = False,
            max_explanatory_num: int = 2,
            cor_alpha: float = 0.01,
            ind_alpha: float = 0.01,
            shapiro_alpha: float = 0.01,
            MLHSICR: bool = False,
            bw_method: Literal["mdbs", "scott", "silverman"] = "mdbs",
            *,
            dim: Literal["n", "p"] = "p",
            nan_policy: Literal[
                "raise", "row-omit", "col-omit"] = settings.current.data_nan_policy):

        self._max_explanatory_num = argchk.check_natural_number(
            max_explanatory_num, arg_name="max_explanatory_num")
        self._cor_alpha = argchk.check_float(
            cor_alpha, exclusive_minimum=0, maximum=0.1, arg_name="cor_alpha")
        self._ind_alpha = argchk.check_float(
            ind_alpha, exclusive_minimum=0, maximum=0.1, arg_name="ind_alpha")
        self._shapiro_alpha = argchk.check_float(
            shapiro_alpha, exclusive_minimum=0, maximum=0.1, arg_name="shapiro_alpha")
        self._MHLSICR = MLHSICR
        self._bw_method = bw_method

        super().__init__(
            standardise=standardise,
            dim=dim,
            nan_policy=nan_policy)
    
    def _compute(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]]
    ) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:

        return self._compute_adjacency_matrix(
            data,
            model_type=RCD,
            random_seed=self.random_seed,
            max_explanatory_num=self._max_explanatory_num,
            cor_alpha=self._cor_alpha,
            ind_alpha=self._ind_alpha,
            shapiro_alpha=self._shapiro_alpha,
            MHLSICR=self._MHLSICR,
            bw_method=self._bw_method)


    @classmethod
    def _standardise_adjacency(
            cls,
            graph: np.ndarray[tuple[int, int], np.dtype[Any]]
    ) -> np.ndarray[tuple[int, int], np.dtype[Any]]:
        
        graph = super()._standardise_adjacency(graph)

        # Latent confounders encoded as NaNs
        graph[np.isnan(graph)] = 2
        return graph


class CAMUVLiNGAMCausalGraph(CausalGraphStatistic):

    # CAM-UV builds an ADMG.
    _tags = (Tag.GRAPH_MIXED, Tag.GRAPH_ACYCLIC, Tag.SIGNED)
    
    def __init__(
            self,
            standardise: bool = True,
            alpha: float = 0.01,
            max_explanatory_num: int = 2,
            *,
            dim: Literal["n", "p"] = "p",
            nan_policy: Literal[
                "raise", "row-omit", "col-omit"] = settings.current.data_nan_policy):

        self._max_explanatory_num = argchk.check_natural_number(
            max_explanatory_num, arg_name="max_explanatory_num")
        self._alpha = argchk.check_float(
            alpha, exclusive_minimum=0, maximum=0.1, arg_name="alpha")
        self._U_: tuple[set[int], ...] | None = None
        self._P_: tuple[set[int], ...] | None = None

        super().__init__(
            standardise=standardise,
            dim=dim,
            nan_policy=nan_policy)


    def get_P(self) -> tuple[set[int], ...] | None:
        if self._P_ is None:
            print(
                self.get_name() + " must be fit to data first - call compute()")

        return self._P_

    def get_U(self) -> tuple[set[int], ...] | None:
        if self._U_ is None:
            print(
                self.get_name() + " must be fit to data first - call compute()")
            return

        return self._U_
    
    def _compute(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]]
    ) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:

        try:
            from causallearn.search.FCMBased.lingam.CAMUV import execute
        except ModuleNotFoundError as e:
            if e.name in OPTIONAL_MODULES:
                raise ModuleNotFoundError(
                    f"Extended module {e.name} is required for {self.get_name()}. "
                    "Install with 'pip install \"pyspoc[extended]\"'")

            raise e

        P, U = execute(
            data,
            num_explanatory_vals=self._max_explanatory_num,
            alpha=self._alpha)
        
        self._P_ = tuple(P)
        self._U_ = tuple(U)
        return self._make_graph(P, U)
    

    def _make_graph(
            self,
            P: tuple[set[int], ...],
            U: tuple[set[int], ...]
    ) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:

        p = len(P)
        adjacency = np.zeros(shape=(p,p))
        
        for i, parents in enumerate(P):
            if not parents:
                continue
            
            for j in parents:
                if self._standardise:
                    # Add directed edge in DAG convention.
                    adjacency[j,i] = 1
                else:
                    # Add directed edge in causal convention.
                    adjacency[j,i] = -1
                    adjacency[i,j] = 1

        # Add undirected edges for UCPs and UBPs.
        for pair in U:
            i, j = pair
            adjacency[i,j] = -1
            adjacency[j,i] = -1

        return adjacency

    @classmethod
    def _standardise_adjacency(cls, graph: np.ndarray[tuple[int, int], np.dtype[np.number]]
    ) -> np.ndarray[tuple[int, int], np.dtype[np.number]]:
        # None needed, already standardised by construction in _make_graph()
        return graph
