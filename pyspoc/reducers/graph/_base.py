from typing import Literal

import numpy as np

from enum import Enum
from abc import abstractmethod

from pyspoc.reducers.base import Reducer
from pyspoc.statistics.base import Statistic
from pyspoc.types import SummaryType
from pyspoc._tag import Tag, Signature


class GraphType(Enum):
    UNWEIGHTED = "unweighted"
    WEIGHTED = "weighted"
    CAUSAL_UNWEIGHTED = "causal_unweighted"
    CAUSAL_WEIGHTED = "causal_weighted"


class GraphReducer(Reducer):

    
    def __init__(
            self,
            internal_parallelism: Literal["serial","parallel"] = "serial",
            thread_safety: Literal["safe","unsafe"] = "unsafe"):
        
        super().__init__(
            internal_parallelism=internal_parallelism,
            thread_safety=thread_safety)

        self._stat_graph_type = None
        self._stat_tags = None
        self._is_stat_causal = None
    
    @classmethod
    def get_statistic_tags_filter(cls) -> tuple[Tag, ...]:
        """Graph tags that the GraphReducer is restricted to.
        """
        return tuple(Tag(tag) for name, tag
            in Tag._member_map_.items()
            if "graph" in name.lower())

    
    @classmethod
    def _supported_graph_types(cls):
        return (
            GraphType.UNWEIGHTED,
            GraphType.WEIGHTED,
            GraphType.CAUSAL_UNWEIGHTED,
            GraphType.CAUSAL_WEIGHTED)


    def _has_causal_signature(
            self,
            stat: Statistic) -> bool:
        return Signature.CAUSAL in stat.get_signatures()


    def _reformat_unweighted_causal(
            self,
            adjacency: np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]]
        ) -> np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]]:
                
        # Remove latent confounding and set conflicted direction to undirected.
        adjacency = np.where(adjacency == 2, 0, adjacency)
        return np.where(adjacency == -2, -1, adjacency)


    def _get_adjacency_type(
            self,
            adjacency: np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]],
            is_causal: bool
    ) -> GraphType | None:

        if not np.isfinite(adjacency).all():
            return None

        if np.issubdtype(adjacency.dtype, np.integer):
            return self._get_integer_adjacency_type(
                adjacency=adjacency,
                is_causal=is_causal
            )

        if not np.issubdtype(adjacency.dtype, np.floating):
            return None

        int_adjacency = np.rint(adjacency)

        if not np.isclose(adjacency, int_adjacency, rtol=0).all():
            if is_causal:
                return GraphType.CAUSAL_WEIGHTED

            return GraphType.WEIGHTED

        return self._get_integer_adjacency_type(
            adjacency=adjacency,
            is_causal=is_causal
        )


    def _get_integer_adjacency_type(
            self,
            adjacency: np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]],
            is_causal: bool):

        unweighted_causal_components = (-2,-1,0,1,2)
        unweighted_components = (-1,0,1)
        amax = adjacency.max()
        amin = adjacency.min()

        if is_causal:
            if (amax in unweighted_causal_components
                and amin in unweighted_causal_components):
                return GraphType.CAUSAL_UNWEIGHTED

            return GraphType.CAUSAL_WEIGHTED
       
        if amax in unweighted_components and amin in unweighted_components:
            return GraphType.UNWEIGHTED

        return GraphType.WEIGHTED


    def calculate(
            self,
            statistic: Statistic
    ) -> (np.ndarray[tuple[int], np.dtype[np.floating | np.integer]] |
          np.ndarray[tuple[()], np.dtype[np.floating | np.integer]] |
          np.floating |
          np.integer |
          float |
          int):
        
        self._stat_graph_type = None
        self._stat_tags = None
        self._is_stat_causal = None
        adjacency = self._get_statistic_result(statistic)
        is_causal = self._has_causal_signature(statistic)

        if not isinstance(adjacency, np.ndarray):
            raise ValueError("GraphReducer can only reduce statistics "
                "that return an ndarray result.")

        if adjacency.ndim < 2:
            raise ValueError("GraphReducer can only reduce statistics that "
                "return a 2D array result.")

        graph_type = self._get_adjacency_type(adjacency, is_causal)
        supported_graph_types = self._supported_graph_types()

        if graph_type not in supported_graph_types:
            raise ValueError(
                "{name} can only reduce adjacency matrices of the following graph types:\n"
                "\t{graph_types}".format(name=self.get_name(),
                                         graph_types=supported_graph_types))
        
        if graph_type in (GraphType.UNWEIGHTED, GraphType.CAUSAL_UNWEIGHTED):
            adjacency = np.rint(adjacency)

        self._stat_graph_type = graph_type
        self._stat_tags = statistic.get_tags()
        self._is_stat_causal = is_causal
        return super().calculate(statistic)

    
    def reduce(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]]
    ) -> (np.ndarray[tuple[int], np.dtype[np.floating | np.integer]] |
          np.ndarray[tuple[()], np.dtype[np.floating | np.integer]] |
          np.floating |
          np.integer |
          float |
          int):

        if self._stat_graph_type == GraphType.CAUSAL_UNWEIGHTED:
            data = self._reformat_unweighted_causal(adjacency=data)

        return self._reduce_graph(adjacency=data)
    

    @abstractmethod
    def _reduce_graph(
            self,
            adjacency: np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]]
    ) -> (np.ndarray[tuple[int], np.dtype[np.floating | np.integer]] |
          np.ndarray[tuple[()], np.dtype[np.floating | np.integer]] |
          np.floating |
          np.integer |
          float |
          int):
        pass
        
    

class GraphSummaryReducer(GraphReducer):

    def __init__(
            self,
            summary_type: SummaryType = "mean"):

        self._summary_type = summary_type
        super().__init__()


    @abstractmethod
    def _reduce_graph(
            self,
            adjacency: np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]]
    ) -> np.floating | np.integer | float | int:
        pass
