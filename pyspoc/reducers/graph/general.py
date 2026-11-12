import numpy as np
import networkx as nx

from typing import Callable

from pyspoc.types import SummaryType
from pyspoc._utils.numerical import summarize
from ._base import GraphType, GraphReducer

class GraphEdgeCount(GraphReducer):

    _long_name = "Graph Edges Count"
    _identifier = "graph-r-edges"

    def __init__(self):
        super().__init__()

    @classmethod
    def _supported_graph_types(cls):
        return (
            GraphType.UNWEIGHTED,
            GraphType.WEIGHTED,
            GraphType.CAUSAL_UNWEIGHTED,
            GraphType.CAUSAL_WEIGHTED)
    
    
    def _reduce_graph(
            self,
            formatted_adjacency: np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]]
    ) -> np.integer:
        
        counting_adjacency = self._get_edge_counting_adjacency(
            formatted_adjacency=formatted_adjacency)
        return np.count_nonzero(counting_adjacency)


class GraphDegreeSummary(GraphReducer):

    _long_name = "Graph Degree Summary"
    _identifier = "graph-r-degree"
    
    def __init__(
            self,
            summary_type: SummaryType = "mean"):

        self._summary_type = summary_type
        super().__init__()

    
    @classmethod
    def _supported_graph_types(cls):
        return (
            GraphType.UNWEIGHTED,
            GraphType.WEIGHTED,
            GraphType.CAUSAL_UNWEIGHTED,
            GraphType.CAUSAL_WEIGHTED)
    

    def _reduce_graph(
            self,
            formatted_adjacency: np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]]
    ) -> float:
        
        counting_adjacency = self._get_edge_counting_adjacency(
            formatted_adjacency=formatted_adjacency)
        row_neighbours = counting_adjacency.sum(axis=0)
        col_neighbours = counting_adjacency.sum(axis=1)
        degree = np.column_stack((row_neighbours, col_neighbours)).sum(axis=1)
        return float(summarize(degree, self._summary_type))
    

class GraphCliqueCount(GraphReducer):

    _long_name = "Graph Clique Count"
    _identifier = "graph-r-clique"

    def __init__(self):
        super().__init__()

    @classmethod
    def _supported_graph_types(cls):
        return (
            GraphType.UNWEIGHTED,
            GraphType.WEIGHTED,
            GraphType.CAUSAL_UNWEIGHTED,
            GraphType.CAUSAL_WEIGHTED)

    def _reduce_graph(
            self,
            formatted_adjacency: np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]]
    ) -> int:
        undirected_binary = self._get_undirected_adjacency(formatted_adjacency)
        G = nx.Graph(undirected_binary)
        nodes = list(range(undirected_binary.shape[0]))
        n_cliques = 0
        
        for clique in nx.enumerate_all_cliques(G):
            n_cliques += 1

            for node in clique:
                nodes.remove(node)

            if len(nodes) == 0:
                break

        return n_cliques


class GraphBridgeCount(GraphReducer):

    _long_name = "Graph Bridge Count"
    _identifier = "graph-r-bridge"

    def __init__(self):
        super().__init__()

    @classmethod
    def _supported_graph_types(cls):
        return (
            GraphType.UNWEIGHTED,
            GraphType.WEIGHTED,
            GraphType.CAUSAL_UNWEIGHTED,
            GraphType.CAUSAL_WEIGHTED)

    def _reduce_graph(
            self,
            formatted_adjacency: np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]]
    ) -> int:
        undirected_binary = self._get_undirected_adjacency(formatted_adjacency)
        G = nx.Graph(undirected_binary)
        return len(list(nx.bridges(G)))


class DisconnectedGraphReducer(GraphReducer):

    def __init__(
            self,
            use_weighted_average: bool):
        super().__init__()
        self._use_weighted_average = use_weighted_average

    def _reduce_disconnected(
            self,
            G: nx.Graph,
            nx_func: Callable[[nx.Graph], float | int]
    ):
        subgraphs = [G.subgraph(nodes) for nodes in nx.connected_components(G)]

        if self._use_weighted_average:
            tot_nodes = nx.number_of_nodes(G)
            return np.sum(
                [nx_func(G) * nx.number_of_nodes(G) for G in subgraphs]
            ) / tot_nodes

        return np.mean([nx_func(G) for G in subgraphs])


class GraphCentreSize(DisconnectedGraphReducer):

    _long_name = "Graph Centre Size"
    _identifier = "graph-r-centre"

    @classmethod
    def _supported_graph_types(cls):
        return (
            GraphType.UNWEIGHTED,
            GraphType.WEIGHTED,
            GraphType.CAUSAL_UNWEIGHTED,
            GraphType.CAUSAL_WEIGHTED)

    def _reduce_graph(
            self,
            formatted_adjacency: np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]]
    ) -> float | int:
        undirected_binary = self._get_undirected_adjacency(formatted_adjacency)
        G = nx.Graph(undirected_binary)
        f_centre_size = lambda G: len(nx.center(G))
        return self._reduce_disconnected(
            G,
            f_centre_size)
        

class GraphDiameter(DisconnectedGraphReducer):

    _long_name = "Graph Diameter"
    _identifier = "graph-r-diameter"
    
    @classmethod
    def _supported_graph_types(cls):
        return (
            GraphType.UNWEIGHTED,
            GraphType.WEIGHTED,
            GraphType.CAUSAL_UNWEIGHTED,
            GraphType.CAUSAL_WEIGHTED)

    def _reduce_graph(
            self,
            formatted_adjacency: np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]]
    ) -> float | int:
        undirected_binary = self._get_undirected_adjacency(formatted_adjacency)
        G = nx.Graph(undirected_binary)
        return self._reduce_disconnected(
            G, nx.diameter)


class GraphRadius(DisconnectedGraphReducer):

    _long_name = "Graph Radius"
    _identifier = "graph-r-radius"
    
    @classmethod
    def _supported_graph_types(cls):
        return (
            GraphType.UNWEIGHTED,
            GraphType.WEIGHTED,
            GraphType.CAUSAL_UNWEIGHTED,
            GraphType.CAUSAL_WEIGHTED)

    def _reduce_graph(
            self,
            formatted_adjacency: np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]]
    ) -> float:
        undirected_binary = self._get_undirected_adjacency(formatted_adjacency)
        G = nx.Graph(undirected_binary)
        return self._reduce_disconnected(
            G, nx.radius)


class GraphPeripheryCount(DisconnectedGraphReducer):

    _long_name = "Graph Periphery Count"
    _identifier = "graph-r-periphery"
    

    @classmethod
    def _supported_graph_types(cls):
        return (
            GraphType.UNWEIGHTED,
            GraphType.WEIGHTED,
            GraphType.CAUSAL_UNWEIGHTED,
            GraphType.CAUSAL_WEIGHTED)

    def _reduce_graph(
            self,
            formatted_adjacency: np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]]
    ) -> float | int:
        undirected_binary = self._get_undirected_adjacency(formatted_adjacency)
        G = nx.Graph(undirected_binary)
        f_periphery = lambda G: len(nx.periphery(G))
        return self._reduce_disconnected(
            G,
            f_periphery)

class GraphKCoreSummary(GraphReducer):

    _long_name = "Graph K-Core Summary"
    _identifier = "graph-r-kcore"

    def __init__(
            self,
            summary_type: SummaryType = "mean"):

        self._summary_type = summary_type
        super().__init__()

    @classmethod
    def _supported_graph_types(cls):
        return (
            GraphType.UNWEIGHTED,
            GraphType.WEIGHTED,
            GraphType.CAUSAL_UNWEIGHTED,
            GraphType.CAUSAL_WEIGHTED)

    def _reduce_graph(
            self,
            formatted_adjacency: np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]]
    ) -> int | float | np.integer | np.floating:
        undirected_binary = self._get_undirected_adjacency(formatted_adjacency)
        G = nx.Graph(undirected_binary)
        core_number = nx.core_number(G)
        core_number_arr = np.array(list(core_number.values()))
        return summarize(core_number_arr, self._summary_type)


class GraphLocalEfficiency(GraphReducer):

    _long_name = "Graph Local Efficiency"
    _identifier = "graph-r-lefficiency"

    def __init__(self):
        super().__init__()

    @classmethod
    def _supported_graph_types(cls):
        return (
            GraphType.UNWEIGHTED,
            GraphType.WEIGHTED,
            GraphType.CAUSAL_UNWEIGHTED,
            GraphType.CAUSAL_WEIGHTED)

    def _reduce_graph(
            self,
            formatted_adjacency: np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]]
    ) -> int | float | np.integer | np.floating:
        undirected_binary = self._get_undirected_adjacency(formatted_adjacency)
        G = nx.Graph(undirected_binary)
        return nx.local_efficiency(G)

    

class GraphGlobalEfficiency(GraphReducer):

    _long_name = "Graph Global Efficiency"
    _identifier = "graph-r-gefficiency"

    def __init__(self):
        super().__init__()

    @classmethod
    def _supported_graph_types(cls):
        return (
            GraphType.UNWEIGHTED,
            GraphType.WEIGHTED,
            GraphType.CAUSAL_UNWEIGHTED,
            GraphType.CAUSAL_WEIGHTED)

    def _reduce_graph(
            self,
            formatted_adjacency: np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]]
    ) -> int | float | np.integer | np.floating:
        undirected_binary = self._get_undirected_adjacency(formatted_adjacency)
        G = nx.Graph(undirected_binary)
        return nx.global_efficiency(G)
