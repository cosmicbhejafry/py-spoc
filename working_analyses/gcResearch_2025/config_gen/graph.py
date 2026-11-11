#%%
from pyspoc.statistics.graph._base import GraphStatistic
GraphStatistic.get_available()

#%%

##############
# STATISTICS #
##############

graph_stats = []

#%%
from pyspoc.statistics.graph.association import (
    CovarianceGraph,
    RegressionGraph
)

lw_graph = CovarianceGraph(estimator="LedoitWolf")
graph_stats.append(lw_graph)

lasso_graph = RegressionGraph(estimator="lasso", symmetrize=None, alpha=0.1)
graph_stats.append(lasso_graph)
#%%
from pyspoc.statistics.graph.gaussian import GGM

ggm = GGM(inversion_method="graphical_lasso", alpha=0.1, max_iter=1000)
graph_stats.append(ggm)
#%%
from pyspoc.statistics.graph.causal.constraint_based import (
    PCCausalGraph
)

pc = PCCausalGraph()
graph_stats.append(pc)
#%%
from pyspoc.statistics.graph.causal.functional import (
    ICALiNGAMCausalGraph,
    DirectLiNGAMCausalGraph,
    RCDLiNGAMCausalGraph
)

ica = ICALiNGAMCausalGraph()
graph_stats.append(ica)

dica = DirectLiNGAMCausalGraph()
graph_stats.append(dica)

rcd = RCDLiNGAMCausalGraph()
graph_stats.append(rcd)
#%%
from pyspoc.statistics.graph.information import ChowLiuTree

chow = ChowLiuTree()
graph_stats.append(chow)
#%%
from pyspoc.statistics.graph.proximity import DistributionGraph

dist = DistributionGraph(metric="wasserstein", bins=128)
graph_stats.append(dist)

#%%

############
# REDUCERS #
############

graph_reducers = []

#%%
from pyspoc.reducers.graph import (
    GraphEdgeCount,
    GraphCliqueCount,
    GraphPeripheryCount,
    GraphLocalEfficiency,
    GraphGlobalEfficiency,
    GraphDiameter,
    GraphRadius,
    GraphCentreSize,
    GraphBridgeCount
)

edges = GraphEdgeCount()
graph_reducers.append(edges)

cliques = GraphCliqueCount()
graph_reducers.append(cliques)

periphery = GraphPeripheryCount(use_weighted_average=True)
graph_reducers.append(periphery)

local_eff = GraphLocalEfficiency()
graph_reducers.append(local_eff)

global_eff = GraphGlobalEfficiency()
graph_reducers.append(global_eff)

diameter = GraphDiameter(use_weighted_average=True)
graph_reducers.append(diameter)

radius = GraphRadius(use_weighted_average=True)
graph_reducers.append(radius)

centre_size = GraphCentreSize(use_weighted_average=True)
graph_reducers.append(centre_size)

bridges = GraphBridgeCount()
graph_reducers.append(bridges)

#%%
from pyspoc.reducers.graph.directed.dag import (
    DAGLongestPathLength,
    DAGTopologicalDepth
)

dag_longest_path = DAGLongestPathLength()
graph_reducers.append(dag_longest_path)

dag_depth = DAGTopologicalDepth()
graph_reducers.append(dag_depth)

#%%
nonuq_graph_reducers = []

from pyspoc.reducers.graph.directed import (
    DiGraphReciprocity,
    DiGraphDirectedDegreeSummary,
    DiGraphStrengthSummary,
)

from pyspoc.reducers.graph import (
    GraphDegreeSummary,
    GraphKCoreSummary
)

recip = DiGraphReciprocity()
graph_reducers.append(recip)

indegree_mean = DiGraphDirectedDegreeSummary(direction="in", summary_type="mean")
nonuq_graph_reducers.append((indegree_mean, "in-mean"))

indegree_range = DiGraphDirectedDegreeSummary(direction="in", summary_type="range")
nonuq_graph_reducers.append((indegree_range, "in-range"))

outdegree_mean = DiGraphDirectedDegreeSummary(direction="out", summary_type="mean")
nonuq_graph_reducers.append((outdegree_mean, "out-mean"))

outdegree_range = DiGraphDirectedDegreeSummary(direction="out", summary_type="range")
nonuq_graph_reducers.append((outdegree_range, "out-range"))

instrength = DiGraphStrengthSummary(direction="in", summary_type="mean")
nonuq_graph_reducers.append((instrength, "in"))

outstrength = DiGraphStrengthSummary(direction="out", summary_type="mean")
nonuq_graph_reducers.append((outstrength, "out"))

degree_mean = GraphDegreeSummary(summary_type="mean")
nonuq_graph_reducers.append((degree_mean, "mean"))

degree_range = GraphDegreeSummary(summary_type="range")
nonuq_graph_reducers.append((degree_range, "range"))

kcore_mean = GraphKCoreSummary(summary_type="mean")
nonuq_graph_reducers.append((kcore_mean, "mean"))

kcore_range = GraphKCoreSummary(summary_type="range")
nonuq_graph_reducers.append((kcore_range, "range"))

#%%
from pyspoc.config import Config

cfg = Config("graph")

for stat in graph_stats:
    cfg.add_statistic(stat, scheme_name="main")

for reducer in graph_reducers:
    cfg.add_reducer(reducer, scheme_name="main")

for reducer, scheme_name in nonuq_graph_reducers:
    cfg.add_reducer(reducer, scheme_name=scheme_name)
#%%
import numpy as np
from pyspoc.calculator import Calculator
from pyspoc.dataset import Dataset

X = np.random.standard_normal(size=(1000,100))
X_dataset = Dataset(X)

Y = np.hstack((X[:,:50], X[:,:50] ** 2))
Y_dataset = Dataset(Y)

#%%
calc = Calculator(X,name="graph")
calc.compute(cfg)
#%%
cfg.export_yaml("graph")
#%%
result = ggm.calculate(X_dataset)
#%%
edges.calculate(rcd)
#%%
adjacency = edges._get_statistic_result(rcd)
is_causal = edges._has_causal_signature(rcd)
is_alleged_weighted = edges._is_alleged_weighted(rcd)