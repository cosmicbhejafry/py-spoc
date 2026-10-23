#%%
import pyspoc.statistics.causal as causal
import pyspoc.statistics.association.covariance as covariance
import pyspoc.statistics.association.correlation as corr
import pyspoc.statistics.errors.regressors as ereg
import pyspoc.statistics.distance as dist
import pyspoc.statistics.association.distance.hsic as hsic
import pyspoc.reducers.basic as red
import pyspoc.rstatistics.norms as norms
import pyspoc.rstatistics.ranks as ranks

from pyspoc.config import Config
from pyspoc import (
    Statistic,
    Reducer,
    ReducedStatistic)

# NOTE (GSC): Commented statistics removed as only a small number of their
# features survived coefficient of variation and correlation pruning
# and they were computationally costly.
#%%
import pyspoc.rstatistics.graph.rstatistics

#%%
stats = Statistic.get_available()
stats
#%%
Reducer.get_available()
#%%
ReducedStatistic.get_available()
#%% Gather statistics to use
cfg = Config("esa")
#%% Add causal stats
reci = causal.RegressionErrorCausalInference()
cfg.add_statistic(reci, "std")

#igci = causal.InformationGeometricConditionalIndependence() - Only 4 features survived out of 6000+
#cfg.add_statistic(igci, "std")

#%% Add covariance stats
ee = covariance.EllipticEnvelope()
cfg.add_statistic(ee, "ee")

oas = covariance.OAS()
cfg.add_statistic(oas, "oas")

lw = covariance.LedoitWolf()
cfg.add_statistic(lw, "lw")

sc = covariance.ShrunkCovariance()
cfg.add_statistic(sc, "sc")

kt = corr.KendallTau()
cfg.add_statistic(kt, "std")

sr = corr.SpearmanR(squared=False)
cfg.add_statistic(sr, "std")

orth_pec = corr.PowerEnvelopeCorrelation(orth=True)
cfg.add_statistic(orth_pec, "orth")

#%% Add regression errors

regressors = ["LinearRegression",
    "RidgeCV",
    "Lasso",
    "ElasticNet",
    "LarsCV",
    # "GraphicalLassoCV" - Had issues with convergence on a few datasets.
    # "LassoLarsCV", - Only 8 of 6000+ features survived.
    "LassoLarsIC",
    "OrthogonalMatchingPursuit",
    #"LinearSVR", - Had quite a few but its timings were too slow.
    "ARDRegression",
    "BayesianRidge",
    #"HuberRegressor" - Only 17 of 6000+ features survived.
    ]

for regressor in regressors:
    reg = ereg.LinearRegressorError(model=regressor)
    cfg.add_statistic(reg, regressor)

#%% Pairwise distances
distances = [
    'manhattan',
    'nan_euclidean',
    "cubic",
    'chebyshev',
    'cosine',
    'braycurtis',
    'canberra',
    'correlation'
]

for distance in distances:
    pd = dist.PairwiseDistance(metric=distance)
    cfg.add_statistic(pd, distance)

gwt = dist.GromovWassersteinTau()
cfg.add_statistic(gwt, "std")

hsic = hsic.HilbertSchmidtIndependenceCriterion()
cfg.add_statistic(hsic, "rbf")
#%% Add reducers
det = red.Determinant()
cfg.add_reducer(det, "scaled")

svd = red.SingularValues(4)
cfg.add_reducer(svd, "four")

# Issues with Lasso and fourth moment - likely due to sparsity.
three_mom = red.Moment(moments=[1,2,3])
cfg.add_reducer(three_mom,
                "three") #,
#                [x for x in cfg.statistics.keys() if not x.endswith("Lasso")])

# three_mom = red.Moment(moments=[1,2,3,4])
# cfg.add_reducer(three_mom,
#                 "three",
#                 [x for x in cfg.statistics.keys() if x.endswith("Lasso")])

ewmn12 = norms.EntryWiseMatrixNorm(p=1,q=2)
cfg.add_reducer(ewmn12, "p1q2")

ewmn21 = norms.EntryWiseMatrixNorm(p=2,q=1)
cfg.add_reducer(ewmn12, "p2q1")

fro = norms.Norm("fro")
cfg.add_reducer(fro, "fro")

nuc = norms.Norm("nuc")
cfg.add_reducer(nuc, "nuc")

l2 = norms.Norm(2)
cfg.add_reducer(l2, "l2")

sn3 = norms.SchattenNorm(p=3)
cfg.add_reducer(sn3, "p3")

erank = ranks.EffectiveRank()
cfg.add_reducer(erank, "std")

srank = ranks.StableRank()
cfg.add_reducer(srank, "std")

#%% Serialise to YAML
cfg.export_yaml("cfg.yml")
