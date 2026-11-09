#%%

##############
# STATISTICS #
##############
stats = []

from pyspoc.statistics.association.distance import (
    DistanceCorrelation,
    HellerHellerGorfine,
    MultiscaleGraphCorrelation)

dc1 = DistanceCorrelation("manhattan")
stats.append(("manhattan", dc1))

dc2 = DistanceCorrelation("euclidean")
stats.append(("euclidean", dc2))

mgc1 = MultiscaleGraphCorrelation("manhattan")
stats.append(("manhattan", mgc1))

mgc2 = MultiscaleGraphCorrelation("euclidean")
stats.append(("euclidean", mgc2))

#%%
from pyspoc.statistics.causal import (
    ConditionalDistributionSimilarity,
    InformationGeometricConditionalIndependence,
    RegressionErrorCausalInference)

cds = ConditionalDistributionSimilarity()
stats.append(cds)

igci = InformationGeometricConditionalIndependence()
stats.append(igci)

reci = RegressionErrorCausalInference()
stats.append(reci)

#%%
from pyspoc.statistics.clustering import (
    KMeansSimilarity,
    KMedoidsSimilarity,
    DBScanSimilarity)

kmeans = KMeansSimilarity(k=5)
stats.append(kmeans)

kmeds = KMedoidsSimilarity(k=5)
stats.append(kmeds)

dbs = DBScanSimilarity(eps=None, desired_k=5)
stats.append(dbs)

#%%
from pyspoc.statistics.distance import (
    GromovWassersteinTau,
    PairwiseDistance
)

gwt = GromovWassersteinTau()
stats.append(gwt)

pd1 = PairwiseDistance("manhattan")
stats.append(("manhattan", pd1))

pd2 = PairwiseDistance("euclidean")
stats.append(("euclidean", pd2))

#%%
from pyspoc.statistics.distribution import (
    PairwiseQuantileDistance,
    KullbackLeiblerDivergence
)

pqdhalf = PairwiseQuantileDistance(0.5)
stats.append(("q0.5", pqdhalf))

pqd1 = PairwiseQuantileDistance(1)
stats.append(("q1", pqd1))

pqd2 = PairwiseQuantileDistance(2)
stats.append(("q2", pqd2))

pqd3 = PairwiseQuantileDistance(3)
stats.append(("q3", pqd3))

kld = KullbackLeiblerDivergence()
stats.append(kld)

#%%
from pyspoc.statistics.information import MutualInformation

mi3 = MutualInformation(n_neighbors=3)
stats.append(("nn3", mi3))

mi5 = MutualInformation(n_neighbors=5)
stats.append(("nn5", mi5))

#%%
from pyspoc.statistics.ot import (
    PairwiseSinkhornDivergence,
    PairwiseSinkhornPlanEffectiveDestinationsSummary,
    PairwiseSinkhornPlanEntropy,
    PairwiseWassersteinDistance
)

pspedse = PairwiseSinkhornPlanEffectiveDestinationsSummary(
    loss="euclidean", summary_type="mean")
stats.append(pspedse)

pspee = PairwiseSinkhornPlanEntropy(
    loss="euclidean", entropy_type="mass-weighted")
stats.append(pspee)

pwd1 = PairwiseWassersteinDistance(1)
stats.append(("w1", pwd1))

pwd2 = PairwiseWassersteinDistance(2)
stats.append(("w2", pwd2))
#%%

############
# REDUCERS #
############

from pyspoc.reducers.basic import (
    Moment,
    SingularValues,
    EigenValues,
    LogDeterminant
)

reducers = []

mom = Moment(moments=[2,3,4])
reducers.append(mom)

sv = SingularValues(num_values=5)
reducers.append(sv)

eig = EigenValues(num_values=5)
reducers.append(eig)

ld = LogDeterminant(scaled=True)
reducers.append(ld)
#%%

######################
# REDUCED STATISTICS #
######################

from pyspoc.rstatistics.clustering.kclustering.kmeans import (
    KMeansInertia,
    KMeansSILScores,
    KMeansDispersionSummaries
)

from pyspoc.rstatistics.clustering.kclustering.kmedoids import (
    KMedoidsInertia,
    KMedoidsSILScores,
    KMedoidsDispersionSummaries
)

from pyspoc.rstatistics.fractal.corrsum import CorrelationSum

from pyspoc.rstatistics.symmetry.reflection import HouseholderEmpiricalIsotropy

from pyspoc.rstatistics.symmetry.rotation import (
    EigenIsotropy,
    GivensEmpiricalIsotropy,
    HarrEmpiricalIsotropy
)

rstats = []

kmeansmi = KMeansInertia(min_k=5, max_k=15)
rstats.append(kmeansmi)

kmeanssil = KMeansSILScores(min_k=5, max_k=15)
rstats.append(kmeanssil)

kmeansds = KMeansDispersionSummaries(min_k=5, max_k=15)
rstats.append(kmeansds)

kmedsmi = KMedoidsInertia(min_k=5, max_k=15)
rstats.append(kmedsmi)

kmedssil = KMedoidsSILScores(min_k=5, max_k=15)
rstats.append(kmedssil)

kmedsds = KMedoidsDispersionSummaries(min_k=5, max_k=15)
rstats.append(kmedsds)

cs = CorrelationSum()
rstats.append(cs)

hi = HouseholderEmpiricalIsotropy()
rstats.append(hi)

ei = EigenIsotropy()
rstats.append(ei)

gei = GivensEmpiricalIsotropy()
rstats.append(gei)

hei = HarrEmpiricalIsotropy()
rstats.append(hei)

#%%

################
# CONFIG BUILD #
################

from pyspoc.config import Config

cfg = Config("new_stats")

for stat in stats:
    if isinstance(stat, tuple):
        scheme_name, stat_obj = stat
    else:
        scheme_name = "main"
        stat_obj = stat

    cfg.add_statistic(stat_obj, scheme_name)

for reducer in reducers:
    if isinstance(reducer, tuple):
        scheme_name, reducer_obj = reducer
    else:
        scheme_name = "main"
        reducer_obj = reducer

    cfg.add_reducer(reducer_obj, scheme_name)

for rstat in rstats:
    if isinstance(rstat, tuple):
        scheme_name, rstat_obj = rstat
    else:
        scheme_name = "main"
        rstat_obj = rstat

    cfg.add_statistic(rstat_obj, scheme_name)
#%%

#################
# CONFIG EXPORT #
#################

cfg.export_yaml()