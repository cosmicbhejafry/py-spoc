#%%

##############
# STATISTICS #
##############
stats = []

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

#%%

################
# CONFIG BUILD #
################

from pyspoc.config import Config

cfg = Config("kclustering")

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

#%%
import numpy as np

from pyspoc.calculator import Calculator

X = np.random.standard_normal(size=(1000,100))
calc = Calculator(X, name="kclustering_test")
calc.compute(cfg)