#%%
from pyspoc import Statistic
len(Statistic.get_available()[0])
#%%
from pyspoc import Reducer
len(Reducer.get_available()[0])
#%%
from pyspoc import ReducedStatistic
len(ReducedStatistic.get_available()[0])
#%%
glb_copy = dict(globals())
stats = set()

for obj in glb_copy.values():
    obj_cls = type(obj)

    if issubclass(obj_cls, Statistic):
        stats.add(obj_cls)