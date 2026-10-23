#%%
all_labels = set()
#%%

#%%
import os
from pyspoc.rstatistics.base import Statistic

stats, exps = Statistic.get_available()
yaml = ""
curr_module = None
prev_module = None

for stat in stats:
    used_labels = False

    try:
        curr_module = stat.__module__
    except:
        curr_module = None

    try:
        long_name = stat.get_long_name()
    except:
        long_name = ""

    try:
        identifier = stat.get_identifier()
    except:
        identifier = ""
    
    try:
        tags = [t.value for t in stat.get_tags()]
    except:
        tags = []

    try:
        type = stat.get_type().value
    except:
        type = ""

    try:
        signatures = [s.value for s in stat.get_signatures()]
    except:
        signatures = ""

    yaml_entry = ""
    
    if curr_module is not None and curr_module != prev_module:
        yaml_entry += "# Module: " + curr_module + "\n"

    prev_module = curr_module

    yaml_entry += (stat.get_name() + ":\n"
        "  Long Name: " + long_name + "\n"
        "  Identifier: " + identifier + "\n"
        "  Type: " + type + "\n"
        "  Tags:\n    - " + "\n    - ".join(tags) + "\n"
        "  Signatures:\n    - " + "\n    - ".join(signatures) + "\n\n")
    
    yaml += yaml_entry


with open("d:/Projects/hcda/py-spoc/pyspoc/tags.yml", "w") as y:
    y.write(yaml)

#%%
from pyspoc.rstatistics.base import ReducedStatistic

stats, exps = ReducedStatistic.get_available()
stats

#%%
from pyspoc.reducers.base import Reducer

reducers, exps = Reducer.get_available()
reducers