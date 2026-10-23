#%%
import numpy as np

from pyspoc.calculator import Calculator
from pyspoc.config import Config

data = np.random.standard_normal(size=(10000,10))

calc = Calculator(
    dataset=data,
    name="testing",
)
cfg = Config.from_internal("test_cfg")
calc.compute(cfg)
#%%
import pandas as pd

def get_results(
    calc: Calculator,
    display_names: str = "full") -> pd.DataFrame | None:

    result_row = []
    first_level = []
    second_level = []

    for stat_full_name, reducers in calc._results_dict.items():
        if not isinstance(reducers, dict):
            continue

        for reducer_full_name, reduced_result in reducers.items():
            if isinstance(reduced_result, np.ndarray):
                size = reduced_result.size
                reduced_result = reduced_result.reshape(-1)
                result_row.extend(reduced_result)
            else:
                result_row.append(reduced_result)
            
            stat_name = calc._get_display_name(stat_full_name, display_names)
            reducer_name = calc._get_display_name(reducer_full_name, display_names)
            first_level.extend([stat_name] * size)
            second_level_names = ([reducer_name] if size == 1
                else [f"{i+1}" for i in range(size)] if reducer_name == ""
                else [f"{reducer_name}_{i+1}" for i in range(size)]
            )
            second_level.extend(second_level_names)

    
    if not first_level or not second_level:
        
        return None

    columns = [*zip(first_level, second_level)]
    results_table = pd.DataFrame(
        data = {
            col: datum
            for col, datum in zip(columns, result_row, strict=True)
        },
        index=[calc.name])
    results_table.columns.names = ["Statistic", "Reducer"]
    return results_table
#%%
display_names = "full"
result_row = []
first_level = []
second_level = []

for stat_full_name, reducers in calc._results_dict.items():
    for reducer_full_name, reduced_result in reducers.items():
        if isinstance(reduced_result, np.ndarray):
            size = reduced_result.size
            reduced_result = reduced_result.reshape(-1)
            result_row.extend(reduced_result)
        else:
            result_row.append(reduced_result)
        
        stat_name = calc._get_display_name(stat_full_name, display_names)
        reducer_name = calc._get_display_name(reducer_full_name, display_names)
        first_level.extend([stat_name] * size)
        second_level_names = ([reducer_name] if size == 1
            else [f"{i+1}" for i in range(size)] if reducer_name == ""
            else [f"{reducer_name}_{i+1}" for i in range(size)]
        )
        second_level.extend(second_level_names)

columns = [*zip(first_level, second_level)]
results_table = pd.DataFrame(
    data = {
        col: datum
        for col, datum in zip(columns, result_row, strict=True)
    },
    index=[calc.name])
results_table.columns.names = ["Statistic", "Reducer"]
return results_table