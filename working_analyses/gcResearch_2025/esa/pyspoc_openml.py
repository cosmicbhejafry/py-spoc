#%%
import pyspoc
import numpy as np
import pandas as pd

from pathlib import Path
#%%
cfg = pyspoc.Config.from_yaml_file("testing", "cfg.yaml")
#%%
def find_files(folder: str | Path, extension: str) -> list[Path]:
    folder = Path(folder)
    extension = extension.lstrip(".")
    return list(folder.rglob(f"*.{extension}"))

def get_file_registry(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Path {file_path} does not exist.")

    if not file_path.is_file():
        raise IsADirectoryError(f"Path {file_path} is not a file.")

    if not file_path.suffix != "csv":
        raise ValueError(f"Path {file_path} is not a CSV file.")

    return pd.read_csv(file_path, index_col="DatasetID")

from typing import Literal, Generator

def get_random_consistent_subset(
        required_size: int,
        axis: Literal[0, 1],
        samples: int,
        data: np.ndarray | pd.DataFrame) -> Generator[np.ndarray]:

    data_np = data.values if isinstance(data, pd.DataFrame) else data
    actual_size = data.shape[axis]
    choices = list(range(0,actual_size))
    np.random.seed(0)
    choices_perm = np.random.choice(choices, size=actual_size, replace=False)
    choices_perm_copy = list(choices_perm)

    for i in range(samples):
        vars = choices_perm_copy[:required_size]
        current_size = len(vars)

        if current_size < required_size:
            remaining_size = required_size - current_size
            choices_perm_copy = list(choices_perm)
            vars.extend(choices_perm_copy[:remaining_size])
            choices_perm_copy = choices_perm_copy[remaining_size:]
        else:
            choices_perm_copy = choices_perm_copy[required_size:]

        if axis == 0:
            yield data_np[vars, :]
        else:
            yield data_np[:, vars]
#%%
base_data_path = Path("D:/Projects/hcda/data_gathering/")
openml_format = ".parquet"
openml_path = base_data_path / "OpenML"
openml_files = find_files(openml_path, openml_format)
openml_registry_path = openml_path / "dataset_catalogue.csv"
openml_registry = get_file_registry(openml_registry_path)
result_index_list = list()
results = None
n_files = len(openml_files)
# %%
for i in range(n_files):
    file = openml_files[i]

    if not file.name.startswith("X"):
        print(f"Skipping dataset {i+1}/{n_files} due to lack of design matrix.")
        continue

    file_name = file.name.rstrip(openml_format)
    dataset_id = int(file_name.split("_")[1])
    dataset_info = openml_registry.loc[dataset_id]
    dataset_name = dataset_info.Name
    dataset_modality = dataset_info.Modality
    dataset_domain = dataset_info.Domain
    data = pd.read_parquet(file)

    if data.shape[0] > 10000:
        by_n_subsets = list(get_random_consistent_subset(10000, 0, 2, data))
    else:
        by_n_subsets = [data]

    if data.shape[1] > 50:
        all_subsets = list()

        for subset in by_n_subsets:
            all_subsets.extend(list(get_random_consistent_subset(50, 1, 3, subset)))
    else:
        all_subsets = by_n_subsets
    
    print(f"Calculating stats for dataset {dataset_name} [{i+1}/{n_files}].")
    print("-" * 100)
    
    n_subsets = len(all_subsets)

    for j in range(n_subsets):

        if n_subsets > 1:
            print()
            print(f"Processing subset {j+1}/{n_subsets}.")
            print("-" * 100)

        subset = all_subsets[j]
        calc = pyspoc.Calculator(subset, name=dataset_name, normalise=True)
        calc.compute(cfg)
        result = calc.show_results("full")
        result_index_list.append((f"{dataset_name}_{j}", dataset_modality, dataset_domain))

        if results is None:
            results = result
        else:
            results = pd.concat((results, result))

    if i % 10 == 0 and results is not None:
        results.to_parquet("results.pq", engine="pyarrow")

if results is not None:
    results.to_parquet("results.pq", engine="pyarrow")
#%%
import pandas as pd
import numpy as np

results = pd.read_csv("results.csv", header=[0,1], index_col=0)
results.index = pd.Index([parse_tuple_index(value) for value in results.index])

# %%
import plotting as p
fig, p.plot_pca(results)