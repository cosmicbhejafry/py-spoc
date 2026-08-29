# %%
import pyspoc
import pandas as pd
import numpy as np
# %%
cfg = pyspoc.Config.from_yaml_file("testing", "cfg.yaml")
# %%
from typing import Literal, Generator
from pathlib import Path
# %%

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
#%%

base_data_path = Path("D:\\OneDrive\\OneDrive - Imperial College London\\Projects\\hcda\\data_gathering")
openml_format = ".parquet"
openml_path = base_data_path / "OpenML"
openml_files = find_files(openml_path, openml_format)
openml_registry_path = openml_path / "dataset_catalogue.csv"
openml_registry = get_file_registry(openml_registry_path)
result_index_list = list()
results = None
#n_files = len(openml_files)
n_files = 1
sample_shape = (10000,50)

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
    dataset = pyspoc.Dataset(data)
    
    print(f"Calculating stats for dataset {dataset_name} [{i+1}/{n_files}].")
    print("-" * 100)
    
    data_subsets = dataset.get_samples(
        n_samples=1,
        sample_shape=sample_shape,
        sampling_type=["random","cyclical"],
        seed=0)

    n_subsets = len(data_subsets)
    
    for j, subset in enumerate(data_subsets):

        if n_subsets > 1:
            print()
            print(f"Processing subset {j+1}/{n_subsets}.")
            print("-" * 100)
        
        calc = pyspoc.Calculator(subset, name=dataset_name, normalise=True)
        calc.compute(
            config=cfg,
            subsampling_type = "none",
            subsamplng_size = None,
            n_samples = None,
            debug=True)
        result = calc.show_results("short")
        result_index_list.append((f"{dataset_name}_{j}", dataset_modality, dataset_domain))

        if results is None:
            results = result
        else:
            results = pd.concat((results, result))

    if i % 10 == 0 and results is not None:
        results.to_parquet("results2.parquet", engine="pyarrow")

if results is not None:
    results.to_parquet("results2.parquet", engine="pyarrow")

results