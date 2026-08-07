#%%
import re
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

#%%
base_data_path = Path("/home/gsc225/Projects/hcda/data_gathering/")
automl_path = base_data_path / "AutoML"
automl_format = ".npy"
automl_files = find_files(automl_path, automl_format)
automl_files
#%%
#%%
openml_format = ".parquet"
openml_path = base_data_path / "OpenML"
openml_files = find_files(openml_path, openml_format)
openml_files
#%%
openml_registry_path = openml_path / "dataset_catalogue.csv"
openml_registry = get_file_registry(openml_registry_path)
openml_registry

for file in openml_files:
    if not file.name.startswith("X"):
        continue

    file_name = file.name.rstrip(openml_format)
    dataset_id = int(file_name.split("_")[1])
    dataset_name = openml_registry.loc[dataset_id, "Name"]
    data = pd.read_parquet(file)
    break

#%%
calc = pyspoc.Calculator(data, name=dataset_name, normalise=True)
#%%
calc.compute(cfg)