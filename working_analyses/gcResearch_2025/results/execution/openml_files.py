import pandas as pd

from pathlib import Path
from typing import Iterator

from .annotated_dataset import AnnotatedDataset


def find_files(openml_folder: str | Path, extension: str) -> list[tuple[Path, Path]]:
    folder = Path(openml_folder)
    extension = extension.lstrip(".")
    files = list(folder.rglob(f"*.{extension}"))
    paired_files = list()
    n_files = len(files)

    while files:
        file = files.pop()
        n_files -= 1
        file_name = file.stem
        file_parts = file_name.split("_")

        if len(file_parts) == 1:
            paired_files.append((file,))
            continue

        if len(file_parts) == 2:
            file_type, file_id = file_parts
            paired_type = "y" if file_type == "X" else "X"
            paired_file_name = paired_type + "_" + file_id

            for i in range(n_files-1,-1,-1):
                alt_file = files[i]
                alt_file_name = alt_file.stem

                if alt_file_name != paired_file_name:
                    continue

                if paired_type == "y":
                    paired_files.append((file, alt_file))
                else:
                    paired_files.append((alt_file, file))

                files.remove(alt_file)
                n_files -= 1
                break

    return paired_files


def get_file_registry(openml_registry_path: str | Path) -> pd.DataFrame:
    file_path = Path(openml_registry_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Path {file_path} does not exist.")

    if not file_path.is_file():
        raise IsADirectoryError(f"Path {file_path} is not a file.")

    if not file_path.suffix != "csv":
        raise ValueError(f"Path {file_path} is not a CSV file.")

    return pd.read_csv(file_path, index_col="DatasetID")


def get_annotated_datasets(
        openml_registry_path: str | Path,
        openml_folder: str | Path,
        extension: str = "parquet", # Only parquet files are supported, but sometimes "pq" used.
        include_target: bool = False
    ) -> Iterator[AnnotatedDataset]:

    openml_registry = get_file_registry(openml_registry_path)
    openml_files = find_files(openml_folder=openml_folder, extension=extension)
    n_files = len(openml_files)

    for i, files in enumerate(openml_files):

        X_file = files[0]

        if not X_file:
            print(f"Skipping dataset {i+1}/{n_files} due to lack of design matrix.")
            continue

        data_paths = [X_file]

        if include_target and len(files) > 1:
            y_file = files[1]
            data_paths.append(y_file)

        file_name = X_file.stem
        dataset_id = int(file_name.split("_")[1])
        dataset_info = openml_registry.loc[dataset_id]
        dataset_name = dataset_info.Name
        dataset_modality = dataset_info.Modality
        dataset_domain = dataset_info.Domain

        annotated = AnnotatedDataset(
            data_paths=data_paths,
            modality=dataset_modality,
            domain=dataset_domain,
            name=dataset_name)

        yield annotated
