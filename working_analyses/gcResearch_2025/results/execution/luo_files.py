import anndata as ad
import pandas as pd
import re

from pathlib import Path
from typing import Iterator

from .annotated_dataset import AnnotatedDataset


def get_patient_files(
        data_path: Path
    ) -> dict[int, Path]:

    pattern = re.compile(r"Patient(\d+)_RNA_GEX\.pq")
    patient_files = {}

    for path in data_path.glob("Patient*_RNA_GEX.pq"):
        if match := pattern.fullmatch(path.name):
            patient_number = int(match.group(1))
            patient_files[patient_number] = path

    return patient_files


def get_annotated_datasets(
        luo_folder: str | Path,
        base_exp_file: str | None = None,
        refresh: bool = False,
        use_pca: bool = False,
) -> Iterator[AnnotatedDataset]:

    data_path = Path(luo_folder)

    if refresh:
        # Get gene expression matrix

        if base_exp_file is None:
            raise ValueError(
                "base_exp_file must be provided when refresh is True.")

        exp_file_path = data_path / base_exp_file
        adata = ad.read_h5ad(exp_file_path)

        # Filter on T cells (largest cohort of cells).
        T_cell_mask = adata.obs["manual_annot_leiden_res0.21"] == "T"
        T_cells_df: pd.DataFrame = adata.to_df().loc[T_cell_mask] # pyright: ignore[reportCallIssue]
        T_cells_donors = adata.obs["donor_num"].loc[T_cell_mask] # pyright: ignore[reportCallIssue]
        T_cells_df.index = pd.MultiIndex.from_arrays(
            [T_cells_df.index.to_list(),
            T_cells_donors.to_list()],
            names=["Sequence", "DonorNum"])

        # Save to parquet
        T_cells_df.to_parquet(data_path / "Luo_PBMC_RNA_GEX_only.pq")
        donor_dfs = {num : group for num, group in T_cells_df.groupby("DonorNum")}

        # Split the parquet by donor number
        for donor_num, donor_df in donor_dfs.items():
            file_name = (f"Patient{donor_num}_RNA_GEX.pq" if not use_pca
                else f"Patient{donor_num}_RNA_GEX.pca.pq")
            file_path = data_path / file_name

            if use_pca:
                from sklearn.decomposition import PCA
                donor_pca_arr = PCA(n_components=50).fit_transform(donor_df)
                donor_df = pd.DataFrame(
                    data = donor_pca_arr,
                    index = donor_df.index,
                    columns=pd.Index(data=range(50), name="PCA"))

            if not file_path.exists():
                donor_df.to_parquet(file_path)

        del donor_df

    # Grab references to our files again.
    donor_files = get_patient_files(data_path)

    # Iterate through and yield them.
    for donor_num, donor_file in donor_files.items():
        annotated = AnnotatedDataset(
            data_paths=[donor_file],
            modality="Gene",
            domain="Gene",
            name=f"Patient{donor_num}_Expression")

        yield annotated
