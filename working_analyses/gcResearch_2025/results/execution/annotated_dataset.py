from __future__ import annotations

import pandas as pd

from dataclasses import dataclass
from collections.abc import Sequence
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from pathlib import Path
    

@dataclass
class AnnotatedDataset:
    data_paths: Sequence[Path]
    modality: str
    domain: str
    name: str | None = None

    def get_file_names(self) -> list[str]:
        return [d.stem for d in self.data_paths]

    def get_name(self) -> str:
        return self.name or self.get_file_names()[0]

    def get_data(self) -> pd.DataFrame:
        if len(self.data_paths) == 1:
            return pd.read_parquet(self.data_paths[0])

        data_collection = list()

        for data_path in self.data_paths:
            data = pd.read_parquet(data_path)
            data_collection.append(data)

        return pd.concat(data_collection, axis=1)

