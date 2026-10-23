from __future__ import annotations

import pandas as pd
import pyspoc

from typing import TYPE_CHECKING
from pathlib import Path
from datetime import datetime

if TYPE_CHECKING:
    from .annotated_dataset import AnnotatedDataset
    from collections.abc import Iterator, Collection


def run_pyspoc_loop(
        cfg: pyspoc.Config,
        annotated_datasets: Iterator[AnnotatedDataset] | Collection[AnnotatedDataset],
        batch_n: int,
        batch_p: int,
        max_batches: int,
        results_path: str | Path,
        use_batching: bool = True,
        sampling_seed: int | None = None,
        output_files_prefix: str | None = None,
        save_frequency: int = 10,
        raise_errors: bool = True
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    results = list()
    short_results = list()
    timings = list()
    short_timings = list()

    results_df = None
    short_results_df = None
    timings_df = None
    short_timings_df = None
   
    datasets = list(annotated_datasets)
    n = len(datasets)
    output_files_suffix = datetime.now().strftime("%Y%m%d%H%M%S")

    for i, annotated_dataset in enumerate(datasets):

        try:
            data = annotated_dataset.get_data()
            dataset_name = annotated_dataset.get_name()
            dataset = pyspoc.Dataset(data, name=dataset_name)

            if use_batching:
                batches = dataset.get_batches(
                    batch_shape=(batch_n, batch_p),
                    sampling_type="cyclical",
                    seed=sampling_seed,
                    max_n_batches=max_batches)
            else:
                batches = [data]
            
            print(f"Calculating stats for dataset {dataset_name} [{i+1}/{n}].")
            print("-" * 100)
            
            n_batches = len(batches)

            for j in range(n_batches):

                if n_batches > 1:
                    print()
                    print(f"Processing batch {j+1}/{n_batches}.")
                    print("-" * 100)
                    batch_name = dataset_name + f"_B{j+1}"
                else:
                    batch_name = dataset_name

                batch = batches[j]
                calc = pyspoc.Calculator(batch, name=batch_name, normalise=True)
                calc.compute(cfg)

                index = make_results_index(annotated_dataset, batch_name=batch_name)
                result = calc.get_results("full")
                result.index = index

                short_result = calc.get_results("short")
                short_result.index = index

                timing = calc.get_timings("full")
                short_timing = calc.get_timings("short")

                # Collect results.
                results.append(result)
                short_results.append(short_result)

                # Collect timings.
                timings.append(timing)
                short_timings.append(short_timing)

        except Exception as e:
            if raise_errors:
                raise e
            
            print(e)

        finally:
            if i % save_frequency == 0:
                results_df, short_results_df, timings_df, short_timings_df = save_results(
                    results_path,
                    results,
                    short_results,
                    timings,
                    short_timings,
                    output_files_prefix,
                    output_files_suffix)

                results.clear()
                results.append(results_df)

                short_results.clear()
                short_results.append(short_results_df)

                timings.clear()
                timings.append(timings_df)

                short_timings.clear()
                short_timings.append(short_timings_df)

    return save_results(
        results_path,
        results,
        short_results,
        timings,
        short_timings,
        output_files_prefix,
        output_files_suffix)


def make_results_index(
        annotated_dataset: AnnotatedDataset,
        batch_name: int,
    ) -> pd.MultiIndex:

    dataset_name = annotated_dataset.get_name()

    index = pd.MultiIndex.from_arrays(
    [
        [dataset_name],
        [batch_name],
        [annotated_dataset.modality],
        [annotated_dataset.domain]
    ],
    names=[
        "Dataset Name",
        "Batch Name",
        "Modality",
        "Domain"])

    return index


def save_results(
        results_path: Path,
        results: list[pd.DataFrame | None],
        short_results: list[pd.DataFrame | None],
        timings: list[pd.DataFrame | None],
        short_timings: list[pd.DataFrame | None],
        output_files_prefix: str,
        output_files_suffix: str
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    results_df = pd.concat(results)
    short_results_df = pd.concat(short_results)
    timings_df = pd.concat(timings)
    short_timings_df = pd.concat(short_timings)

    output_files_prefix = (""
        if output_files_prefix is None
        else output_files_prefix.rstrip("_") + "_")

    output_files_suffix = (""
        if output_files_suffix is None
        else "_" + output_files_suffix.lstrip("_"))

    outputs = {
        "results" : results_df,
        "short_results": short_results_df,
        "timings": timings_df,
        "short_timings": short_timings_df
    }

    for file_name, df in outputs.items():
        file_path = results_path / (output_files_prefix + file_name + output_files_suffix + ".pq")
        df.to_parquet(file_path)

    return results_df, short_results_df, timings_df, short_timings_df
