from __future__ import annotations

import pandas as pd
import numpy as np

from tqdm import tqdm
from typing import Literal


def reduce_stat_name(
        stat_name: str,
    ) -> str:

    stat_parts = stat_name.split(".")
    
    for i in range(len(stat_parts)-1,0,-1):
        curr_part = stat_parts[i]
        prev_part = stat_parts[i-1]

        if curr_part == prev_part:
            stat_parts.pop(i)

    return (".".join(stat_parts))


def remove_low_cv_features(
        results: pd.DataFrame,
        epsilon: float = 1e-6
    ) -> pd.DataFrame:

    epsilon = 1e-6
    non_zero_abs_cv = (np.abs(results.std() / results.mean()) >= epsilon)
    results = results.loc[:,non_zero_abs_cv]
    return results


def remove_colinear_features(
        results: pd.DataFrame,
        timings: pd.DataFrame,
        r: float = 0.95) -> pd.DataFrame:

    mean_timings = timings.mean()
    results_corr = results.corr().values
    results_corr = np.array(results_corr)
    results_corr[np.tril_indices_from(results_corr, k=0)] = 0
    abs_results_corr = np.abs(results_corr)
    n_vars = abs_results_corr.shape[0]
    i = 0
    mask = np.full(shape=n_vars, fill_value=True)

    while i < n_vars:
        var_corr = abs_results_corr[i]

        if np.all(np.isnan(var_corr)):
            i += 1
            continue

        max_corr_idx = int(np.nanargmax(var_corr))
        max_corr = var_corr[max_corr_idx]

        if max_corr < r:
            print(f"No high correlation found for variable {i}. Continuing.")
            i += 1
            continue

        var_name = results.columns[i]
        var_timing = mean_timings.loc[var_name[0]]
        other_var_name = results.columns[max_corr_idx]
        other_var_timing = mean_timings.loc[other_var_name[0]]
        removal_idx = i if var_timing > other_var_timing else max_corr_idx
        mask[removal_idx] = False
        abs_results_corr[removal_idx,:] = np.nan
        abs_results_corr[:,removal_idx] = np.nan
        print(f"High correlation found for variable {i}. "
            f"Removing variable {removal_idx} due to slower timing.")

    return results.loc[:,mask]


def remove_features_by_vif(
        data: pd.DataFrame,
        threshold: float,
        min_columns: int,
        max_inf_vif_columns_to_remove: int = 5,
        nan_policy: Literal[
            "report",
            "col-omit",
            "row-omit"] = "report"
    ) -> tuple[pd.DataFrame, pd.Series]:
    
    """Iteratively remove features with excessive variance inflation.

    At each iteration, the variance inflation factor (VIF) is calculated for
    every remaining column and the column with the largest VIF is removed. The
    process stops when every VIF is at or below ``threshold`` or removing
    another column would reduce the result below ``min_columns``.

    Parameters
    ----------
    data : pandas.DataFrame
        Numeric feature matrix containing finite values.
    threshold : float
        Maximum permitted VIF. A common choice is 5 or 10.
    min_columns : int
        Minimum number of columns to retain.
    max_inf_vif_columns_to_remove : int, default=5
        Maximum number of columns to remove in a single chunk if they all have
        infinite VIF score.
    nan_policy : {"report", "col-omit" "row-omit"}, default="report"
        How columns with NaN entries are handled. "report" will raise an error,
        "col-omit" will remove the offending columns, "row-omit" will remove the
        offending rows.

    Returns
    -------
    trimmed_data : pandas.DataFrame
        Copy of ``data`` containing the retained columns. The input frame is
        not modified.
    vif_scores : pandas.Series
        Final VIF of each retained column, indexed by its original label.

    Raises
    ------
    TypeError
        If the arguments have incompatible types or the frame contains
        non-numeric columns.
    ValueError
        If ``threshold`` is not finite and positive, ``min_columns`` is outside
        the available column range, or the data contain missing or infinite
        values.

    Notes
    -----
    A VIF of one is assigned when only one feature remains because there are
    no other features capable of explaining its variance.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    if isinstance(threshold, bool) or not isinstance(
            threshold, (int, float, np.integer, np.floating)):
        raise TypeError("threshold must be a real number.")
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be finite and greater than zero.")
    if isinstance(min_columns, bool) or not isinstance(
            min_columns, (int, np.integer)):
        raise TypeError("min_columns must be an integer.")
    if min_columns < 1 or min_columns > data.shape[1]:
        raise ValueError(
            "min_columns must be between one and the number of data columns."
        )

    non_numeric = data.select_dtypes(exclude=np.number).columns
    if len(non_numeric) > 0:
        raise TypeError(
            "VIF feature removal requires numeric columns; received "
            f"non-numeric columns {list(non_numeric)!r}."
        )

    # Work on an independent frame so both column removal and any future
    # preprocessing added here cannot affect the caller's data.
    trimmed_data = data.copy(deep=True)
    values = trimmed_data.to_numpy(dtype=float, copy=False)
    is_finite_array = np.isfinite(values)

    if not is_finite_array.all():
        if nan_policy == "report":
            raise ValueError("VIF feature removal requires finite, non-missing data.")

        if nan_policy == "col-omit":
            finite_mask = np.all(is_finite_array, axis=0)
            trimmed_data = trimmed_data.loc[:,finite_mask]

        if nan_policy == "row-omit":
            finite_mask = np.all(is_finite_array, axis=1)
            trimmed_data = trimmed_data.loc[finite_mask]

    def calculate_vif(frame: pd.DataFrame) -> pd.Series:
        """Calculate VIF values while retaining the frame's column index."""
        if frame.shape[1] == 1:
            return pd.Series([1.0], index=frame.columns, name="VIF")

        matrix = frame.to_numpy(dtype=float, copy=False)

        with np.errstate(divide="ignore", invalid="ignore"):
            scores = []
            progress = tqdm(range(matrix.shape[1]))

            for column_index in progress:
                name = frame.columns[column_index]
                progress.set_description(str(name))
                score = variance_inflation_factor(matrix, column_index)
                scores.append(score)

        return pd.Series(scores, index=frame.columns, dtype=float, name="VIF")

    print("Beginning VIF filtering...")
    vif_scores = calculate_vif(trimmed_data)

    while trimmed_data.shape[1] > min_columns:
        # An infinite VIF correctly wins this comparison. Undefined VIFs are
        # ignored; if every score is undefined there is no defensible column
        # choice, so return the current frame and scores unchanged.
        comparable_scores = vif_scores.to_numpy(copy=True)
        comparable_scores[np.isnan(comparable_scores)] = -np.inf

        if np.all(np.isneginf(comparable_scores)):
            break

        if max_inf_vif_columns_to_remove > 1:
            inf_mask = np.isposinf(comparable_scores)

            if np.any(inf_mask):
                removal_positions = np.argwhere(inf_mask).squeeze()
                n_inf = removal_positions.shape[0]
                n_remove = min(max_inf_vif_columns_to_remove,
                               trimmed_data.shape[1] - min_columns)

                if n_remove <= 0:
                    break
                
                removal_positions = removal_positions[:n_remove]
                removed_columns = trimmed_data.columns[removal_positions].to_list()
                trimmed_data = trimmed_data.drop(columns=removed_columns)

                for col in removed_columns:
                    print(col, "was removed with infinite VIF")

                if n_inf > max_inf_vif_columns_to_remove:
                    # Compute next iteration of VIF scores.
                    vif_scores = calculate_vif(trimmed_data)
                    continue
       
        removal_position = int(np.argmax(comparable_scores))
        maximum_vif = comparable_scores[removal_position]

        if maximum_vif <= threshold:
            break

        column = trimmed_data.columns[removal_position]
        trimmed_data = trimmed_data.drop(columns=[column])

        # Report on column removal and its VIF.
        print(column, "was removed with VIF", maximum_vif)

        # Compute next iteration of VIF scores.
        vif_scores = calculate_vif(trimmed_data)

    return trimmed_data, vif_scores
