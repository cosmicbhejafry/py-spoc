import numpy as np

from typing import Literal
from sklearn.feature_selection import mutual_info_regression

import pyspoc._argchecking as argchk
from pyspoc.settings import settings
from pyspoc.statistics.base import Statistic



class MutualInformation(Statistic):
    """Construct a nonlinear-dependence estimate using mutual information."""

    _name = "Mutual Information"
    _identifier = "mutual-information"
    _labels = ("feature-wise", "information", "nonlinear", "undirected")


    def __init__(
        self,
        n_neighbors: int = 3,
        threshold: float | None = None,
        random_seed: int | None = None,
        dim: Literal["n", "p"] = "p",
        nan_policy: Literal[
            "raise", "row-omit", "col-omit"] = settings.current.data_nan_policy):

        argchk.check_natural_number(n_neighbors, arg_name="n_neighbors")
        self._n_neighbors = n_neighbors
        self._threshold = (
            argchk.check_float(threshold, minimum=0, arg_name="threshold")
            if threshold is not None
            else None)
        
        super().__init__(
            dim=dim,
            nan_policy=nan_policy)
        

    def _compute(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]]
    ) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
        """Return pairwise estimated mutual information between feature columns."""

        n = data.shape[0]

        if self._n_neighbors >= n:
            raise ValueError(
                f"n_neighbors ({self._n_neighbors}) must be smaller "
                f"than the number of observations ({n}).")

        result = _mutual_information_matrix(
            data,
            n_neighbors=self._n_neighbors,
            random_seed=self.random_seed)

        return result



def _mutual_information_matrix(
    data: np.ndarray,
    *,
    n_neighbors: int,
    random_seed: int | None) -> np.ndarray:
    """Estimate a non-negative symmetric mutual-information matrix."""

    n_features = data.shape[1]
    result = np.zeros((n_features, n_features), dtype=float)

    for i in range(n_features):
        for j in range(i + 1, n_features):
            # sklearn exposes MI through a regression-oriented API. Averaging
            # both orientations removes finite-sample directional variation
            # while preserving the symmetry of theoretical mutual information.
            forward = mutual_info_regression(
                data[:, [i]],
                data[:, j],
                n_neighbors=n_neighbors,
                random_state=random_seed,
            )[0]
            reverse = mutual_info_regression(
                data[:, [j]],
                data[:, i],
                n_neighbors=n_neighbors,
                random_state=random_seed,
            )[0]
            result[i, j] = result[j, i] = max(0.0, (forward + reverse) / 2)

    return result
