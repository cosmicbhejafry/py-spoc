import numpy as np

from typing import Literal
from sklearn.cluster import KMeans

from pyspoc._argchecking import RuntimeTypeCheckedMixin
from pyspoc._estimators import LazyFittedCachedEstimatorMixin
from pyspoc._random import RandomSeedMixin
from pyspoc.settings import settings


class KMeansEstimator(
    RuntimeTypeCheckedMixin,
    RandomSeedMixin,
    LazyFittedCachedEstimatorMixin):

    _freeze_random_seed = True

    def __init__(
            self,
            k: int,
            initializer: Literal["k-means++", "random"] = "k-means++",
            max_iter: int = 300,
            random_seed: int | None = None):

        self._k = k
        self._initializer = initializer
        self._max_iter = max_iter

        self._model_ = None

    def _get_model(self) -> KMeans:
        if self._model_ is None:
            raise ValueError("Internal model has not yet been trained. Call fit() first.")

        return self._model_

    def _fit_estimator(self, data: np.ndarray) -> None:
        kmeans = KMeans(
            n_clusters=self._k,
            init=self._initializer,
            max_iter=self._max_iter,
            random_state=self.random_seed,
            copy_x=True,
            verbose=int(settings.current.verbose))

        self._model_ = kmeans.fit(data)
