"""Shared parameter handling and estimator resolution for KMeans Statistics."""

from __future__ import annotations

import logging
import numpy as np

from typing import Literal
from abc import ABC

from pyspoc import _argchecking
from pyspoc._argchecking import RuntimeTypeCheckedMixin
from pyspoc._random import RandomSeedMixin
from ._estimator import KClusteringEstimator


_LOGGER = logging.getLogger(__name__)


class KClusteringMixin(RuntimeTypeCheckedMixin, RandomSeedMixin, ABC):
    """Provide common initialization and estimator access for KMeans Statistics.

    The mixin validates dataset-independent configuration at construction time.
    Dataset-dependent limits are deferred to :meth:`_resolve_parameters`.

    Notes
    -----
    This class participates in cooperative multiple inheritance. Its
    initializer therefore calls ``super().__init__()`` after storing its own
    state.
    """

    def __init__(
        self,
        k: int,
        initializer: Literal["k-means++", "random"] = "k-means++",
        max_iter: int = 300,
        random_seed: int | None = None,
    ):
        """Initialize shared KMeans configuration.

        Parameters
        ----------
        k : int
            Number of clusters to form.
        initializer : {"k-means++", "random"}, default="k-means++"
            Strategy used to initialize cluster centroids.
        max_iter : int, default=300
            Maximum number of K-Means iterations for a single run.
        random_seed : int or None, optional
            Per-estimator random-seed override. If ``None``, the active
            library-wide seed is used.

        Raises
        ------
        TypeError
            If a runtime-checked argument has an incompatible type.
        ValueError
            If a numeric argument or component is outside its accepted range,
            or if ``components`` is empty.
        """

        # Validate configuration that does not depend on the eventual dataset.
        self._k = _argchecking.check_natural_number(
            k,
            "k",
        )

        self._initializer = initializer
        self._max_iter = _argchecking.check_natural_number(max_iter, "max_iter")

        # Continue the cooperative initializer chain so random-seed and other
        # inherited mixins can initialize their state.
        super().__init__()

    def _compute_estimator_output(self, data: np.ndarray) -> KClusteringEstimator:
        """Resolve, fit, and retain the estimator for a Statistic computation.

        Parameters
        ----------
        data : numpy.ndarray
            Dataset used both as the estimator cache identity and training
            input.

        Returns
        -------
        KMeansEstimator
            Fitted cached estimator. The returned object may be shared by
            multiple Statistics and should therefore be treated as read-only.
        """
        # Constructor arguments and data identify a reusable cached estimator.
        self._estimator_ = KClusteringEstimator.get_or_create(
            data=data,
            k=self._k,
            initializer=self._initializer,
            max_iter=self._max_iter,
            random_seed=self.random_seed,
        )

        # fit() is lazy and synchronized: an already-fitted cached estimator
        # returns without repeating the training operation.
        self._estimator_.fit(data)
        return self._estimator_
