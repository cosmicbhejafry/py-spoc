"""Cached, lazily fitted estimator for scikit-learn K-Means models."""

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
    LazyFittedCachedEstimatorMixin,
):
    """Manage a cached scikit-learn K-Means model.

    The estimator combines runtime constructor checking, library-wide random
    seed resolution, estimator caching, and synchronized lazy fitting. A fitted
    estimator may be shared by multiple Statistics requesting equivalent
    clustering configuration for equivalent data.

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

    Notes
    -----
    The underlying :class:`sklearn.cluster.KMeans` object remains private so
    cache consumers cannot accidentally replace it through a public property.
    """

    # Include the resolved random seed in cache identity. This prevents an
    # estimator created under one settings context from being reused after the
    # effective library-wide seed changes.
    _freeze_random_seed = True

    def __init__(
        self,
        k: int,
        initializer: Literal["k-means++", "random"] = "k-means++",
        max_iter: int = 300,
        random_seed: int | None = None,
    ):
        """Initialize K-Means configuration and unfitted state.

        Parameters
        ----------
        k : int
            Number of clusters to form.
        initializer : {"k-means++", "random"}, default="k-means++"
            Centroid initialization strategy passed to scikit-learn.
        max_iter : int, default=300
            Maximum number of iterations for a single K-Means run.
        random_seed : int or None, optional
            Per-estimator random-seed override. If ``None``, use the active
            library-wide seed.

        Notes
        -----
        Constructor arguments are recorded automatically by the inherited
        estimator machinery and subsequently participate in cache matching.
        """

        # Retain constructor configuration unchanged so it can be passed to the
        # scikit-learn implementation when lazy fitting is first requested.
        self._k = k
        self._initializer = initializer
        self._max_iter = max_iter

        # The model is created only inside the synchronized fitting hook.
        self._model_ = None

    def _get_model(self) -> KMeans:
        """Return the fitted private K-Means model.

        Returns
        -------
        sklearn.cluster.KMeans
            Fitted scikit-learn model owned by this estimator.

        Raises
        ------
        ValueError
            If :meth:`fit` has not yet completed successfully.

        Notes
        -----
        The returned model may underpin multiple cached Statistic objects and
        should therefore be treated as read-only.
        """
        if self._model_ is None:
            raise ValueError("Internal model has not yet been trained. Call fit() first.")

        return self._model_

    # Override the abstract fitting hook inherited from LazyFittedCachedEstimatorMixin.
    # Its public fit() method handles thread-safe locking, data equivalence,
    # and the fitted-state transition around this operation.
    def _fit_estimator(self, data: np.ndarray):
        """Construct and fit the private K-Means model.

        Parameters
        ----------
        data : numpy.ndarray
            Two-dimensional observations accepted by the estimator fitting
            lifecycle.

        Returns
        -------
        None
            The fitted scikit-learn model is stored on this estimator.

        Notes
        -----
        This method overrides the abstract
        :meth:`LazyFittedCachedEstimatorMixin._fit_estimator` hook. Call
        :meth:`fit` rather than invoking this method directly so cache and
        thread-safety invariants remain enforced.
        """
        # Translate the library's boolean verbosity setting to scikit-learn's
        # integer verbosity parameter. False becomes 0 and True becomes 1.
        kmeans = KMeans(
            n_clusters=self._k,
            init=self._initializer,
            max_iter=self._max_iter,
            random_state=self.random_seed,
            copy_x=True,
            verbose=int(settings.current.verbose),
        )

        # scikit-learn returns the fitted estimator from fit(); retain that
        # object as the private model published by the lazy fitting lifecycle.
        self._model_ = kmeans.fit(data)
