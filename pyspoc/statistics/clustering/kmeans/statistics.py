"""Concrete Statistics derived from fitted K-Means cluster centres."""

import numpy as np

from typing import Literal
from sklearn.metrics.pairwise import (
    laplacian_kernel,
    rbf_kernel,
    cosine_similarity,
)
from scipy.spatial.distance import euclidean

from pyspoc._argchecking import check_integer_bounds, check_float
from pyspoc._utils import numerical as pn
from ._estimator import KMeansEstimator
from ._base import KMeansStatistic


class KMeansClusterSimilarity(KMeansStatistic):
    """Calculate pairwise similarities between fitted K-Means centres.

    Each row and column of the result represents one fitted cluster. The
    resulting matrix is symmetric, has a unit diagonal, and contains
    similarities in the interval ``[0, 1]``.

    Parameters
    ----------
    k : int
        Number of clusters to form. Must be at least two.
    initializer : {"k-means++", "random"}, default="k-means++"
        Strategy used by scikit-learn to initialize cluster centres.
    max_iter : int, default=300
        Maximum number of K-Means iterations for a single initialization.
    kernel : {"rbf", "mahalanobis_rbf", "laplacian", \
              "inverse_distance", "mahalanobis_inverse_distance", \
              "cosine", "correlation"}, default="rbf"
        Similarity measure applied to fitted cluster centres. Mahalanobis
        variants first whiten centres using the pseudo-inverse square root of
        the empirical feature covariance matrix.
    gamma : float or None, optional
        Strictly positive similarity decay coefficient. If ``None``, use
        ``1 / n_features`` for the dataset supplied to :meth:`compute`.
    random_seed : int or None, optional
        Per-Statistic random-seed override. If ``None``, use the active
        library-wide random seed.

    Notes
    -----
    RBF, Laplacian, and inverse-distance similarities measure spatial
    proximity. Cosine and correlation similarities instead compare the
    direction or feature profile of centres.

    Singular covariance matrices are supported by discarding covariance
    eigenvalues below a numerical tolerance. Consequently, Mahalanobis
    similarities ignore directions containing no independently measurable
    variance.
    """

    _name = "KMeans - Cluster Similarity"
    _identifier = "kmeans-cs"
    _labels = ["non-linear"]

    def __init__(
        self,
        k: int,
        initializer: Literal["k-means++"] | Literal["random"] = "k-means++",
        max_iter: int = 300,
        kernel: Literal[
            "rbf",
            "mahalanobis_rbf",
            "laplacian",
            "inverse_distance",
            "mahalanobis_inverse_distance",
            "cosine",
            "correlation",
        ] = "rbf",
        gamma: float | None = None,
        random_seed: int | None = None,
    ):
        """Initialize cluster-fitting and similarity configuration.

        Parameters
        ----------
        k : int
            Number of clusters to form. Must be at least two.
        initializer : {"k-means++", "random"}, default="k-means++"
            Strategy used by scikit-learn to initialize cluster centres.
        max_iter : int, default=300
            Maximum number of K-Means iterations for a single run.
        kernel : {"rbf", "mahalanobis_rbf", "laplacian", \
                  "inverse_distance", "mahalanobis_inverse_distance", \
                  "cosine", "correlation"}, default="rbf"
            Similarity measure applied to fitted cluster centres.
        gamma : float or None, optional
            Strictly positive similarity decay coefficient. If ``None``, use
            ``1 / n_features`` when computation begins. Gamma is ignored by
            cosine and correlation similarities.
        random_seed : int or None, optional
            Per-Statistic random-seed override. If ``None``, the active
            library-wide seed is used.

        Raises
        ------
        TypeError
            If a runtime-checked argument has an incompatible type.
        ValueError
            If ``k`` is less than two or ``gamma`` is not strictly positive.
        """

        # Similarity configuration does not participate in estimator cache
        # identity: Statistics using different kernels can reuse the same
        # fitted K-Means estimator.
        self._kernel = kernel
        self._gamma = (
            check_float(gamma, exclusive_minimum=0, arg_name="gamma")
            if gamma is not None
            else gamma
        )

        # A pairwise cluster matrix is only meaningful for at least two
        # clusters. The shared mixin performs the remaining K-Means checks.
        check_integer_bounds(k, 2, arg_name="k")
        super().__init__(k, initializer, max_iter, random_seed)

    @property
    def name(self) -> str:
        """Return the human-readable Statistic name.

        Returns
        -------
        str
            Display name used by library configuration and reporting.
        """
        return self._name

    @property
    def labels(self) -> tuple[str, ...]:
        """Return immutable labels describing the Statistic.

        Returns
        -------
        tuple of str
            Labels used to categorize the Statistic.
        """
        return tuple(self._labels)

    @property
    def identifier(self) -> str:
        """Return the stable configuration identifier.

        Returns
        -------
        str
            Short identifier used to select this Statistic.
        """
        return self._identifier

    # Override the abstract extraction hook inherited from KMeansStatistic.
    # The base class has already resolved and fitted the shared estimator.
    def _get_result(self, data: np.ndarray, fitted_estimator: KMeansEstimator) -> np.ndarray:
        """Construct a pairwise similarity matrix from a fitted estimator.

        Parameters
        ----------
        data : numpy.ndarray
            Original two-dimensional observations. Mahalanobis kernels use
            these data to estimate feature covariance.
        fitted_estimator : KMeansEstimator
            Shared estimator fitted to ``data``. Its underlying model is
            borrowed as read-only state.

        Returns
        -------
        numpy.ndarray
            Symmetric ``(k, k)`` cluster-similarity matrix with unit diagonal.

        Notes
        -----
        This method overrides the abstract
        :meth:`KMeansStatistic._get_result` hook.
        """

        # Borrow fitted centres without exposing or mutating the cached model.
        model = fitted_estimator._get_model()
        centres = model.cluster_centers_
        gamma = self._gamma if self._gamma is not None else 1 / data.shape[1]

        # Scikit-learn provides vectorized implementations for kernels that
        # operate directly on the complete centre matrix.
        if self._kernel in ("rbf", "laplacian", "cosine"):
            return self._compute_sklearn_kernel(centres, self._kernel, gamma=gamma)

        # Mahalanobis distance is Euclidean distance after whitening. The
        # pseudo-inverse square root supports rank-deficient feature
        # covariance by discarding its numerical null space.
        if self._kernel in ("mahalanobis_rbf", "mahalanobis_inverse_distance"):
            covariance = np.atleast_2d(np.cov(data, rowvar=False))
            covariance_inv_sq = pn.spsd_matrix_power(covariance, -0.5)
            centres = centres @ covariance_inv_sq

        # RBF remains vectorizable after the Mahalanobis transformation.
        if self._kernel == "mahalanobis_rbf":
            return self._compute_sklearn_kernel(centres, self._kernel, gamma=gamma)

        # Manual measures are evaluated once per unordered pair. Initialize
        # the diagonal to one because every centre is maximally similar to
        # itself, including centres for which correlation is undefined.
        n_centres = centres.shape[0]
        similarities = np.ones(shape=(n_centres, n_centres))

        for i in range(n_centres):
            for j in range(i + 1, n_centres):
                similarity = self._compute_manual_kernel(
                    centres[i],
                    centres[j],
                    kernel=self._kernel,
                    gamma=gamma,
                )
                # Fill both locations immediately to preserve exact symmetry.
                similarities[i, j] = similarity
                similarities[j, i] = similarity

        return similarities

    def _compute_sklearn_kernel(
        self,
        centres: np.ndarray,
        kernel: Literal["rbf", "mahalanobis_rbf", "laplacian", "cosine"],
        gamma: float,
    ) -> np.ndarray:
        """Evaluate a vectorized scikit-learn similarity kernel.

        Parameters
        ----------
        centres : numpy.ndarray
            Matrix shaped ``(k, n_features)`` containing one centre per row.
        kernel : {"rbf", "mahalanobis_rbf", "laplacian", "cosine"}
            Kernel implementation to evaluate. ``"mahalanobis_rbf"`` denotes
            RBF applied to centres that were whitened by the caller.
        gamma : float
            Strictly positive decay coefficient. Cosine similarity ignores
            this value.

        Returns
        -------
        numpy.ndarray
            Symmetric ``(k, k)`` similarity matrix.

        Raises
        ------
        ValueError
            If ``kernel`` is unsupported by this helper.
        """

        match kernel:
            case "rbf" | "mahalanobis_rbf":
                # RBF produces exp(-gamma * squared Euclidean distance).
                return rbf_kernel(centres, gamma=gamma)

            case "laplacian":
                # Laplacian similarity uses L1/Manhattan distance.
                return laplacian_kernel(centres, gamma=gamma)

            case "cosine":
                # Map cosine's natural [-1, 1] interval onto [0, 1].
                return (1 + cosine_similarity(centres)) / 2

            case _:
                raise ValueError(f"Unsupported kernel {kernel!r} for sklearn kernel computation.")

    def _compute_manual_kernel(
        self,
        x: np.ndarray,
        y: np.ndarray,
        kernel: Literal["inverse_distance", "mahalanobis_inverse_distance", "correlation"],
        gamma: float,
    ) -> float:
        """Evaluate a scalar similarity between two cluster centres.

        Parameters
        ----------
        x, y : numpy.ndarray
            One-dimensional cluster-centre vectors.
        kernel : {"inverse_distance", "mahalanobis_inverse_distance", \
                  "correlation"}
            Pairwise similarity to evaluate. Mahalanobis inverse distance
            receives centres already whitened by the caller.
        gamma : float
            Strictly positive inverse-distance decay coefficient. Correlation
            similarity ignores this value.

        Returns
        -------
        float
            Similarity in the closed interval ``[0, 1]``.

        Raises
        ------
        ValueError
            If ``kernel`` is unsupported by this helper.

        Notes
        -----
        Correlation is undefined when either centred vector has zero norm. For
        this degenerate case, identical vectors receive similarity ``1`` and
        distinct vectors receive neutral similarity ``0.5``.
        """

        match kernel:
            case "inverse_distance" | "mahalanobis_inverse_distance":
                # Distance zero maps to one and similarity decays
                # asymptotically towards zero.
                return float(1 / (1 + gamma * euclidean(x, y)))

            case "correlation":
                if np.array_equal(x, y):
                    return 1.0

                # Pearson correlation is cosine similarity after independently
                # centring each vector. A constant centre becomes the zero
                # vector and therefore has no defined correlation direction.
                x_centred = x - np.mean(x)
                y_centred = y - np.mean(y)
                x_norm = np.linalg.norm(x_centred)
                y_norm = np.linalg.norm(y_centred)

                if np.isclose(x_norm, 0.0) or np.isclose(y_norm, 0.0):
                    return 0.5

                coefficient = np.dot(x_centred, y_centred) / (x_norm * y_norm)
                # Floating-point roundoff can place a theoretical endpoint
                # infinitesimally outside [-1, 1].
                coefficient = np.clip(coefficient, -1.0, 1.0)
                return float((1 + coefficient) / 2)

            case _:
                raise ValueError(f"Unsupported kernel {kernel!r} for manual kernel computation.")
