import inspect
import numpy as np
import scipy as sp
import mne_connectivity as mnec

from sklearn import covariance as skcov
from typing import Union

from pyspoc.statistic import Statistic, PairwiseStatistic


class Covariance(Statistic):
    """
    Computes a variety of covariance statistics for static datasets (n x p) returning a p x p matrix.
    If a time series (n x p x t) is provided, dynamic covariance will be returned instead as a p x p x t tensor.
    Information on covariance estimators can be found at: https://scikit-learn.org/stable/modules/covariance.html
    """

    _name = "Covariance"
    _identifier = "cov"
    _labels = ["basic", "unordered", "linear"]

    def __init__(self,
                 estimator: str = "EmpiricalCovariance",
                 squared: bool = False):

        if squared:
            self._labels.append("unsigned")
            self._identifier += ".sq"
        else:
            self._labels.append("signed")

        self.__is_squared = squared
        self.__estimator = estimator
        super().__init__()

    @property
    def name(self) -> str:
        return self._name

    @property
    def identifier(self) -> str:
        return self._identifier

    @property
    def labels(self) -> list[str]:
        return self._labels

    def compute(self, data: np.ndarray) -> np.ndarray | float:
        cov_obj = self.__fit(data)
        cov = cov_obj.covariance_

        if self.__is_squared:
            cov = np.square(cov)

        return cov

    def __fit(self, data: np.ndarray):
        cov_dir = [x for x in skcov.__dir__() if inspect.isclass(getattr(skcov, x))] # pyright: ignore[reportAttributeAccessIssue]

        if self.__estimator not in cov_dir:
            available_estimators = ", ".join(cov_dir)
            raise AttributeError(
                f"The {self.__class__.__name__} estimator {self.__estimator} is not supported.\n"
                f"Options include: {available_estimators}.")

        cov_class = getattr(skcov, self.__estimator)()
        cov_obj = cov_class.fit(data)
        return cov_obj


class Precision(Covariance):

    _name = "Precision"
    _identifier = "prec"

    def __init__(self,
                 estimator: str = "EmpiricalCovariance",
                 squared: bool = False):

        super().__init__(estimator=estimator,
                         squared=squared)

    def compute(self, dataset: np.ndarray) -> np.ndarray:
        cov_obj = self.__fit(dataset)
        cov = cov_obj.precision_

        if self.__is_squared:
            cov = np.square(cov)

        return cov

class SpearmanR(PairwiseStatistic):
    # Setting the name internally.
    _name = "Spearman's correlation coefficient"

    # Setting the identifier internally.
    _identifier = "spearmanr"

    # Setting the labels internally.
    _labels = ["basic", "rank", "linear", "undirected"]

    def __init__(self, squared: bool):

        # Storing the squared argument.
        self.__squared = squared

        # If squared,
        if squared:

            # Add the "unsigned" label.
            self._labels += ["unsigned"]

            # And the ".sq" suffix to the identifier.
            self._identifier += ".sq"

        else:

            # Else, add the "signed" label.
            self._labels += ["signed"]

        # Call the base class initialiser with required arguments.
        super().__init__(dim="p", is_ordered=False, symmetry="yes")

    # Implementing the name property.
    @property
    def name(self) -> str:
        return self._name

    # Implementing the identifier property.
    @property
    def identifier(self) -> str:
        return self._identifier

    # Implementing the labels property.
    @property
    def labels(self) -> list[str]:
        return self._labels

    # Implementing the PairwiseStatistic's pairwise_compute method.
    # Arguments:
    #  - x: A given observation as an n x 1 numpy array OR a given variable as a p x 1 numpy.
    #  - y: A given observation as an n x 1 numpy array OR a given variable as a p x 1 numpy.
    def pairwise_compute(self,
                         x: np.ndarray,
                         y: np.ndarray) -> Union[np.ndarray, float]:

        # Compute the Spearman rank correlation coefficient using Scipy library.
        corr = sp.stats.spearmanr(x, y).correlation

        # Square results if required.
        if self.__squared:
            return corr ** 2

        # Return value.
        return corr


class KendallTau(PairwiseStatistic):

    _name = "Kendall's tau"
    _identifier = "kendalltau"
    _labels = ["basic", "unordered", "rank", "linear", "undirected"]

    def __init__(self, squared: bool, dim: str = "p"):
        self.__squared = squared
        if squared:
            self._identifier += ".sq"
            self._labels += ["unsigned"]
        else:
            self._labels += ["signed"]

        super().__init__(dim="p", is_ordered=False, symmetry="yes")

    def name(self) -> str:
        return self._name

    def identifier(self) -> str:
        return self._identifier

    def labels(self) -> list[str]:
        return self._labels

    def pairwise_compute(self,
                         x: np.ndarray,
                         y: np.ndarray) -> Union[np.ndarray, float]:

        corr = sp.stats.kendalltau(x, y).correlation

        if self.__squared:
            return corr ** 2

        return corr


class PowerEnvelopeCorrelation(Statistic):
    # Setting the name internally.
    __name = "Power Envelope Correlation"

    # Setting the identifier internally.
    __identifier = "pec"

    # Setting the labels internally.
    __labels = ["unsigned", "misc", "undirected"]

    def __init__(self, orth=False, log=False, absolute=False):

        # If the orthogonal argument is provided...
        if orth:
            # Store the orthogonal argument to use in the compute method.
            self.__orth = "pairwise"

            # Update the identifier.
            self.__identifier += "_orth"

        # Otherwise, default to False.
        else:
            self.__orth = False

        # Store the log argument for use in the compute method.
        self.__log = log

        # If set...
        if log:
            # Update the identifier.
            self.__identifier += "_log"

        # Store the absolute argument for use in the compute method.
        self.__absolute = absolute

        # If set...
        if absolute:
            # Update the identifier.
            self.__identifier += "_abs"

        # Call the base class initialiser.
        super().__init__()

    # Implementing the name property.
    @property
    def name(self) -> str:
        return self.__name

    # Implementing the identifier property.
    @property
    def identifier(self) -> str:
        return self.__identifier

    # Implementing the labels property.
    @property
    def labels(self) -> list[str]:
        return self.__labels

    # Compute method requires implementing.
    # data: Full dataset with n observations, p variables as a numpy array.
    def compute(self, data: np.ndarray) -> np.ndarray:

        # Utilizing the envelop_correlation function provided by the MNE-Connectivity library.
        env_corr = mnec.envelope_correlation(

            # Passing through the stored arguments from the initialiser.
            data, orthogonalize=self.__orth, log=self.__log, absolute=self.__absolute
        )

        # Squeezing the result to remove any excess dimensions.
        adj = np.squeeze(env_corr)

        # Filling self/auto-correlations with NaNs.
        np.fill_diagonal(adj, np.nan)

        # Returning the p by p matrix as numpy array.
        return adj