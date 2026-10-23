import warnings
import numpy as np
import inspect

from abc import ABC
from typing import Literal
from sklearn.gaussian_process import kernels, GaussianProcessRegressor
from sklearn import linear_model, svm, metrics

from pyspoc.settings import settings
from pyspoc.statistics.base import PairwiseStatistic
from pyspoc._tag import Tag, Type, Signature


AVAILABLE_KERNELS = (
    "RBF",
    "Matern",
    "DotProduct"
    "ExpSineSquared",
    "RationalQuadratic")

AVAILABLE_ERRORS = (
    "mean_squared_error",
    "mean_absolute_error",
    "median_absolute_error",
    "mean_gamma_deviance",
    "mean_poisson_deviance",
    "mean_pinball_loss")

AVAILABLE_REGRESSORS = (
    "LinearRegression",
    "Ridge",
    "RidgeCV",
    "Lasso",
    "LassoCV",
    "ElasticNet",
    "ElasticNetCV",
    "Lars",
    "LarsCV",
    "LassoLars",
    "LassoLarsCV",
    "LassoLarsIC",
    "OrthogonalMatchingPursuit",
    "OrthogonalMatchingPursuitCV",
    "LinearSVR",
    "ARDRegression",
    "BayesianRidge",
    "HuberRegressor",
    "QuantileRegressor",
    "RANSACRegressor",
    "TheilSenRegressor",
    "GammaRegressor",
    "PoissonRegressor",
    "InverseGaussianRegressor")

__all__ = ["LinearRegressorError",
           "GPRegressorError"]


class LinearRegressor(PairwiseStatistic, ABC):

    _tags = (Tag.ASSOCIATION, Tag.PARAMETRIC,)

    def __init__(
            self,
            model: str,
            force_positive_tweedie: bool,
            dim : Literal["n", "p"],
            nan_policy: Literal[
                "raise", "row-omit", "col-omit"]
            ):

        super().__init__(
            dim=dim,
            nan_policy=nan_policy,
            is_ordered=False)
        
        self._identifier += f".{model}"
        self._model_type = model
        self._force_positive_tweedie = force_positive_tweedie
        self._model_params = {}

        if model == "LinearSVR":
            self._model = svm.LinearSVR
        elif model == "InverseGaussianRegressor":
            self._model = linear_model.TweedieRegressor
            self._model_params["power"] = 3
        else:
            self._model = getattr(linear_model, model)

        model_params = inspect.signature(self._model).parameters

        if "random_state" in model_params:
            self._model_params["random_state"] = (
                settings.current.random_seed
            )

    def _internal_compute(
            self,
            x: np.ndarray,
            y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

        if (self._model_type in ("GammaRegressor",
                                 "PoissonRegressor")
            and self._force_positive_tweedie):

            feature_translation = min(x.min(), y.min())

            if feature_translation < 0:
                x += 1 - feature_translation
                y += 1 - feature_translation

        x_2d = x.reshape(-1, 1) if x.ndim == 1 else x

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_raveled = np.ravel(y)
            model = self._model(**self._model_params).fit(x_2d, y_raveled)

        y_predict = model.predict(x_2d)
        return y_predict, y_raveled
    

class LinearRegressorError(LinearRegressor):

    _long_name = "Linear Regressor Error"
    _identifier = "lr-error"
    _types = (Type.ERROR,)
    _signatures = (Signature.NOISE,)

    def __init__(
            self,
            model: Literal[
                "LinearRegression",
                "Ridge",
                "RidgeCV",
                "Lasso",
                "LassoCV",
                "ElasticNet",
                "ElasticNetCV",
                "Lars",
                "LarsCV",
                "LassoLars",
                "LassoLarsCV",
                "LassoLarsIC",
                "OrthogonalMatchingPursuit",
                "OrthogonalMatchingPursuitCV",
                "LinearSVR",
                "ARDRegression",
                "BayesianRidge",
                "HuberRegressor",
                "QuantileRegressor",
                "RANSACRegressor",
                "TheilSenRegressor",
                "GammaRegressor",
                "PoissonRegressor",
                "InverseGaussianRegressor"] = "LinearRegression",
            error: Literal[
                "mean_squared_error",
                "mean_absolute_error",
                "median_absolute_error",
                "mean_gamma_deviance",
                "mean_poisson_deviance",
                "mean_pinball_loss"] = "mean_squared_error",
            force_positive_tweedie: bool = False,
            dim: Literal["n", "p"] = "p",
            nan_policy: Literal[
                "raise", "row-omit", "col-omit"] = settings.current.data_nan_policy
            ):

        super().__init__(
            model=model,
            force_positive_tweedie=force_positive_tweedie,
            dim=dim,
            nan_policy=nan_policy)
        
        self._error = error
        self._error_func = getattr(metrics, error)

    @staticmethod
    def available_estimators() -> tuple[str, ...]:
        return AVAILABLE_REGRESSORS

    @staticmethod
    def available_errors() -> tuple[str, ...]:
        return AVAILABLE_ERRORS
    
    def _pairwise_compute(self, x: np.ndarray, y: np.ndarray
    ) -> np.ndarray | float:
        y_predict, y_raveled = self._internal_compute(x,y)

        if (self._error in ("mean_gamma_deviance",
                            "mean_poisson_deviance")
            and self._force_positive_tweedie):

            # Push values into positive range for Gamma and Poisson deviance.
            y_translation = min(y_predict.min(), y_raveled.min())

            if y_translation < 0:
                y_predict += 1 - y_translation
                y_raveled += 1 - y_translation

        return float(self._error_func(y_predict, y_raveled))
    

class GPRegressor(PairwiseStatistic, ABC):

    _tags = (Tag.PARAMETRIC, Tag.ASSOCIATION, Tag.NONLINEAR, Tag.DISTRIBUTION, Tag.GAUSSIAN)

    def __init__(
            self,
            kernel: str,
            dim: Literal["n", "p"],
            nan_policy: Literal[
                "raise", "row-omit", "col-omit"]):
        
        self._identifier += f"_{kernel}"
        self._kernel = kernels.ConstantKernel() + kernels.WhiteKernel()
        self._kernel += getattr(kernels, kernel)()
        super().__init__(
            dim=dim,
            nan_policy=nan_policy,
            is_ordered=False)


    def _internal_compute(
            self,
            x: np.ndarray,
            y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:

        x_2d = x.reshape(-1, 1) if x.ndim == 1 else x
        y_raveled = np.ravel(y)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp = GaussianProcessRegressor(kernel=self._kernel).fit(x_2d, y_raveled)

        y_predict = gp.predict(x_2d)
        return y_predict, y_raveled # pyright: ignore[reportReturnType]


class GPRegressorError(GPRegressor):

    _long_name = "Gaussian Process Classifier Error"
    _identifier = "gpc-error"
    _types = (Type.ERROR,)
    _signatures = (Signature.NOISE,)
    
    def __init__(
            self,
            kernel: Literal[
                "RBF",
                "Matern",
                "DotProduct"
                "ExpSineSquared",
                "RationalQuadratic"] = "RBF",
            error: Literal[
                "mean_squared_error",
                "mean_absolute_error",
                "median_absolute_error",
                "mean_gamma_deviance",
                "mean_poisson_deviance",
                "mean_pinball_loss"] = "mean_squared_error",
            force_positive_tweedie: bool = False,
            dim: Literal["n", "p"] = "p",
            nan_policy: Literal[
                "raise", "row-omit", "col-omit"] = settings.current.data_nan_policy
            ):

        super().__init__(
            kernel=kernel,
            dim=dim,
            nan_policy=nan_policy)
        
        self._error = error
        self._error_func = getattr(metrics, error)
        self._force_positive_tweedie = force_positive_tweedie


    @staticmethod
    def available_kernels() -> tuple[str, ...]:
        return AVAILABLE_KERNELS

    @staticmethod
    def available_errors() -> tuple[str, ...]:
        return AVAILABLE_ERRORS

    def _pairwise_compute(self, x: np.ndarray, y: np.ndarray):
        y_predict, y_raveled = self._internal_compute(x,y)

        if (self._error in ("mean_gamma_deviance",
                            "mean_poisson_deviance")
            and self._force_positive_tweedie):

            # Push values into positive range for Gamma and Poisson deviance.
            y_translation = max(y_predict.max(), y_raveled.max())
            y_predict += y_translation
            y_raveled += y_translation

        return self._error_func(y_predict, y_raveled)
