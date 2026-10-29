import warnings
import numpy as np
import inspect

from abc import ABC
from typing import Literal
from sklearn.gaussian_process import kernels, GaussianProcessClassifier
from sklearn import linear_model, svm, metrics

from pyspoc.settings import settings
from pyspoc.component.statistics.base import PairwiseStatistic
from pyspoc._tag import Tag, Type, Signature


__all__ = [
    "LinearClassifierError",
    "GPClassifierError"
]

class LinearClassifier(PairwiseStatistic, ABC):
        
    _tags = (Tag.PARAMETRIC, Tag.ASSOCIATION)

    def __init__(
            self,
            model: str,
            dim: Literal["n", "p"] = "p",
            nan_policy: Literal[
                "raise", "row-omit", "col-omit"] = settings.current.data_nan_policy):

        super().__init__(
            dim=dim,
            nan_policy=nan_policy,
            is_ordered=False)
        
        self._identifier += f".{model}"
        self._model_type = model
        self._model_params = {}

        if model == "LinearSVC":
            self._model = getattr(svm, model)

        else:
            internal_model_type = (
                "Lasso" if model == "LassoClassifier" else
                "LassoCV" if model == "LassoClassifierCV" else
                "ElasticNet" if model == "ElasticNetClassifier" else
                "ElasticNetCV" if model == "ElasticNetClassifierCV" else
                model)

            self._bipolar_transform = (model in (
                "LassoClassifier",
                "LassoClassifierCV",
                "ElasticNetClassifier",
                "ElasticNetClassifierCV"
            ))

            self._model = getattr(linear_model, internal_model_type)

        model_params = inspect.signature(self._model).parameters

        if "random_state" in model_params:
            self._model_params["random_state"] = (
                settings.current.random_seed
            )
    

    def _internal_compute(self,
                         x: np.ndarray,
                         y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:

        # LassoClassifier and ElasticNetClassifier convert target
        # values to {-1, 1} and treats the problem as a regression
        # task, using a Lasso regressor. This matches the SKLearn
        # approach to RidgeClassifier:
        # (https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html#sklearn.linear_model.RidgeClassifier)
        x_2d = x.reshape(-1, 1) if x.ndim == 1 else x

        if self._bipolar_transform:
            y = 2 * (y - y.min()) / (y.max() - y.min()) - 1

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_raveled = np.ravel(y)
            model = self._model(**self._model_params).fit(x_2d, y_raveled)

        y_predict = model.predict(x_2d)
        return y_predict, y_raveled
    


class LinearClassifierError(LinearClassifier):

    _long_name = "Linear Classifier Error"
    _identifier = "lc-error"
    _types = (Type.ERROR,)
    _signatures = (Signature.NOISE,)

    def __init__(
            self,
            model: Literal[
                "LogisticRegression",
                "LogisticRegressionCV",
                "Perceptron",
                "LassoClassifier",
                "LassoClassifierCV",
                "RidgeClassifier",
                "RidgeClassifierCV",
                "ElasticNetClassifier",
                "ElasticNetClassifierCV",
                "LinearSVC"] = "LogisticRegression",
            error: Literal[
                "log_loss",
                "mean_squared_error",
                "mean_absolute_error",
                "median_absolute_error",
                "mean_gamma_deviance",
                "mean_poisson_deviance",
                "mean_pinball_loss"] = "log_loss"):

        super().__init__(model)
        self._error_func = getattr(metrics, error)

    @staticmethod
    def available_estimators() -> tuple[str, ...]:
        return (
            "LogisticRegression",
            "LogisticRegressionCV",
            "Perceptron",
            "LassoClassifier",
            "LassoClassifierCV",
            "RidgeClassifier",
            "RidgeClassifierCV",
            "ElasticNetClassifier",
            "ElasticNetClassifierCV",
            "LinearSVC")

    @staticmethod
    def available_errors() -> tuple[str, ...]:
        return (
            "log_loss",
            "mean_squared_error",
            "mean_absolute_error",
            "median_absolute_error",
            "mean_gamma_deviance",
            "mean_poisson_deviance",
            "mean_pinball_loss")

    def _pairwise_compute(self, x: np.ndarray, y: np.ndarray
    ) -> np.ndarray | float:
        y_predict, y_raveled = self._internal_compute(x,y)
        return float(self._error_func(y_predict, y_raveled))



class GPClassifier(PairwiseStatistic):

    _tags = (Tag.PARAMETRIC, Tag.ASSOCIATION, Tag.NONLINEAR, Tag.DISTRIBUTION, Tag.GAUSSIAN)

    def __init__(
            self,
            kernel: str,
            dim: Literal["n", "p"] = "p",
            nan_policy: Literal[
                "raise", "row-omit", "col-omit"] = settings.current.data_nan_policy):

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
            gp = GaussianProcessClassifier(kernel=self._kernel).fit(x_2d, y_raveled)

        y_predict = gp.predict(x_2d)
        return y_predict, y_raveled # pyright: ignore[reportReturnType]



class GPClassifierError(GPClassifier):

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
                "log_loss",
                "mean_squared_error",
                "mean_absolute_error",
                "median_absolute_error",
                "mean_gamma_deviance",
                "mean_poisson_deviance",
                "mean_pinball_loss"] = "log_loss"):

        super().__init__(kernel)
        self._error_func = getattr(metrics, error)

    @staticmethod
    def available_kernels() -> tuple[str, ...]:
        return (
            "RBF",
            "Matern",
            "DotProduct"
            "ExpSineSquared",
            "RationalQuadratic")

    @staticmethod
    def available_errors() -> tuple[str, ...]:
        return (
            "log_loss",
            "mean_squared_error",
            "mean_absolute_error",
            "median_absolute_error",
            "mean_gamma_deviance",
            "mean_poisson_deviance",
            "mean_pinball_loss")

    def _pairwise_compute(self, x: np.ndarray, y: np.ndarray):
        y_predict, y_raveled = self._internal_compute(x,y)
        return self._error_func(y_predict, y_raveled)
