import numpy as np
import inspect

from sklearn.gaussian_process import kernels, GaussianProcessRegressor
from sklearn.metrics import mean_squared_error
from sklearn import linear_model

from pyspoc.statistic import PairwiseStatistic
from pyspoc.settings import settings


class LinearModelError(PairwiseStatistic):
    name = "Linear model regression"
    identifier = "lmfit"
    labels = ["misc", "unsigned", "unordered", "normal", "linear", "directed"]

    def __init__(self, model):
        self.identifier += f".{model}"
        self._model = getattr(linear_model, model)
        super().__init__(dim="p", is_ordered=False)

    def pairwise_compute(self,
                         x: np.ndarray,
                         y: np.ndarray):

        y_raveled = np.ravel(y)
        model_params = inspect.signature(self._model).parameters
        x_2d = x.reshape(-1, 1) if x.ndim == 1 else x
        
        if "random_state" in model_params:
            mdl = self._model(random_state=settings.current.random_seed).fit(x_2d, y_raveled)
        else:
            mdl = self._model().fit(x_2d, y_raveled)

        y_predict = mdl.predict(x_2d)
        return mean_squared_error(y_predict, y_raveled)


class GPModelError(PairwiseStatistic):
    name = "Gaussian process regression"
    identifier = "gpfit"
    labels = ["misc", "unsigned", "unordered", "normal", "nonlinear", "directed"]

    def __init__(self, kernel="RBF"):
        self.identifier += f"_{kernel}"
        self._kernel = kernels.ConstantKernel() + kernels.WhiteKernel()
        self._kernel += getattr(kernels, kernel)()
        super().__init__(dim="p", is_ordered=False)

    def pairwise_compute(self,
                         x: np.ndarray,
                         y: np.ndarray):

        x_2d = x.reshape(-1, 1) if x.ndim == 1 else x
        y_raveled = np.ravel(y)
        gp = GaussianProcessRegressor(kernel=self._kernel).fit(x_2d, y_raveled)
        y_predict = gp.predict(x_2d)
        return mean_squared_error(y_predict, y_raveled)
