import numpy as np

from tqdm import tqdm
from scipy import stats
from pyspoc import Statistic
from typing import Optional

from .func import box_counting, box_counting_generalized, get_default_init_scale


class BoxCountingDimension(Statistic):

    """
    Computes a variety of box counting fractal dimension statistics for static datasets (n x p) returning a scalar.
    If a time series (n x p x t) is provided, dynamic fractal dimension will be returned instead as a 1 x t vector.

    Base code is thanks to Cui, T., & Wang, T. (2024). Exact box-counting and temporal sampling algorithms for 
    fractal dimension estimation with applications to animal behavior analysis. Results in Engineering. 
    https://doi.org/10.1016/j.rineng.2024.103755 and can be found at: https://github.com/wanglab-georgetown/fractal

    Parameters
    ----------
    method : str
        The box counting method to use. Options are 'original', 'exact' and 'generalized'. Default is 'original'.
    p_value_threshold : float
        The p-value threshold for significance testing. Default is 0.05.
    q : int, optional
        The order of the generalized dimension. Required if method is 'generalized'.
    
    Returns
    -------
    np.ndarray
        An array containing the fractal dimension(s).
    
    Raises
    ------
    ValueError
        If 'q' is not provided when using the 'generalized' method.
    """

    __name = "BoxCountingDimension"
    __identifier = "boxcount"
    __labels = ["fractal", "linear"]

    def __init__(self,
                 p_value_threshold: float = 0.05):
        
        self.__p_value_thresh = p_value_threshold
        super().__init__()

    @property
    def name(self) -> str:
        return self.__name

    @property
    def identifier(self) -> str:
        return self.__identifier

    @property
    def labels(self) -> list[str]:
        return self.__labels

    def compute(self, data: np.ndarray) -> np.ndarray:
        #scales = self.__get_box_scales(data)
        scales = get_default_init_scale(data)
        nan_array = np.array(np.nan)

        # NOTE: Add warnings for NaN returns.
        
        # Return NaN array if no scales found.
        if scales is None:
            return nan_array
        
        result = self.__get_box_count(data, scales)

        # Return NaN array if no result.
        if not result:
            return nan_array
        
        p_value = result.get("p_value")
        
        # Return NaN array if no result or p-value threshold not met.
        if not p_value:
            return nan_array
        
        if p_value > self.__p_value_thresh:
            return nan_array
        
        # Otherwise, return the fractal dimension.
        return np.array(result["fd"])
    
    def _get_box_count(self, data, scales) -> dict:
        results = []

        for scale in tqdm(scales):
            scaled_boxes = {(int(pt[0] / scale), int(pt[1] / scale)) for pt in data}
            results.append([scale, len(scaled_boxes)])

        results = np.array(results)
        log_scales = -np.log(results[:, 0])
        log_boxes = np.log(results[:, 1])

        slope, _, r_value, p_value, _ = stats.linregress(log_scales, log_boxes)
        fd = slope

        result = {
            "fd": fd,
            "r_squared": r_value**2,
            "p_value": p_value,
        }

        return result
    
    def _get_box_count_vec(self, data, scales) -> dict:
        data = np.expand_dims(data, axis=2)
        scaled_boxes = (data / scales).astype(int)        
        log_scales = -np.log(results[:, 0])
        log_boxes = np.log(results[:, 1])

        slope, _, r_value, p_value, _ = stats.linregress(log_scales, log_boxes)
        fd = slope

        result = {
            "fd": fd,
            "r_squared": r_value**2,
            "p_value": p_value,
        }

        return result