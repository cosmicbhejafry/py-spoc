# %%
from __future__ import annotations

import numpy as np
from functools import cache
from typing import TYPE_CHECKING
from types import FunctionType

import pyspoc._numba as h


def _cache_compute(compute_func: FunctionType):
    
    if compute_func.

    def hasher(*args, **kwargs):
        


        new_args = list()
        new_kwargs = dict()
        np_arg_idxs = set()
        np_kwarg_names = set()
        
        for i, arg in enumerate(args):

            if isinstance(arg, np.ndarray):
                arg = h._get_unordered_array_key(arg)
                np_arg_idxs.add(i)
            
            new_args.append(arg)

        for arg_name, arg in kwargs.items():

            if isinstance(arg, np.ndarray):
                arg = h._get_unordered_array_key(arg)
                np_kwarg_names.add(arg_name)

            new_kwargs[arg_name] = arg
        
        
        return unhasher(*new_args, **new_kwargs)
    return hasher


@_cache_compute
def _tester(x: np.ndarray) -> np.float64:
    return x.mean()
# %%
test_arr = np.array([1,2,3])
test_arr_2 = np.array([1,2,30])
print(_tester(test_arr),
    _tester(test_arr_2))
# %%
