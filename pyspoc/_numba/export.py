import inspect

from types import FunctionType, ModuleType
from numba.core.registry import CPUDispatcher


def export_func(
    module: ModuleType, func: FunctionType | CPUDispatcher, name: str | None = None
) -> None:

    if not inspect.ismodule(module):
        raise ValueError(f"module must be a Python module, got {type(module)}.")

    if not isinstance(func, CPUDispatcher) and not inspect.isfunction(func):
        raise ValueError(
            f"func must be either a Python function or Numba CPUDispatcherobject, got {type(func)}."
        )

    func_name = func.__name__ if not name else name
    module.__dict__[func_name] = func


def export_module_funcs(module: ModuleType) -> None:

    for obj_name, obj in vars(module):
        if inspect.isfunction(obj) or isinstance(obj, CPUDispatcher):
            export_func(module, obj)
