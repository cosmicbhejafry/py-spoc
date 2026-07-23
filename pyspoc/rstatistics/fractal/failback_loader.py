#from __future__ import annotations

import inspect
import importlib.util
import sys


from collections.abc import Iterable, Callable
from functools import wraps
from types import ModuleType
from typing import Literal, ParamSpec, TypeVar, cast
from numba.core.registry import CPUDispatcher

from pyspoc._numba import _export_func


P = ParamSpec("P")
R = TypeVar("R")

DebugNumba = Iterable[Literal["ignore", "warn", "raise", "bounds"]]

# ---------------------------------------------------------------------
# Loader machinery
# ---------------------------------------------------------------------

def _get_module_source_path(module: ModuleType) -> str:
    spec = getattr(module, "__spec__", None)

    if spec is not None and spec.origin not in (None, "built-in", "frozen"):
        return spec.origin

    source_path = getattr(module, "__file__", None)

    if source_path is None:
        raise ValueError(f"Module {module.__name__!r} has no source path.")

    return source_path


def _load_module_copy(
    module: ModuleType,
    *,
    enable_numba: bool) -> ModuleType:
    source_path = _get_module_source_path(module)

    suffix = "__nb" if enable_numba else "__py"
    copy_name = f"{module.__name__}{suffix}"

    spec = importlib.util.spec_from_file_location(copy_name, source_path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {source_path!r}")

    module_copy = importlib.util.module_from_spec(spec)

    module_copy.__dict__["__skip_jit_compile__"] = not enable_numba
    module_copy.__dict__["__internal_copy__"] = True

    sys.modules[copy_name] = module_copy
    spec.loader.exec_module(module_copy)

    return module_copy


def _py_failback(numba_func: CPUDispatcher) -> Callable[[Callable[P, R]], Callable[P, R]]:
    def decorator(py_func: Callable[P, R]) -> Callable[P, R]:
        @wraps(py_func)
        def wrapper(
            *args: P.args,
            **kwargs: P.kwargs) -> R:
            
            debug_numba = kwargs.get("debug_numba", ["ignore"])

            try:
                result = numba_func(*args, **kwargs)

                if "ignore" not in debug_numba:
                    print(f"Numba ran {py_func.__name__} successfully.")

                return cast(R, result)

            except Exception as e:
                if "raise" in debug_numba:
                    raise

                if "warn" in debug_numba:
                    print(
                        f"Numba failed to run {py_func.__name__}. "
                        f"Exception {e} was raised."
                    )

                return py_func(*args, **kwargs)

        #wrapper.__signature__ = inspect.signature(py_func)
        return cast(Callable[P, R], wrapper)

    return decorator


def _install_numba_funcs(module: ModuleType, install_failbacks: bool) -> None:
    module_py = _load_module_copy(
        module,
        enable_numba=False,
    )

    module_nb = _load_module_copy(
        module,
        enable_numba=True,
    )

    for name, nb_obj in vars(module_nb).items():
        func_obj = None
        
        if isinstance(nb_obj, CPUDispatcher):
            py_obj = getattr(module_py, name, None)

            if py_obj is None:
                py_obj = nb_obj.py_func

            if install_failbacks:
                #func_obj = _py_failback(py_obj)(nb_obj)
                func_obj = _py_failback(nb_obj)(py_obj)
                
            _export_func(module, nb_obj, nb_obj.__name__ + "_numba")

        elif inspect.isfunction(nb_obj):
            func_obj = nb_obj

        if func_obj is not None:
            _export_func(module, func_obj)
            
