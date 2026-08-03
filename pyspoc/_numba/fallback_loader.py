# from __future__ import annotations

import inspect
import importlib.util
import sys
import warnings

from collections.abc import Iterable, Callable
from functools import wraps
from types import ModuleType
from typing import Literal, ParamSpec, TypeVar, cast
from numba.core.registry import CPUDispatcher
from numba.core.errors import LoweringError, TypingError, UnsupportedError

from .export import export_func
from pyspoc.settings import settings

P = ParamSpec("P")
R = TypeVar("R")

DebugNumba = Iterable[Literal["ignore", "warn", "raise", "bounds"]]

_NUMBA_FALLBACK_ERRORS = (
    TypingError,
    LoweringError,
    UnsupportedError,
)

_PYTHON_FALLBACK_ATTRIBUTE = "__pyspoc_python_fallback__"

# ---------------------------------------------------------------------
# Loader machinery
# ---------------------------------------------------------------------


def python_fallback(fallback: Callable[..., R]) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Associate an explicit Python fallback with a Numba function."""

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        setattr(func, _PYTHON_FALLBACK_ATTRIBUTE, fallback)
        return func

    return decorator


def numba_dispatch(
    numba_func: CPUDispatcher,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Dispatch between an explicitly compiled Numba function and Python."""

    def decorator(python_func: Callable[P, R]) -> Callable[P, R]:
        return _create_numba_dispatch_wrapper(numba_func)(python_func)

    return decorator


def install_numba_funcs(module: ModuleType):
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

            explicit_fallback = getattr(py_obj, _PYTHON_FALLBACK_ATTRIBUTE, None)

            func_obj = _create_numba_dispatch_wrapper(nb_obj, fallback_func=explicit_fallback)(
                py_obj
            )

            export_func(module, nb_obj, nb_obj.__name__ + "_numba")

        elif inspect.isfunction(nb_obj):
            func_obj = nb_obj

        if func_obj is not None:
            export_func(module, func_obj)


def _get_module_source_path(module: ModuleType) -> str:
    spec = getattr(module, "__spec__", None)

    if spec is not None and spec.origin not in (None, "built-in", "frozen"):
        return spec.origin

    source_path = getattr(module, "__file__", None)

    if source_path is None:
        raise ValueError(f"Module {module.__name__!r} has no source path.")

    return source_path


def _load_module_copy(module: ModuleType, *, enable_numba: bool) -> ModuleType:
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


def _create_numba_dispatch_wrapper(
    numba_func: CPUDispatcher, *, fallback_func: Callable[P, R] | None = None
) -> Callable[[Callable[P, R]], Callable[P, R]]:

    def decorator(py_func: Callable[P, R]) -> Callable[P, R]:
        selected_fallback = fallback_func or py_func

        @wraps(py_func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:

            # Resolve the complete typed snapshot once so every setting used
            # by this call comes from the same execution context.
            current_settings = settings.current
            mode = current_settings.numba_mode

            if mode == "python":
                if current_settings.verbose:
                    print(f"Python ran {py_func.__name__} successfully.")

                return selected_fallback(*args, **kwargs)

            try:
                result = numba_func(*args, **kwargs)

                if current_settings.verbose:
                    print(f"Numba ran {py_func.__name__} successfully.")

                return cast(R, result)

            except _NUMBA_FALLBACK_ERRORS as e:
                if mode == "numba":
                    raise

                if current_settings.verbose:
                    warnings.warn(
                        f"Numba could not compile {py_func.__qualname__}; "
                        "using its Python implementation instead. "
                        f"Numba reported: {e}",
                        RuntimeWarning,
                        stacklevel=2,
                    )

                return py_func(*args, **kwargs)

        # wrapper.__signature__ = inspect.signature(py_func)
        return cast(Callable[P, R], wrapper)

    return decorator
