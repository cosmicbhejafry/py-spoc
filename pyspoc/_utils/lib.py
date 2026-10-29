import os
import sys
import inspect
import importlib

from types import ModuleType
from pathlib import Path
from importlib import util
from importlib.resources.abc import Traversable

from pyspoc.exceptions import (
    InternalModuleImportError,
    InternalModuleNotFoundError,
    ExternalModuleNotFoundError,
    ExternalModuleImportError
)



def get_library_resource(path: Path) -> Traversable | None:
    error_msg = f"File could not be found at: {path}"
    
    if path.is_absolute():
        if path.is_file():
            return path

        raise FileNotFoundError(error_msg)

    from importlib.resources import files
    package_name = __package__

    if package_name is None or package_name == "_pyrepl":
        top_package_name = "pyspoc"
    else:
        top_package_name = package_name.split(".")[0]

    resource = files(top_package_name).joinpath(*path.parts)
    
    if resource.is_file():
        return resource

    raise FileNotFoundError(error_msg)
    
    
def _get_package_module(
        module_reference: str) -> ModuleType:
    
    # Get package modules.
    try:
        module = importlib.import_module(module_reference, __package__)
    except ModuleNotFoundError as e:
        raise InternalModuleNotFoundError(e.name)
    except ImportError as e:
        raise InternalModuleImportError(e.name)

    return module

    
def _get_loaded_module(
        module_reference: str,
        global_modules: dict | None = None) -> ModuleType:

    if not global_modules:
        global_modules = {obj.__name__.lower(): obj for obj
            in globals().values() if inspect.ismodule(obj)}
    
    module = global_modules.get(module_reference)

    if not module:
        raise ModuleNotFoundError(
            f"Module {module_reference} not found in memory.",
            name=module_reference)
    
    return module


def _load_external_module(path: str | Path, name: str) -> ModuleType:
    path = Path(path).resolve()
    
    # Otherwise check if module path exists
    if not os.path.isfile(path):
        raise ExternalModuleNotFoundError(path=path)

    spec = util.spec_from_file_location(name, Path(path).resolve())

    if spec is None or spec.loader is None:
        raise ExternalModuleImportError(path=path)

    module = util.module_from_spec(spec)

    # Register before execution, allowing self-imports and class introspection
    # used by tools such as dataclasses.
    previous = sys.modules.get(name)
    sys.modules[name] = module

    try:
        spec.loader.exec_module(module)
    except BaseException as e:
        # Restore the previous state if execution fails.
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous

        if isinstance(e, ImportError):
            raise ExternalModuleImportError(path=path) from e

        if isinstance(e, ModuleNotFoundError):
            raise ModuleNotFoundError(path=path) from e
        
        raise e

    return module


def get_containing_package(module: ModuleType) -> ModuleType | None:
    """
    Return the module itself if it is a package,
    otherwise its parent package.
    """
    
    if hasattr(module, "__path__"):
        return module

    if not hasattr(module, "__package__"):
        return None
    
    package_name = module.__package__

    if not package_name:
        return None

    return importlib.import_module(package_name)
