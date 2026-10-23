from importlib.resources.abc import Traversable
from pathlib import Path


OPTIONAL_MODULES = (
    "torch",
    "pygam"
)

# URLs
TORCH_URL = "https://pytorch.org/"
PYSPOC_GITHUB_URL = "https://github.com/cosmicbhejafry/py-spoc"


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
