"""Component packages and identity-preserving public import aliases.

Implementation modules live under ``pyspoc.component``. The shorter
``pyspoc.statistics``, ``pyspoc.rstatistics`` and ``pyspoc.reducers`` paths are
aliases, including their descendants: a file must never execute independently
under both names, otherwise classes and their registries would be duplicated.
"""

import sys
from importlib import import_module
from importlib.abc import Loader, MetaPathFinder
from importlib.util import find_spec, spec_from_loader


_PUBLIC_PACKAGES = {
    f"{__name__.rsplit('.', 1)[0]}.{name}": f"{__name__}.{name}"
    for name in ("statistics", "rstatistics", "reducers")
}


class _ComponentAliasLoader(Loader):
    """Return an existing canonical module instead of executing its file again."""

    def __init__(self, canonical_name):
        self.canonical_name = canonical_name

    def create_module(self, spec):
        # import_module supplies normal import locking and executes the
        # implementation at most once, whether the public or internal path is
        # requested first. Python registers this same object under the alias.
        module = import_module(self.canonical_name)
        self.canonical_spec = module.__spec__
        return module

    def exec_module(self, module):
        # module_from_spec temporarily assigns the alias spec even when
        # create_module returns an existing module. Restore canonical metadata
        # so relative imports, reload(), resources and class __module__ names
        # continue to use the implementation path.
        module.__spec__ = self.canonical_spec


class _ComponentAliasFinder(MetaPathFinder):
    """Redirect only the three public component namespaces and their children."""

    _pyspoc_flat_component_imports = True

    def find_spec(self, fullname, path=None, target=None):
        for public, canonical in _PUBLIC_PACKAGES.items():
            if fullname != public and not fullname.startswith(public + "."):
                continue
            canonical_name = canonical + fullname[len(public):]
            try:
                canonical_spec = find_spec(canonical_name)
            except ModuleNotFoundError as error:
                # Translate a missing requested module back to its public
                # spelling, but never disguise a missing third-party dependency.
                if error.name and (
                    canonical_name == error.name
                    or canonical_name.startswith(error.name + ".")
                ):
                    missing_name = public + error.name[len(canonical):]
                    raise ModuleNotFoundError(
                        f"No module named {missing_name!r}", name=missing_name,
                    ) from error
                raise

            if canonical_spec is None:
                # Do not fall through to PathFinder: the alias package shares
                # the canonical __path__, so fallback could execute a file under
                # the wrong name and defeat the identity guarantee.
                raise ModuleNotFoundError(
                    f"No module named {fullname!r}", name=fullname,
                )
            spec = spec_from_loader(
                fullname,
                _ComponentAliasLoader(canonical_name),
                origin=canonical_spec.origin,
                is_package=canonical_spec.submodule_search_locations is not None,
            )
            return spec
        return None


# Register before importing children: their own imports may already use the
# public spelling. The marker also makes reloading this initializer idempotent.
if not any(
    getattr(finder, "_pyspoc_flat_component_imports", False)
    for finder in sys.meta_path
):
    sys.meta_path.insert(0, _ComponentAliasFinder())


from . import statistics, rstatistics, reducers  # noqa: E402

# Re-exporting attributes alone does not register importable module names.
# Descendants are registered lazily by the finder, without a recursive scan.
for _public_name, _canonical_name in _PUBLIC_PACKAGES.items():
    sys.modules[_public_name] = sys.modules[_canonical_name]

__all__ = [
    "statistics",
    "rstatistics",
    "reducers"
]
