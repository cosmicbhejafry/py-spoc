"""Validated component configuration and normalized reduction pipelines."""

from __future__ import annotations

import re
import ntpath
import posixpath
import sys
from typing import Annotated, Any, Self

from pydantic import (
    BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator,
)

from pyspoc.component.statistics.base import Statistic
from pyspoc.component.rstatistics.base import ReducedStatistic
from pyspoc.component.reducers.base import Reducer


_GROUP_BRACKETS = {"[": "]", "(": ")", "{": "}"}
_GROUP_CHARACTERS = "[](){}"


def _is_module_path(reference: str) -> bool:
    """Return whether a qualifier denotes a file path rather than a dotted module.

    Parameters
    ----------
    reference : str
        Module qualifier, excluding its component name and scheme.

    Returns
    -------
    bool
        Whether the qualifier contains path syntax or a Python file extension.
    """
    return any(char in reference for char in "/\\:") or reference.endswith(".py")


def _module_identity(reference: str) -> tuple[str, str]:
    """Normalize a module qualifier without importing modules or accessing files.

    Parameters
    ----------
    reference : str
        Dotted module name or file path, optionally ending in ``.py``.

    Returns
    -------
    tuple of str
        Qualifier kind and normalized identity. Windows drive/UNC paths use
        Windows path semantics even on other platforms. File suffix omission
        and public/internal pySPoC import aliases do not change identity.
    """
    if _is_module_path(reference):
        path = reference.replace("\\", "/")
        if ntpath.splitdrive(path)[0]:
            path = ntpath.normcase(ntpath.normpath(path))
        else:
            path = posixpath.normpath(path)
        if path.endswith(".py"):
            path = path[:-3]
        return "path", path
    for package in ("statistics", "rstatistics", "reducers"):
        public = f"pyspoc.{package}"
        if reference == public or reference.startswith(public + "."):
            reference = "pyspoc.component." + reference[len("pyspoc."):]
            break
    return "module", reference


def _component_module_identities(component) -> set[tuple[str, str]]:
    """Collect the already-loaded module names and source paths for a component.

    Parameters
    ----------
    component : Statistic or Reducer
        Configured instance whose defining module is being matched.

    Returns
    -------
    set of tuple of str
        Comparable identities from class metadata and the ComponentModule
        attached by the external-component loader. No new files are executed.
    """
    module_name = type(component).__module__
    references = {module_name}
    module = sys.modules.get(module_name)
    if module is not None and getattr(module, "__file__", None):
        references.add(str(module.__file__))
    component_module = getattr(component, "_component_module", None)
    if component_module is not None:
        for field in ("module_reference", "module_path"):
            value = getattr(component_module, field, None)
            if value is not None:
                references.add(str(value))
    return {_module_identity(reference) for reference in references}


class ComponentReference(BaseModel):
    """Name a configured component, optionally selecting a single scheme.

    Parameters
    ----------
    name : str
        Component class name, without its module prefix.
    module : str or None, optional
        Dotted module qualifier or file path, with or without a ``.py`` suffix.
    scheme : str or None, optional
        Scheme selected with ``@scheme`` in expressions. ``None`` leaves
        scheme selection unrestricted. Dots are never scheme delimiters.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str
    module: str | None = None
    scheme: str | None = None

    @field_validator("name", "module", "scheme")
    @classmethod
    def _validate_reference_part(
        cls, value: str | None, info: ValidationInfo,
    ) -> str | None:
        """Trim and validate one component-reference field.

        Parameters
        ----------
        value : str or None
            Candidate component name, module qualifier, or scheme name.
        info : ValidationInfo
            Pydantic context identifying the field being validated.

        Returns
        -------
        str or None
            Trimmed reference field.

        Raises
        ------
        ValueError
            If a field is empty, ambiguous, or has an invalid class name.
        """
        if value is None:
            return value
        value = value.strip()
        if not value or any(
            (char.isspace() and not (info.field_name == "module" and _is_module_path(value)))
            or char in ",=>@" + _GROUP_CHARACTERS for char in value
        ):
            raise ValueError("Component and scheme names cannot be empty or contain separators.")
        if info.field_name == "name" and not value.isidentifier():
            raise ValueError("Component names must be Python class identifiers.")
        if info.field_name == "module" and not _is_module_path(value):
            if not all(part.isidentifier() for part in value.split(".")):
                raise ValueError("Dotted module qualifiers must contain valid identifiers.")
        return value


ReducerStage = Annotated[list[ComponentReference], Field(min_length=1)]


class MappingRule(BaseModel):
    """Represent a Statistic followed by one or more ordered Reducer stages.

    Parameters
    ----------
    statistic : ComponentReference
        Input Statistic reference. Only one is permitted per rule.
    reducer_stages : list of list of ComponentReference
        Ordered stages; each stage may select one or several Reducers. Stage
        boundaries are preserved for subsequent execution, not flattened.

    Notes
    -----
    ``model_validate`` also accepts an arrow expression, using either ``=>``
    or ``->`` between stages. Reducer groups optionally use matching square,
    round, or curly brackets; all three preserve order and repeated entries.
    A Reducer without a scheme selects all its configured schemes at any stage;
    a scheme-qualified reference selects only that configuration. Unrestricted
    references remain ``scheme=None`` until pipeline resolution.
    References use ``[module.]Component[@scheme]``. The module may instead be
    a path such as ``D:/plugins/metrics.py``; only the final dot separates it
    from the component class. The old ``Component.scheme`` spelling now denotes
    a module-qualified component, never a scheme selection.
    This model parses structure only: it never executes a component or checks
    intermediate result shapes. Scalar termination and warnings belong to the
    eventual pipeline executor.
    """

    model_config = ConfigDict(extra="forbid")
    statistic: ComponentReference
    reducer_stages: list[ReducerStage] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _parse_expression(cls, value: Any) -> Any:
        """Normalize an arrow expression into structured pipeline stages.

        Parameters
        ----------
        value : Any
            Expression string, or an already structured model/dictionary.

        Returns
        -------
        Any
            Structured mapping data, or the original non-string input for
            standard Pydantic model validation.

        Raises
        ------
        ValueError
            If arrows, references, brackets, or reducer groups are malformed.
        """
        if not isinstance(value, str):
            return value
        stages = re.split(r"=>|->", value.strip())
        if len(stages) < 2 or any(not stage.strip() for stage in stages):
            raise ValueError(
                "Mappings require a Statistic and nonempty Reducer stages joined by => or ->."
            )

        def reference(text: str) -> dict[str, str | None]:
            # @ makes scheme selection independent of dots in module names or
            # filenames. Only the final dot before @ separates module and class.
            component, separator, scheme = text.strip().partition("@")
            if separator and not scheme.strip():
                raise ValueError("@ must be followed by a scheme name.")
            module, dot, name = component.strip().rpartition(".")
            return {
                "name": name,
                "module": module if dot else None,
                "scheme": scheme if separator else None,
            }

        first = stages[0].strip()
        if any(char in first for char in "," + _GROUP_CHARACTERS):
            raise ValueError("The first stage must contain exactly one Statistic reference.")
        reducers = []
        for stage in stages[1:]:
            text = stage.strip()
            if text[0] in _GROUP_BRACKETS or text[-1] in _GROUP_BRACKETS.values():
                if _GROUP_BRACKETS.get(text[0]) != text[-1]:
                    raise ValueError("Reducer groups require matching [], (), or {} brackets.")
                text = text[1:-1].strip()
            if any(char in text for char in _GROUP_CHARACTERS):
                raise ValueError("Nested or unmatched reducer-group brackets are not supported.")
            entries = text.split(",")
            if any(not entry.strip() for entry in entries):
                raise ValueError("Reducer stages cannot be empty or contain empty entries.")
            reducers.append([reference(entry) for entry in entries])
        return {"statistic": reference(first), "reducer_stages": reducers}


class MappingsConfig(BaseModel):
    """Hold independent positive and negative lists using the same rule grammar.

    Parameters
    ----------
    Positive : list of MappingRule, optional
        Positive mapping pipelines. A single expression string is also accepted.
    Negative : list of MappingRule, optional
        Negative mapping pipelines, with the same syntax as ``Positive``.
    """

    model_config = ConfigDict(extra="forbid")
    Positive: list[MappingRule] = Field(default_factory=list)
    Negative: list[MappingRule] = Field(default_factory=list)

    @field_validator("Positive", "Negative", mode="before")
    @classmethod
    def _normalize_rules(cls, value: Any) -> Any:
        """Wrap a single expression in a list without altering existing lists.

        Parameters
        ----------
        value : Any
            Raw YAML section value.

        Returns
        -------
        Any
            A one-element list for strings, otherwise the original value.
        """
        return [value] if isinstance(value, str) else value



class ComponentStore(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    Statistics: dict[tuple[str, str], dict[str, Statistic]]
    ReducedStatistics: dict[tuple[str, str], dict[str, ReducedStatistic]]
    Reducers: dict[tuple[str, str], dict[str, Reducer]]
    Mappings: MappingsConfig = Field(default_factory=MappingsConfig)

    @model_validator(mode="after")
    def _validate_mapping_references(self) -> Self:
        """Require references to exist in their role-specific component registry.

        Returns
        -------
        Self
            Validated configuration, without resolving or executing pipelines.

        Raises
        ------
        ValueError
            If a Statistic or Reducer reference has no matching configured
            component/scheme in its required registry.
        """
        def check(reference: ComponentReference, registry, role: str, location: str):
            matches = set()
            for (registered_name, scheme), component in registry.items():
                if reference.scheme is not None and scheme != reference.scheme:
                    continue
                # Registries may retain short names or qualified names to store
                # colliding classes with identical scheme names independently.
                key_module, dot, short_name = registered_name.rpartition(".")
                if short_name != reference.name:
                    continue

                if reference.module is not None:
                    identities = _component_module_identities(component)
                    if dot:
                        identities.add(_module_identity(key_module))
                    if _module_identity(reference.module) not in identities:
                        continue
                # Multiple schemes of one class are not ambiguous. Distinct
                # classes sharing a short name must be qualified explicitly.
                matches.add(type(component))

            label = reference.name
            if reference.module is not None:
                label = f"{reference.module}.{label}"
            if reference.scheme is not None:
                label += f"@{reference.scheme}"
            if not matches:
                raise ValueError(f"{location}: unknown configured {role} reference {label!r}.")
            if len(matches) > 1:
                raise ValueError(
                    f"{location}: ambiguous {role} reference {label!r}; "
                    "qualify it with its module name or file path."
                )

        for section in ("Positive", "Negative"):
            for index, rule in enumerate(getattr(self.Mappings, section)):
                location = f"Mappings.{section}[{index}]"
                check(rule.statistic, self.Statistics, "Statistic", location)
                for stage_index, stage in enumerate(rule.reducer_stages):
                    for reducer in stage:
                        check(
                            reducer, self.Reducers, "Reducer",
                            f"{location}, reducer stage {stage_index + 1}",
                        )
        return self


class MappingTrunk(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    Branches: dict[Statistic, MappingTree]


class MappingTree(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    Branches: dict[Reducer, MappingTree]
