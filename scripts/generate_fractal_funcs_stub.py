from __future__ import annotations

import inspect

from pathlib import Path
from types import ModuleType
from typing import Iterable
from numba.core.registry import CPUDispatcher

from pyspoc.rstatistics.fractal import _funcs_numba


def _format_annotation(annotation: object) -> str:
    if annotation is inspect.Signature.empty:
        return "Any"

    if annotation is None:
        return "None"

    if isinstance(annotation, str):
        return f"'{annotation}'"

    if isinstance(annotation, type):
        if annotation.__module__ == "builtins":
            return annotation.__name__
        return f"{annotation.__module__}.{annotation.__qualname__}"

    text = repr(annotation)
    text = text.replace("typing.", "")
    text = text.replace("numpy.", "np.")
    return text


def _default_to_stub(default: object) -> str:
    if default is inspect.Signature.empty:
        return ""
    
    if isinstance(default, str):
        return f" = '{default}'"
    
    return f" = {default}"


def _signature_to_stub(sig: inspect.Signature) -> str:
    parts: list[str] = []

    inserted_kw_separator = False

    for name, param in sig.parameters.items():
        prefix = ""

        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            prefix = "*"
        elif param.kind is inspect.Parameter.VAR_KEYWORD:
            prefix = "**"
        elif param.kind is inspect.Parameter.KEYWORD_ONLY and not inserted_kw_separator:
            if not any(
                p.kind is inspect.Parameter.VAR_POSITIONAL
                for p in sig.parameters.values()
            ):
                parts.append("*")
            inserted_kw_separator = True

        annotation = _format_annotation(param.annotation)
        default = _default_to_stub(param.default)

        parts.append(f"{prefix}{name}: {annotation}{default}")

    return_annotation = _format_annotation(sig.return_annotation)
    return f"({', '.join(parts)}) -> {return_annotation}"


def _signature_with_debug_numba(
    sig: inspect.Signature,
) -> inspect.Signature:
    
    if "debug_numba" in sig.parameters:
        return sig

    params = list(sig.parameters.values())

    debug_param = inspect.Parameter(
        "debug_numba",
        kind=inspect.Parameter.KEYWORD_ONLY,
        default="ignore",
        annotation=Iterable[str],
    )

    for i, param in enumerate(params):
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            params.insert(i, debug_param)
            break
    else:
        params.append(debug_param)

    return sig.replace(parameters=params)


def generate_stub(
    source_module: ModuleType,
    *,
    output_path: Path,
    include_private: bool = False,
    include_debug_numba: bool = True,
) -> None:
    lines: list[str] = [
        "from __future__ import annotations",
        "",
        "from typing import Any",
        "import numpy as np",
        "",
        "# This file is generated from kernels_numba.py.",
        "# Do not edit manually.",
        "",
    ]

    exported_names: list[str] = []

    for name, obj in vars(source_module).items():
        if not include_private and name.startswith("_"):
            continue

        if not inspect.isfunction(obj) and not isinstance(obj, CPUDispatcher):
            continue

        try:
            sig = inspect.signature(obj)

            if include_debug_numba:
                sig = _signature_with_debug_numba(sig)

            sig_text = _signature_to_stub(sig)

        except (TypeError, ValueError):
            sig_text = "(*args: Any, **kwargs: Any) -> Any"

        lines.append(f"def {name}{sig_text}: ...")
        exported_names.append(name)

    lines.append("")
    lines.append("__all__: list[str]")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    output_path = Path(_funcs_numba.__file__).with_suffix(".pyi")

    generate_stub(
        _funcs_numba,
        output_path=output_path,
        include_private=True,
        include_debug_numba=False,
    )

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
