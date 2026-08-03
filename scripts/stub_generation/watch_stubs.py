from __future__ import annotations

from pathlib import Path

from watchfiles import watch

from pyspoc.tools.generate_func_stubs import main as generate_stubs


def main() -> None:
    watched_files = {
        Path("pyspoc/kernels_numba.py").resolve(),
        Path("pyspoc/kernels_py.py").resolve(),
    }

    print("Watching for kernel changes...")
    generate_stubs()

    for changes in watch(*[str(path.parent) for path in watched_files]):
        changed_paths = {Path(path).resolve() for _, path in changes}

        if changed_paths & watched_files:
            print("Kernel source changed; regenerating funcs.pyi...")
            generate_stubs()


if __name__ == "__main__":
    main()