import pytest
import numpy as np
import os
import sys
from pathlib import Path

# Keep JIT caches local to the checkout, including third-party cached kernels.
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(__file__).resolve().parents[1] / ".pytest_cache" / "numba"))

# Config probes external executables during package import. Unit-test
# collection must not start Octave or a JVM (or wait for their startup).
# Restore normal imports immediately afterwards; individual dependency tests
# can still exercise the real import/error paths with their own mocks.
_missing = object()
_external_modules = {name: sys.modules.get(name, _missing) for name in ("oct2py", "jpype")}
try:
    sys.modules.update({name: None for name in _external_modules})
    import pyspoc
finally:
    # Restore only these two entries. Restoring the entire sys.modules mapping
    # would unload NumPy/SciPy extensions imported above, making reimport unsafe.
    for _name, _previous in _external_modules.items():
        if _previous is _missing:
            sys.modules.pop(_name, None)
        else:
            sys.modules[_name] = _previous

@pytest.fixture
def rng():
    return np.random.default_rng(0)

@pytest.fixture
def data_factory():
    def make(n=50, dim=2, seed=0):
        rng = np.random.default_rng(seed)
        return rng.random((n, dim))
    return make
