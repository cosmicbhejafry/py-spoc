"""Python, nopython and settings-dispatch contracts for condensed distances."""

import numpy as np
import pytest

from pyspoc.settings import settings
from pyspoc.statistics.clustering._base import (
    _minkowski_condensed_distances_numba,
    _minkowski_condensed_distances_numpy,
    _minkowski_condensed_distances_numpy_nb,
)


@pytest.fixture(params=[_minkowski_condensed_distances_numpy,
                        _minkowski_condensed_distances_numba], ids=["python", "numba"])
def implementation(request):
    # Direct compiled calls ensure automatic fallback cannot hide a typing failure.
    return request.param


@pytest.mark.parametrize("p", [0.5, 1., 2., 3.])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_values_pair_order_and_input_preservation(implementation, p, dtype):
    x = np.random.default_rng(27).normal(size=(7, 4)).astype(dtype)
    original = x.copy()
    i, j = np.triu_indices(len(x), k=1)
    converted = x.astype(np.float64)
    expected = np.linalg.norm(converted[i] - converted[j], ord=p, axis=1)
    result = implementation(x, p)
    assert result.shape == (21,)
    assert result.dtype == np.float64
    np.testing.assert_allclose(result, expected, rtol=1e-13, atol=1e-13)
    np.testing.assert_array_equal(x, original)


@pytest.mark.parametrize("shape", [(0, 3), (1, 3), (4, 0)])
def test_empty_axes(implementation, shape):
    n = shape[0]
    result = implementation(np.zeros(shape), 2.)
    np.testing.assert_array_equal(result, np.zeros(n * (n - 1) // 2))


def test_duplicates_offsets_and_strided_readonly_input(implementation):
    x = np.array([[0., 0.], [99., 99.], [0., 0.], [99., 99.], [3., 4.]])
    x = (x + 1e9)[::2]
    x.flags.writeable = False
    np.testing.assert_array_equal(implementation(x, 2.), [0., 5., 5.])


@pytest.mark.parametrize("p", [0., -1., np.nan, np.inf, -np.inf])
def test_invalid_p(implementation, p):
    with pytest.raises(ValueError, match="p must"):
        implementation(np.ones((3, 2)), p)


@pytest.mark.parametrize("mode", ["python", "numba"])
def test_settings_dispatch(mode):
    with settings.override(numba_mode=mode):
        result = _minkowski_condensed_distances_numpy_nb(
            np.array([[0., 0.], [3., 4.]]), 2.)
    np.testing.assert_array_equal(result, [5.])
    if mode == "numba":
        assert _minkowski_condensed_distances_numba.nopython_signatures
