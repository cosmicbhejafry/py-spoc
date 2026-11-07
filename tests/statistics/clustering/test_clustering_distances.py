"""Direct compilation tests for the standalone clustering distance kernel."""

import numpy as np
import pytest

from pyspoc.statistics.clustering._base import _minkowski_pairwise_distances_numba


@pytest.mark.parametrize("x_dtype,y_dtype", [
    (np.float32, np.float32), (np.float64, np.float64), (np.float32, np.float64),
])
@pytest.mark.parametrize("block_size", [1, 3, 128])
def test_compiled_distances_match_coordinate_differences(x_dtype, y_dtype, block_size):
    """Call the dispatcher directly so Python fallback cannot hide compilation errors."""
    rng = np.random.default_rng(21)
    x = rng.normal(size=(7, 4)).astype(x_dtype)
    y = rng.normal(size=(5, 4)).astype(y_dtype)
    original_x, original_y = x.copy(), y.copy()
    expected = np.linalg.norm(
        x.astype(np.float64)[:, None, :] - y.astype(np.float64)[None, :, :], axis=2)

    result = _minkowski_pairwise_distances_numba(x, y, p=2., block_size=block_size)

    assert result.shape == (7, 5)
    assert result.dtype == np.float64
    assert _minkowski_pairwise_distances_numba.nopython_signatures
    np.testing.assert_allclose(result, expected, rtol=1e-14, atol=1e-14)
    np.testing.assert_array_equal(x, original_x)
    np.testing.assert_array_equal(y, original_y)


def test_identical_rows_and_small_distances_with_large_offset():
    """A large shared offset must not obscure exact zeros or small differences."""
    x = 1e9 + np.array([[0., 1., 2.], [3., 4., 5.]])
    y = x.copy()
    y[1, 0] += 1e-4
    result = _minkowski_pairwise_distances_numba(x, y, p=2.)
    assert result[0, 0] == 0.0
    assert result[1, 1] == abs(x[1, 0] - y[1, 0])


def test_noncontiguous_readonly_inputs():
    """The kernel must not require writable or contiguous source arrays."""
    x = np.arange(60., dtype=np.float64).reshape(10, 6)[::2, ::2]
    y = x[::-1]
    x.flags.writeable = False
    y.flags.writeable = False
    expected = np.linalg.norm(x[:, None, :] - y[None, :, :], axis=2)
    np.testing.assert_allclose(_minkowski_pairwise_distances_numba(x, y, p=2.), expected)


@pytest.mark.parametrize("n,m,p", [(0, 3, 2), (3, 0, 2), (0, 0, 2), (2, 3, 0)])
def test_empty_axes(n, m, p):
    result = _minkowski_pairwise_distances_numba(np.zeros((n, p)), np.zeros((m, p)), p=2.)
    assert result.shape == (n, m)
    np.testing.assert_array_equal(result, np.zeros((n, m)))


@pytest.mark.parametrize("block_size", [0, -1])
def test_invalid_block_size(block_size):
    with pytest.raises(ValueError, match="block_size"):
        _minkowski_pairwise_distances_numba(
            np.zeros((2, 3)), np.zeros((4, 3)), p=2., block_size=block_size)


def test_mismatched_feature_counts():
    with pytest.raises(ValueError, match="features"):
        _minkowski_pairwise_distances_numba(np.zeros((2, 3)), np.zeros((4, 2)), p=2.)
