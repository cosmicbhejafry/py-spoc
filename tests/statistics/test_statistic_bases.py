"""Contract tests for the core Statistic result-shape hierarchy."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from typeguard import TypeCheckError

from pyspoc.core.statistic import (
    SPSDMatrixStatistic,
    ScalarStatistic,
    SquareMatrixStatistic,
    VectorStatistic,
)
from pyspoc.core.types import NumpyRealSPSDMatrix, NumpyRealSquareMatrix
from pyspoc.settings import settings


FloatDataMatrix = np.ndarray[tuple[int, int], np.dtype[np.floating]]
ResultFactory = Callable[[FloatDataMatrix], object]


class _ScalarStatistic(ScalarStatistic):
    def __init__(self, result_factory: ResultFactory) -> None:
        self._result_factory = result_factory
        super().__init__("test-scalar", "test-scalar", [])

    def _summarize(self, data: FloatDataMatrix) -> float:
        return self._result_factory(data)  # type: ignore[return-value]


class _VectorStatistic(VectorStatistic):
    def __init__(self, result_factory: ResultFactory) -> None:
        self._result_factory = result_factory
        super().__init__("test-vector", "test-vector", [])

    def _validate_output(self, result):
        # Concrete test doubles implement the abstract result-validation hook.
        self.validate(result)

    def _summarize(
        self,
        data: FloatDataMatrix,
    ) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
        return self._result_factory(data)  # type: ignore[return-value]


class _SquareMatrixStatistic(SquareMatrixStatistic):
    def __init__(self, result_factory: ResultFactory) -> None:
        self._result_factory = result_factory
        super().__init__("test-square-matrix", "test-square-matrix", [])

    def _summarize(self, data: FloatDataMatrix) -> NumpyRealSquareMatrix:
        return self._result_factory(data)  # type: ignore[return-value]


class _SPSDMatrixStatistic(SPSDMatrixStatistic):
    def __init__(self, result_factory: ResultFactory) -> None:
        self._result_factory = result_factory
        super().__init__("test-spsd-matrix", "test-spsd-matrix", [])

    def _summarize(self, data: FloatDataMatrix) -> NumpyRealSPSDMatrix:
        return self._result_factory(data)  # type: ignore[return-value]


@pytest.fixture
def data_matrix() -> FloatDataMatrix:
    """Return a deterministic valid Statistic input matrix."""
    return np.arange(24, dtype=np.float64).reshape(8, 3)


@pytest.mark.parametrize(
    "invalid_data",
    [
        np.array(0.0),
        np.arange(8, dtype=np.float64),
        np.ones((2, 3, 4), dtype=np.float64),
        [[1.0, 2.0], [3.0, 4.0]],
    ],
    ids=["zero-dimensional", "vector", "three-dimensional", "list"],
)
def test_summarize_rejects_invalid_input_data(invalid_data: object) -> None:
    """The public boundary accepts only two-dimensional floating ndarrays."""
    statistic = _ScalarStatistic(lambda _data: 0.0)

    with settings.override(result_validation="both"):
        with pytest.raises((TypeError, ValueError, TypeCheckError)):
            statistic.summarize(invalid_data)  # type: ignore[arg-type]


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_summarize_accepts_floating_matrix_dtypes(dtype: type[np.floating]) -> None:
    """All NumPy floating precisions satisfy the shared input contract."""
    data = np.ones((5, 2), dtype=dtype)
    statistic = _ScalarStatistic(lambda _data: 1.0)

    assert statistic.summarize(data) == 1.0


def test_scalar_statistic_accepts_float(data_matrix: FloatDataMatrix) -> None:
    result = 2.5
    statistic = _ScalarStatistic(lambda _data: result)

    assert statistic.summarize(data_matrix) is result


@pytest.mark.parametrize(
    "invalid_result",
    [np.array([1.0]), np.ones((2, 2)), np.array(1.0 + 2j), "invalid"],
    ids=["vector", "matrix", "complex", "string"],
)
def test_scalar_statistic_rejects_non_float_results(
    data_matrix: FloatDataMatrix,
    invalid_result: object,
) -> None:
    statistic = _ScalarStatistic(lambda _data: invalid_result)

    with pytest.raises(TypeCheckError):
        statistic.summarize(data_matrix)


def test_scalar_statistic_accepts_integer_as_float_compatible(
    data_matrix: FloatDataMatrix,
) -> None:
    """Typeguard's numeric tower permits integers for a float annotation."""
    statistic = _ScalarStatistic(lambda _data: 0)

    assert statistic.summarize(data_matrix) == 0


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_vector_statistic_accepts_nonempty_float_vector(
    data_matrix: FloatDataMatrix,
    dtype: type[np.floating],
) -> None:
    result = np.array([1.0, 2.0], dtype=dtype)
    statistic = _VectorStatistic(lambda _data: result)

    assert statistic.summarize(data_matrix) is result


@pytest.mark.parametrize(
    "invalid_result",
    [
        np.array(0.0),
        np.array([], dtype=float),
        np.array([[0.0, 1.0], [2.0, 3.0]]),
        np.zeros((2, 2, 2)),
        np.array([1 + 1j, 2 + 1j]),
        [1.0, 2.0],
    ],
    ids=["zero-dimensional", "empty", "matrix", "three-dimensional", "complex", "list"],
)
def test_vector_statistic_rejects_invalid_results(
    data_matrix: FloatDataMatrix,
    invalid_result: object,
) -> None:
    statistic = _VectorStatistic(lambda _data: invalid_result)
    with pytest.raises((TypeError, TypeCheckError, ValueError)):
        statistic.summarize(data_matrix)


@pytest.mark.parametrize("size", [2, 5])
def test_square_matrix_statistic_accepts_float_square_matrix(
    data_matrix: FloatDataMatrix,
    size: int,
) -> None:
    result = np.eye(size, dtype=np.float32)
    statistic = _SquareMatrixStatistic(lambda _data: result)

    assert statistic.summarize(data_matrix) is result


@pytest.mark.parametrize(
    "invalid_result",
    [
        np.array(0.0),
        np.array([0.0, 1.0]),
        np.empty((0, 0), dtype=float),
        np.array([[0.0]]),
        np.ones((1, 2)),
        np.ones((2, 3)),
        np.ones((2, 2, 2)),
        np.eye(2, dtype=complex),
    ],
    ids=[
        "zero-dimensional",
        "vector",
        "empty",
        "embedded-scalar",
        "embedded-vector",
        "rectangular",
        "three-dimensional",
        "complex",
    ],
)
def test_square_matrix_statistic_rejects_invalid_results(
    data_matrix: FloatDataMatrix,
    invalid_result: object,
) -> None:
    statistic = _SquareMatrixStatistic(lambda _data: invalid_result)

    with pytest.raises((TypeCheckError, ValueError)):
        statistic.summarize(data_matrix)


@pytest.mark.parametrize(
    "valid_result",
    [
        np.eye(3, dtype=np.float64),
        np.array([[2.0, -1.0], [-1.0, 2.0]], dtype=np.float32),
        np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64),
    ],
    ids=["positive-definite", "symmetric-positive-definite", "semidefinite"],
)
def test_spsd_statistic_accepts_symmetric_positive_semidefinite_results(
    data_matrix: FloatDataMatrix,
    valid_result: np.ndarray,
) -> None:
    statistic = _SPSDMatrixStatistic(lambda _data: valid_result)

    assert statistic.summarize(data_matrix) is valid_result


@pytest.mark.parametrize(
    ("invalid_result", "message"),
    [
        (np.array([[1.0, 2.0], [0.0, 1.0]]), "Hermitian"),
        (np.array([[1.0, 0.0], [0.0, -1.0]]), "positive-semidefinite"),
        (np.ones((2, 3), dtype=float), None),
    ],
    ids=["non-hermitian", "negative-eigenvalue", "non-square"],
)
def test_spsd_statistic_rejects_invalid_results(
    data_matrix: FloatDataMatrix,
    invalid_result: np.ndarray,
    message: str | None,
) -> None:
    statistic = _SPSDMatrixStatistic(lambda _data: invalid_result)

    if message is None:
        with pytest.raises(TypeCheckError):
            statistic.summarize(data_matrix)
    else:
        with pytest.raises(ValueError, match=message):
            statistic.summarize(data_matrix)

@pytest.mark.parametrize("value", [np.float32(2.0), np.array(2.0), 2, np.int64(2)])
def test_scalar_statistic_accepts_real_scalar_representations(data_matrix, value):
    """The expanded real-number contract includes NumPy and embedded scalars."""
    statistic = _ScalarStatistic(lambda _data: value)
    assert statistic.summarize(data_matrix) is value


@pytest.mark.parametrize("statistic_type, value", [
    (_VectorStatistic, np.array([1, 2])),
    (_SquareMatrixStatistic, np.eye(2, dtype=int)),
])
def test_array_statistic_accepts_integer_outputs(data_matrix, statistic_type, value):
    """Integer-valued statistics are now legitimate real-valued outputs."""
    assert statistic_type(lambda _data: value).summarize(data_matrix) is value
