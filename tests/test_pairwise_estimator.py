"""Pair-model identity, lazy fitting and worker-pool integration tests."""

from concurrent.futures import Future, ThreadPoolExecutor
from threading import Barrier, Event, Lock

import numpy as np
import pytest

from pyspoc._estimators import LazyFittedPairwiseCachedEstimatorMixin
from pyspoc._execution import execute_pairwise
from pyspoc._execution.pairwise import resolve_worker_limit
from pyspoc.settings import settings
from pyspoc.statistics.base import PairwiseStatistic


class PairEstimator(LazyFittedPairwiseCachedEstimatorMixin[tuple[float, float]]):
    def __init__(self, scale: float = 1.0):
        self.scale = scale
        self.calls = 0
        self.lock = Lock()

    def _fit_estimator(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        assert not x.flags.writeable and not y.flags.writeable
        with self.lock:
            self.calls += 1
        return self.scale * float(x.sum()), float(y.sum())


class PairModelStatistic(PairwiseStatistic):
    def __init__(self, dim="p", symmetry_type=None, is_ordered=False):
        self.estimator = PairEstimator()
        super().__init__(dim=dim, symmetry_type=symmetry_type,
                         is_ordered=is_ordered, thread_safety="safe")

    def _compute(self, data):
        # Cache keys refer to prepared columns, so attach sorted data if this
        # Statistic requests sorting (axis selection already happened in compute).
        if self.is_ordered:
            data = np.sort(data, axis=0)
        self.estimator.fit(data)
        return super()._compute(data)

    def _pairwise_compute_indexed(self, x, y, index):
        model = self.estimator._get_model(index)
        # Verify indices address precisely the columns supplied to the hook.
        assert model == (float(x.sum()), float(y.sum()))
        return model[0] - model[1]


def test_lazy_models_have_ordered_pair_identity_and_reuse():
    data = np.arange(12.0).reshape(4, 3)
    estimator = PairEstimator().fit(data)
    assert estimator.calls == 0
    first = estimator._get_model((0, 1))
    assert estimator._get_model((0, 1)) is first
    assert estimator._get_model((1, 0)) == first[::-1]
    estimator._get_model((0, 0))
    assert estimator.calls == 3
    np.testing.assert_array_equal(data, np.arange(12.0).reshape(4, 3))


def test_estimator_cache_shares_models_but_separates_configuration_and_data():
    PairEstimator._reset_cache()
    data = np.arange(12.0).reshape(4, 3)
    first = PairEstimator.get_or_create(data, scale=2.0)
    model = first._get_model((0, 1))
    second = PairEstimator.get_or_create(data.copy(), 2.0)
    assert second is first
    assert second._get_model((0, 1)) is model
    assert PairEstimator.get_or_create(data, 3.0) is not first
    assert PairEstimator.get_or_create(data + 1, 2.0) is not first


@pytest.mark.parametrize("key", [(-1, 0), (0, 3)])
def test_out_of_range_model_keys_are_rejected(key):
    with pytest.raises(IndexError):
        PairEstimator().fit(np.ones((4, 3)))._get_model(key)


@pytest.mark.parametrize("key", [(True, 0), (0.5, 1), (0,), [0, 1]])
def test_noninteger_or_malformed_model_keys_are_rejected(key):
    with pytest.raises(TypeError):
        PairEstimator().fit(np.ones((4, 3)))._get_model(key)


def test_attachment_validation_and_unchanged_existing_models():
    estimator = PairEstimator()
    with pytest.raises(ValueError, match="No dataset"):
        estimator._get_model((0, 0))
    with pytest.raises(ValueError, match="two-dimensional"):
        estimator.fit(np.ones(4))
    estimator.fit(np.ones((4, 3)))
    model = estimator._get_model((0, 1))
    assert estimator.fit(np.ones((4, 3))) is estimator
    with pytest.raises(ValueError, match="does not match"):
        estimator.fit(np.zeros((4, 3)))
    assert estimator._get_model((0, 1)) is model


def test_same_pair_waiters_share_one_fit():
    start = Barrier(4)
    entered, release = Event(), Event()

    class SlowEstimator(PairEstimator):
        def _fit_estimator(self, x, y):
            entered.set()
            assert release.wait(5)
            return super()._fit_estimator(x, y)

    estimator = SlowEstimator().fit(np.ones((4, 3)))

    def request():
        # Launch all callers together. Only the owner may enter fitting; the
        # others wait on its Future rather than owning additional fits.
        start.wait(timeout=5)
        return estimator._get_model((0, 1))

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(request) for _ in range(4)]
        assert entered.wait(5)
        release.set()
        models = [future.result(timeout=5) for future in futures]
    assert estimator.calls == 1
    assert all(model is models[0] for model in models)


def test_distinct_pairs_fit_concurrently_in_existing_executor():
    rendezvous = Barrier(2)

    class ConcurrentEstimator(PairEstimator):
        def _fit_estimator(self, x, y):
            # This times out if one global fitting lock serializes all pairs.
            rendezvous.wait(timeout=5)
            return super()._fit_estimator(x, y)

    data = np.arange(8.0).reshape(4, 2)
    estimator = ConcurrentEstimator().fit(data)
    result, errors = execute_pairwise(
        data, lambda x, y: 0.0, max_workers=2, chunks_per_worker=2,
        indexed_operation=lambda x, y, index: sum(estimator._get_model(index)),
    )
    assert errors == []
    sums = data.sum(axis=0)
    np.testing.assert_allclose(result, sums[:, None] + sums[None, :])
    assert estimator.calls == 4


def test_fit_failure_reaches_waiters_and_allows_retry(monkeypatch):
    entered, waiting, release = Event(), Event(), Event()

    class ObservedFuture(Future):
        def result(self, timeout=None):
            # Confirm the second caller is waiting on the same pending Future
            # before releasing the failing owner. No timing-based sleeps needed.
            waiting.set()
            return super().result(timeout)

    monkeypatch.setattr("pyspoc._estimators.pairwise.Future", ObservedFuture)

    class FailingEstimator(PairEstimator):
        def _fit_estimator(self, x, y):
            entered.set()
            assert release.wait(5)
            if self.calls == 0:
                self.calls += 1
                raise ValueError("pair fitting failed")
            return super()._fit_estimator(x, y)

    estimator = FailingEstimator().fit(np.ones((4, 2)))
    with ThreadPoolExecutor(max_workers=2) as pool:
        owner = pool.submit(estimator._get_model, (0, 1))
        assert entered.wait(5)
        waiter = pool.submit(estimator._get_model, (0, 1))
        try:
            assert waiting.wait(5)
        finally:
            release.set()
        for future in (owner, waiter):
            with pytest.raises(ValueError, match="pair fitting failed"):
                future.result(timeout=5)
    assert estimator._get_model((0, 1)) == (4.0, 4.0)
    assert estimator.calls == 2


@pytest.mark.parametrize("dim", ["n", "p"])
@pytest.mark.parametrize("workers", [1, 2])
def test_statistic_pipeline_and_model_reuse(dim, workers):
    data = np.arange(12.0).reshape(4, 3)
    statistic = PairModelStatistic(dim=dim)
    sums = data.sum(axis=1 if dim == "n" else 0)
    expected = sums[:, None] - sums[None, :]
    with settings.override(max_worker_threads=workers):
        np.testing.assert_allclose(statistic.compute(data), expected)
        np.testing.assert_allclose(statistic.compute(data.copy()), expected)
    assert statistic.estimator.calls == len(sums) ** 2


def test_symmetric_statistic_only_fits_scheduled_triangle():
    data = np.arange(12.0).reshape(4, 3)
    statistic = PairModelStatistic(symmetry_type="negative")
    with settings.override(max_worker_threads=2):
        result = statistic.compute(data)
    sums = data.sum(axis=0)
    np.testing.assert_allclose(result, sums[:, None] - sums[None, :])
    assert statistic.estimator.calls == 6


def test_configured_all_workers_sentinel(monkeypatch):
    monkeypatch.setattr("pyspoc._execution.pairwise.os.process_cpu_count", lambda: 3)
    with settings.override(max_worker_threads=-1):
        assert resolve_worker_limit(None) == 3
        assert resolve_worker_limit(1) == 1


def test_ordered_statistic_attaches_prepared_data():
    data = np.array([[3.0, 1.0], [1.0, 2.0], [2.0, 0.0]])
    statistic = PairModelStatistic(is_ordered=True)
    statistic.compute(data)
    np.testing.assert_array_equal(
        statistic.estimator._get_attached_dataset(), np.sort(data, axis=0))


@pytest.mark.parametrize("workers", [1, 2])
def test_indexed_executor_two_arrays_and_error_context(workers):
    data = np.ones((4, 2))
    second = np.ones((4, 3))

    def indexed(x, y, index):
        if index == (1, 2):
            raise ValueError("bad pair")
        return float(10 * index[0] + index[1])

    result, errors = execute_pairwise(
        data, lambda x, y: -1.0, second,
        indexed_operation=indexed, max_workers=workers,
    )
    np.testing.assert_allclose(result, [[0, 1, 2], [10, 11, np.nan]])
    assert len(errors) == 1
    assert "i=1, j=2" in errors[0].__notes__[0]
    with pytest.raises(ValueError, match="bad pair"):
        execute_pairwise(data, lambda x, y: -1.0, second,
                         indexed_operation=indexed, max_workers=workers,
                         throw_first_error=True)
