from __future__ import annotations

import gc
import time
import weakref

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import numpy as np

from pyspoc._caching.caching import CachedEstimatorMixin
from pyspoc._random import RandomSeedMixin
from pyspoc.settings import settings


class AmbiguousEquality:
    def __eq__(self, other):
        return np.array([True, True])


class FailingEquality:
    def __eq__(self, other):
        raise TypeError("values cannot be compared")


class VerboseAgnosticEstimator(CachedEstimatorMixin):
    def __init__(self, alpha: float, verbose: bool = False):
        self.alpha = alpha
        self.verbose = verbose

    @classmethod
    def _canonicalize_cache_args(cls, estimator_kwargs):
        canonical_args = super()._canonicalize_cache_args(estimator_kwargs)
        canonical_args.pop("verbose", None)
        return canonical_args


class CoarseCandidateEstimator(CachedEstimatorMixin):
    @classmethod
    def _get_candidate_args(cls, estimator_kwargs):
        candidate_args = super()._get_candidate_args(estimator_kwargs)
        candidate_args.pop("components", None)
        return candidate_args


class SlowEstimator(CachedEstimatorMixin):
    """Estimator with deliberately slow construction for concurrency tests."""

    def __init__(self, alpha: float):
        time.sleep(0.02)
        self.alpha = alpha


class PositionalSeedEstimator(RandomSeedMixin, CachedEstimatorMixin):
    """Cached estimator combining a positional-only argument and seed."""

    _freeze_random_seed = True

    def __init__(
            self,
            alpha: float,
            /,
            random_seed: int | None = None):
        self.alpha = alpha


def test_is_arg_match_accepts_equal_python_values():
    args = {"components": [1, 2], "alpha": 0.1}
    reference_args = {"components": [1, 2], "alpha": 0.1}

    assert CachedEstimatorMixin._is_arg_match(args, reference_args)


def test_is_arg_match_rejects_different_argument_names():
    args = {"alpha": 0.1}
    reference_args = {"beta": 0.1}

    assert not CachedEstimatorMixin._is_arg_match(args, reference_args)


def test_is_arg_match_compares_numpy_arrays_by_value():
    args = {"weights": np.array([1.0, 2.0])}
    equal_args = {"weights": np.array([1.0, 2.0])}
    different_args = {"weights": np.array([1.0, 3.0])}

    assert CachedEstimatorMixin._is_arg_match(args, equal_args)
    assert not CachedEstimatorMixin._is_arg_match(args, different_args)


def test_is_arg_match_rejects_ambiguous_equality_results():
    args = {"value": AmbiguousEquality()}
    reference_args = {"value": AmbiguousEquality()}

    assert not CachedEstimatorMixin._is_arg_match(args, reference_args)


def test_is_arg_match_rejects_failed_equality_comparisons():
    args = {"value": FailingEquality()}
    reference_args = {"value": FailingEquality()}

    assert not CachedEstimatorMixin._is_arg_match(args, reference_args)


def test_is_arg_match_accepts_identical_objects_without_equality():
    value = FailingEquality()

    assert CachedEstimatorMixin._is_arg_match(
        {"value": value},
        {"value": value},
    )


def test_canonicalized_arguments_do_not_affect_hash_or_exact_match():
    quiet_args = {"alpha": 0.1, "verbose": False}
    verbose_args = {"alpha": 0.1, "verbose": True}

    assert (
        VerboseAgnosticEstimator._get_args_hash(quiet_args)
        == VerboseAgnosticEstimator._get_args_hash(verbose_args)
    )
    assert VerboseAgnosticEstimator._is_arg_match(quiet_args, verbose_args)


def test_candidate_arguments_can_be_coarser_than_exact_comparison():
    small_request = {"alpha": 0.1, "components": (1, 2)}
    large_request = {"alpha": 0.1, "components": (1, 2, 3)}

    assert (
        CoarseCandidateEstimator._get_args_hash(small_request)
        == CoarseCandidateEstimator._get_args_hash(large_request)
    )
    assert not CoarseCandidateEstimator._is_arg_match(
        small_request,
        large_request,
    )


def test_default_cache_request_match_checks_arguments_and_data():
    data = np.array([[1.0, 2.0]])
    estimator = VerboseAgnosticEstimator(alpha=0.1)
    estimator._set_attached_dataset(data)

    assert VerboseAgnosticEstimator._matches_cache_request(
        estimator,
        {"alpha": 0.1, "verbose": True},
        data.copy(),
    )
    assert not VerboseAgnosticEstimator._matches_cache_request(
        estimator,
        {"alpha": 0.2, "verbose": False},
        data,
    )
    assert not VerboseAgnosticEstimator._matches_cache_request(
        estimator,
        {"alpha": 0.1, "verbose": False},
        np.array([[1.0, 3.0]]),
    )


def test_get_or_create_is_atomic_across_threads():
    """Concurrent equivalent requests should resolve to one estimator."""
    SlowEstimator._reset_cache()
    data = np.array([[1.0, 2.0]])
    worker_count = 8
    start = Barrier(worker_count)

    def resolve_estimator() -> SlowEstimator:
        start.wait()
        return SlowEstimator.get_or_create(data, alpha=0.1)

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        estimators = list(executor.map(
            lambda _: resolve_estimator(),
            range(worker_count),
        ))

    assert all(estimator is estimators[0] for estimator in estimators[1:])


def test_cache_resolution_preserves_positional_only_parameters():
    """Resolved defaults should not convert positional-only values to kwargs."""
    PositionalSeedEstimator._reset_cache()
    data = np.array([[1.0, 2.0]])

    with settings.override(random_seed=59):
        implicit = PositionalSeedEstimator.get_or_create(data, 0.1)

    explicit = PositionalSeedEstimator.get_or_create(
        data,
        0.1,
        random_seed=59,
    )
    different = PositionalSeedEstimator.get_or_create(
        data,
        0.1,
        random_seed=61,
    )

    assert implicit.alpha == 0.1
    assert implicit.random_seed == 59
    assert explicit is implicit
    assert different is not implicit


def test_cache_retains_estimator_without_external_strong_reference():
    """The cache should own estimators after their requesting Statistic dies."""
    VerboseAgnosticEstimator._reset_cache()
    data = np.array([[1.0, 2.0]])
    estimator = VerboseAgnosticEstimator.get_or_create(data, alpha=0.1)
    estimator_reference = weakref.ref(estimator)

    del estimator
    gc.collect()

    retained = estimator_reference()
    assert retained is not None
    assert (
        VerboseAgnosticEstimator.get_or_create(data, alpha=0.1)
        is retained
    )


def test_cache_capacity_uses_current_settings_limit():
    """Insertion should evict the least-recently-used entry at the set limit."""
    VerboseAgnosticEstimator._reset_cache()
    data = np.array([[1.0, 2.0]])

    with settings.override(max_cache_results=2):
        first = VerboseAgnosticEstimator.get_or_create(data, alpha=0.1)
        second = VerboseAgnosticEstimator.get_or_create(data, alpha=0.2)
        third = VerboseAgnosticEstimator.get_or_create(data, alpha=0.3)

        cached_estimators = {
            estimator
            for bucket in VerboseAgnosticEstimator._get_cache().values()
            for estimator in bucket
        }

    assert len(cached_estimators) == 2
    assert first not in cached_estimators
    assert second in cached_estimators
    assert third in cached_estimators
