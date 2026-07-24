from __future__ import annotations

import numpy as np

from pyspoc._caching._cls import CachedEstimatorMixin


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
