"""Constructor-aware caching machinery shared by estimator classes.

The module provides :class:`CachedEstimatorMixin`, which records normalized
constructor arguments and uses them, together with an attached NumPy dataset,
to find equivalent estimator instances. Cache buckets contain weak references
so that caching does not by itself extend an estimator's lifetime.
"""

# TODO: Implement verbose warnings.

from __future__ import annotations

import numpy as np
import inspect
import sys

from typing import Any, ParamSpec, TypeVar, Union, Callable, Concatenate
from collections.abc import Hashable
from weakref import WeakSet
from datetime import datetime
from functools import wraps


_T_cache_enabled = TypeVar("_T_cache_enabled", bound="CachedEstimatorMixin")
_P = ParamSpec("_P")
_R = TypeVar("_R")

PYSPOC_CACHE_ATTRIBUTE = "_pyspoc_cache"
PYSPOC_CACHE_WRAPPED_ATTRIBUTE = "_pyspoc_cache_wrapped"
PYSPOC_INIT_ARGS_ATTRIBUTE = "_pyspoc_init_arguments"
PYSPOC_HASH_ATTRIBUTE = "_pyspoc_hash"
PYSPOC_LRU_ATTRIBUTE = "_pyspoc_lru"
PYSPOC_ATTACHED_DATA_ATTRIBUTE = "_pyspoc_attached_dataset"


class CachedEstimatorMixin:
    """Provide opt-in, constructor-aware caching for estimator instances.

    Every subclass constructor is wrapped automatically by
    :meth:`__init_subclass__`. The wrapper binds positional and keyword
    arguments to the subclass constructor signature, applies declared
    defaults, and stores the resulting mapping on the estimator. Consequently,
    equivalent calls expressed with different positional/keyword forms can be
    compared consistently.

    Hashable constructor arguments are combined into a preliminary lookup
    hash. Hash collisions do not imply estimator equivalence: candidates in a
    hash bucket are subsequently compared using their complete normalized
    argument mappings and attached NumPy arrays. Dataset comparison includes
    shape, dtype, and element values.

    Cache buckets are :class:`weakref.WeakSet` instances. A cached estimator is
    therefore removed automatically when it has no other strong references.
    Estimator instances must support weak references and hashing to be stored.

    Subclasses that need a more specialized definition of equivalence may use
    :meth:`_get_cached` as the initial candidate lookup and apply additional
    checks before returning an estimator.

    Attributes
    ----------
    _hash_arg_ignore_list : list[str], optional
        Constructor argument names excluded from preliminary hashing. Ignored
        arguments still participate in exact comparison. The default is an
        empty list.
    _arg_comparison_ignore_list : list[str], optional
        Constructor argument names excluded from both preliminary hashing and
        exact argument comparison. Use this for arguments, such as verbosity,
        that do not affect estimator equivalence. The default is an empty list.
    _cache_limit : int, optional
        Maximum number of live estimator instances across all cache buckets.
        The default is ``10``.
    _pyspoc_cache : dict[int, weakref.WeakSet[CachedEstimatorMixin]], optional
        Mapping from preliminary argument hashes to weak candidate sets. The
        default is an empty dictionary.

    Notes
    -----
    Calling a subclass constructor directly always creates a new instance.
    Call :meth:`get_or_create` when cache resolution is required.
    """

    _hash_arg_ignore_list = []
    _arg_comparison_ignore_list = []
    _cache_limit = 10
    _pyspoc_cache = dict()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Wrap a subclass constructor to record normalized arguments.

        Parameters
        ----------
        **kwargs : Any, optional
            Keyword arguments accepted by other classes participating in
            subclass initialization. The default is no keyword arguments.

        Returns
        -------
        None
            This method modifies the subclass constructor in place and does
            not return a value.
        """
        super().__init_subclass__(**kwargs)

        original_init = cls.__init__

        # Avoid wrapping the same __init__ repeatedly.
        if getattr(original_init, PYSPOC_CACHE_WRAPPED_ATTRIBUTE, False):
            return

        @wraps(original_init)
        def wrapped_init(self: CachedEstimatorMixin, *args: Any, **kwargs: Any) -> None:
            arguments = self._bind_init_args(args, kwargs)

            # Store normalized constructor arguments for later exact comparison.
            setattr(self, PYSPOC_INIT_ARGS_ATTRIBUTE, arguments)

            # Store a hashable key for inexpensive preliminary lookup.
            estimator_hash = cls._get_args_hash(arguments)
            self._set_hash(estimator_hash)

            original_init(self, *args, **kwargs)

        setattr(wrapped_init, PYSPOC_CACHE_WRAPPED_ATTRIBUTE, True)
        cls.__init__ = wrapped_init


    @classmethod
    def _get_args_hash(cls, kwargs: dict[str, Any]) -> int:
        """Create a preliminary cache key from hashable constructor arguments.

        Parameters
        ----------
        kwargs : dict[str, Any]
            Normalized constructor argument mapping.

        Returns
        -------
        int
            Hash of the retained argument values. This hash selects candidate
            estimators only and is not treated as proof of equivalence.
        """
        hash_args = cls._get_hashables(kwargs)
        return hash(tuple(hash_args.values()))


    @classmethod
    def _bind_init_args(cls, args, kwargs) -> dict[str, Any]:
        """Normalize arguments against the subclass constructor signature.

        Parameters
        ----------
        args : tuple[Any, ...]
            Positional arguments intended for the subclass constructor.
        kwargs : dict[str, Any]
            Keyword arguments intended for the subclass constructor.

        Returns
        -------
        dict[str, Any]
            Mapping from constructor parameter names to supplied or default
            values, excluding ``self``.

        Raises
        ------
        TypeError
            If the supplied arguments cannot be bound to the constructor.
        """
        init = cls.__init__
        init_signature = inspect.signature(init)
        bound = init_signature.bind(None, *args, **kwargs)
        bound.apply_defaults()

        arguments = dict(bound.arguments)
        arguments.pop("self", None)

        return arguments
    

    @classmethod
    def _get_cache(cls) -> dict[int, WeakSet[CachedEstimatorMixin]]:
        """Return the cache associated with the estimator subclass.

        Returns
        -------
        dict[int, weakref.WeakSet[CachedEstimatorMixin]]
            Mapping of constructor hashes to weak candidate sets. A new empty
            cache is installed if the stored cache is missing or invalid.
        """
        cache = getattr(cls, PYSPOC_CACHE_ATTRIBUTE, None)

        if not isinstance(cache, dict):
            cache = cls._reset_cache()

        return cache

    @classmethod
    def _update_cache(cls, estimator: CachedEstimatorMixin
        ) -> dict[int, WeakSet[CachedEstimatorMixin]] | None:

        """Add an estimator to the candidate bucket for a hash.

        Parameters
        ----------
        hash : int
            Preliminary constructor-argument hash identifying the bucket.
        estimator : CachedEstimatorMixin
            Live estimator instance to store weakly.

        Returns
        -------
        None
            The class cache is updated in place.
        """
        hash = estimator._get_hash()

        if hash is None:
            return
        
        cache = cls._get_cache()
        estimators = cache.get(hash)

        # Create a weak bucket for the first estimator with this hash. WeakSet
        # requires its contents to support weak references and hashing.
        if not isinstance(estimators, WeakSet):
            cache[hash] = WeakSet([estimator])
        else:
            cache[hash].add(estimator)

        return cache

            
    @classmethod
    def _reset_cache(cls) -> dict[int, WeakSet[CachedEstimatorMixin]]:
        """Replace the subclass cache with an empty mapping.

        Returns
        -------
        dict[int, weakref.WeakSet[CachedEstimatorMixin]]
            Newly installed empty cache.
        """
        cache = dict()
        setattr(cls, PYSPOC_CACHE_ATTRIBUTE, cache)
        return cache


    def _get_init_args(self) -> dict[str, Any]:
        """Return normalized arguments recorded during construction.

        Returns
        -------
        dict[str, Any]
            Recorded constructor arguments, or an empty dictionary if argument
            recording did not occur.
        """
        return getattr(self, PYSPOC_INIT_ARGS_ATTRIBUTE, dict())
    

    def _get_hash(self) -> Union[int, None]:
        """Return the estimator's preliminary constructor hash.

        Returns
        -------
        int or None
            Recorded hash, or ``None`` if no hash has been assigned.
        """
        return getattr(self, PYSPOC_HASH_ATTRIBUTE, None)


    def _set_hash(self, hash: int):
        """Record the estimator's preliminary constructor hash.

        Parameters
        ----------
        hash : int
            Hash produced from the estimator's normalized, hashable arguments.

        Returns
        -------
        None
            The hash is stored on the estimator in place.
        """
        setattr(self, PYSPOC_HASH_ATTRIBUTE, hash)


    def _get_attached_dataset(self) -> Union[np.ndarray, None]:
        """Return the dataset associated with the estimator.

        Returns
        -------
        numpy.ndarray or None
            Attached dataset, or ``None`` when no dataset has been recorded.
        """
        return getattr(self, PYSPOC_ATTACHED_DATA_ATTRIBUTE, None)
    

    def _set_attached_dataset(self, dataset: np.ndarray):
        """Attach a dataset used for exact cache matching.

        Parameters
        ----------
        dataset : numpy.ndarray
            Dataset to associate with this estimator. The array is stored by
            reference; it is not copied.

        Returns
        -------
        None
            The dataset is stored on the estimator in place.
        """
        setattr(self, PYSPOC_ATTACHED_DATA_ATTRIBUTE, dataset)


    def _get_lru(self) -> float:
        """Return the estimator's last-use time as a POSIX timestamp.

        Returns
        -------
        float
            Recorded last-use timestamp, or ``0.0`` if no valid
            :class:`datetime.datetime` has been stored.
        """
        lru = getattr(self, PYSPOC_LRU_ATTRIBUTE, None)
        
        if not isinstance(lru, datetime):
            return 0.0

        return lru.timestamp()

    def _set_lru(self, lru: datetime):
        """Record when the estimator was last used.

        Parameters
        ----------
        lru : datetime.datetime
            Date and time of the estimator's most recent use.

        Returns
        -------
        None
            The timestamp is stored on the estimator in place.
        """
        setattr(self, PYSPOC_LRU_ATTRIBUTE, lru)
        

    @classmethod
    def get_or_create(
            cls: type[_T_cache_enabled],
            data: np.ndarray,
            *args,
            **kwargs) -> _T_cache_enabled:
        """Return an equivalent cached estimator or construct a new one.

        Parameters
        ----------
        data : numpy.ndarray
            Dataset against which cached candidates are compared.
        *args : Any, optional
            Positional arguments forwarded to the subclass constructor. The
            default is no positional arguments.
        **kwargs : Any, optional
            Keyword arguments forwarded to the subclass constructor. The
            default is no keyword arguments.

        Returns
        -------
        T_cache_enabled
            Existing matching instance when available; otherwise a newly
            constructed and cached instance of ``cls``.
        """
        # Use the dataset and normalized constructor call to find candidates.
        cached = cls._get_cached(data, args, kwargs)

        # Normal construction cannot replace an instance, so cache resolution
        # belongs in this factory method rather than in ``__init__``.
        if not cached:
            return cls._instantiate(data, args, kwargs)

        # The base implementation treats every exact match as interchangeable.
        return cached[0]

    @classmethod
    def _get_cached(cls: type[_T_cache_enabled],
                    data: np.ndarray,
                    args,
                    kwargs) -> Union[list[_T_cache_enabled], None]:
        """Return all candidates matching the arguments and dataset.

        The constructor hash is only a preliminary filter. Each candidate must
        also have the same concrete subclass, normalized arguments, array
        shape, dtype, and values. ``None`` indicates that no candidate matched.

        Parameters
        ----------
        data : numpy.ndarray
            Dataset to compare with each candidate's attached dataset.
        args : tuple[Any, ...]
            Positional arguments intended for the subclass constructor.
        kwargs : dict[str, Any]
            Keyword arguments intended for the subclass constructor.

        Returns
        -------
        list[T_cache_enabled] or None
            Live estimators satisfying every comparison, or ``None`` when the
            hash bucket is absent or contains no exact matches.
        """

        # Normalize the requested call exactly as it would be normalized when
        # a new estimator is constructed.
        estimator_args = cls._bind_init_args(args, kwargs)
        estimator_hash = cls._get_args_hash(estimator_args)
        cache = cls._get_cache()

        # Hash lookup cheaply reduces the number of estimators that require
        # potentially expensive argument and array comparisons.
        objs = cache.get(estimator_hash)

        if objs is None:
            return

        matched_objs = list()

        # WeakSet iteration yields only estimators that are still alive.
        for obj in objs:
            # Buckets can be inherited or shared, so enforce the requested
            # estimator class before performing detailed comparisons.
            if not isinstance(obj, cls):
                continue

            # Delegate detailed equivalence to an overridable hook so
            # subclasses can implement asymmetric or model-specific matching.
            if not cls._matches_cache_request(obj, estimator_args, data):
                continue

            matched_objs.append(obj)

        if not matched_objs:
            return

        return matched_objs


    @classmethod
    def _matches_cache_request(
            cls,
            estimator: CachedEstimatorMixin,
            estimator_args: dict[str, Any],
            data: np.ndarray) -> bool:
        """Return whether an estimator satisfies a complete cache request.

        The default implementation requires matching normalized constructor
        arguments and an exactly matching attached dataset. Subclasses may
        override this hook for model-specific rules, including asymmetric
        relationships such as a cached component superset satisfying a subset
        request.

        Parameters
        ----------
        estimator : CachedEstimatorMixin
            Cached estimator candidate selected by the preliminary hash.
        estimator_args : dict[str, Any]
            Normalized constructor arguments for the current request.
        data : numpy.ndarray
            Dataset supplied with the current request.

        Returns
        -------
        bool
            ``True`` if the candidate satisfies the request; otherwise
            ``False``.

        Notes
        -----
        Arguments handled specially by an override must also be excluded from
        preliminary hashing through ``_hash_arg_ignore_list`` so requests with
        differing values can reach this hook.
        """
        # The hash is only a candidate filter: collisions and omitted or
        # unhashable values can place unequal requests in the same bucket.
        if not cls._is_arg_match(estimator._get_init_args(), estimator_args):
            return False

        return cls._is_data_match(estimator, data)


    @classmethod
    def _is_data_match(
            cls,
            estimator: CachedEstimatorMixin,
            data: np.ndarray) -> bool:
        """Compare a cached estimator's attached dataset with request data.

        Parameters
        ----------
        estimator : CachedEstimatorMixin
            Cached estimator candidate whose attached dataset is compared.
        data : numpy.ndarray
            Dataset supplied with the current cache request.

        Returns
        -------
        bool
            ``True`` when shape, dtype, and element values match exactly;
            otherwise ``False``.
        """
        attached_dataset = estimator._get_attached_dataset()

        if attached_dataset is None:
            return False

        # Check inexpensive metadata before comparing every element.
        if attached_dataset.shape != data.shape:
            return False

        if attached_dataset.dtype != data.dtype:
            return False

        return np.array_equal(attached_dataset, data)


    @classmethod
    def _is_arg_value_match(cls, arg: Any, reference_arg: Any) -> bool:
        """Compare two individual constructor argument values safely.

        Identity is checked first so that an object always matches itself,
        including objects whose equality operation is unsupported or
        ambiguous. NumPy arrays require explicit array comparison because
        their equality operator returns an array of booleans. Other values use
        their normal equality operation only when it produces a scalar Boolean
        result.

        Parameters
        ----------
        arg : Any
            Constructor argument recorded on a cached estimator.
        reference_arg : Any
            Constructor argument supplied with the current cache request.

        Returns
        -------
        bool
            ``True`` when the values are unambiguously equal; otherwise
            ``False``.

        Notes
        -----
        The conservative ``False`` result for ambiguous or unsupported
        equality prevents the cache from returning an incorrect estimator.
        Subclasses may override this method to support additional value types.
        """
        # Identity is definitive and avoids invoking potentially problematic
        # equality implementations when both arguments are the same object.
        if arg is reference_arg:
            return True

        # NumPy equality is element-wise, so compare arrays explicitly and
        # include shape in the equivalence check.
        if isinstance(arg, np.ndarray) and isinstance(reference_arg, np.ndarray):
            return np.array_equal(arg, reference_arg)

        # Tensor equality is element-wise, so compare arrays explicitly and
        # include shape and type in the equivalence check.
        # If torch isn't loaded, ignore the check entirely.
        tensors_equal = cls._are_tensors_equal(arg, reference_arg)

        if tensors_equal is not None:
            return tensors_equal

        try:
            is_equal = arg == reference_arg
        except Exception:
            # Equality is user-defined and may reject cross-type comparisons.
            # A cache miss is safer than propagating or guessing in that case.
            return False

        # Do not coerce array-like equality results to bool: NumPy arrays,
        # tensors, and similar objects may be ambiguous or apply implicit
        # aggregation rules that are unsuitable for cache equivalence.
        if isinstance(is_equal, (bool, np.bool_)):
            return bool(is_equal)

        return False
    

    @staticmethod
    def _are_tensors_equal(arg: Any, reference_arg: Any) -> bool | None:
        """
        Return tensor equality, or None when Torch comparison is inapplicable.
        
        Parameters
        ----------
        arg : Any
            Constructor argument recorded on a cached estimator.
        reference_arg : Any
            Constructor argument supplied with the current cache request.

        Returns
        -------
        bool | None
            ``None`` if torch is not loaded;
            ``True`` when the values are unambiguously equal; otherwise
            ``False``.

        Notes
        -----
        The ``None`` result allows for checks to continue in the main check
        method, signalling that the arguments are not tensors.
        Subclasses may override this method to support additional checks,
        for example to include checking on device match.
        """
        torch = sys.modules.get("torch")

        if torch is None:
            return None

        if not (
            isinstance(arg, torch.Tensor)
            and isinstance(reference_arg, torch.Tensor)
        ):
            return None

        return (
            arg.shape == reference_arg.shape
            and arg.dtype == reference_arg.dtype
            #and arg.device == reference_arg.device
            and torch.equal(arg, reference_arg)
        )


    @classmethod
    def _is_arg_match(
            cls,
            args: dict[str, Any],
            reference_args: dict[str, Any]) -> bool:
        
        """
        Compare two complete normalized constructor mappings.

        Parameters
        ----------
        args : dict[str, Any]
            Constructor arguments recorded on a cached estimator.
        reference_args : dict[str, Any]
            Constructor arguments for the current cache request.

        Returns
        -------
        bool
            ``True`` if both mappings contain the same names and equal values;
            otherwise ``False``.
        """
        # A differing parameter set cannot describe the same constructor call.
        ignored_args = set(cls._arg_comparison_ignore_list)
        args_keys = set(args).difference(ignored_args)
        reference_keys = set(reference_args).difference(ignored_args)

        if args_keys != reference_keys:
            return False

        for arg_name, arg in args.items():
            if arg_name in ignored_args:
                continue

            reference_arg = reference_args[arg_name]

            if not cls._is_arg_value_match(arg, reference_arg):
                return False

        return True

    @classmethod
    def _get_hashables(cls, estimator_kwargs: dict[str, Any]):
        """Select arguments suitable for preliminary cache hashing.

        Parameters
        ----------
        estimator_kwargs : dict[str, Any]
            Complete normalized constructor argument mapping.

        Returns
        -------
        dict[str, Any]
            Ordered subset containing values that are not explicitly ignored
            and can be hashed successfully.
        """
        hashable_kwargs = dict()

        for arg_name, arg in estimator_kwargs.items():
            # Subclasses may omit expensive or intentionally variable values
            # from the preliminary hash without omitting exact comparison.
            if (
                arg_name in cls._hash_arg_ignore_list
                or arg_name in cls._arg_comparison_ignore_list
            ):
                continue

            # The protocol check is a cheap first pass; the actual hash call
            # below handles objects that advertise hashing but reject it.
            if not isinstance(arg, Hashable):
                continue

            try:
                hash(arg)
            except TypeError:
                continue

            hashable_kwargs[arg_name] = arg

        return hashable_kwargs


    @classmethod
    def _instantiate(
            cls: type[_T_cache_enabled],
            data: np.ndarray,
            args,
            kwargs) -> _T_cache_enabled:
        """Construct an estimator and register it with the bounded cache.

        Parameters
        ----------
        args : tuple[Any, ...]
            Positional arguments forwarded to the subclass constructor.
        kwargs : dict[str, Any]
            Keyword arguments forwarded to the subclass constructor.

        Returns
        -------
        T_cache_enabled
            Newly constructed and cached estimator instance.
        """
        # Construction invokes the wrapper installed by ``__init_subclass__``,
        # which records normalized arguments and their preliminary hash.
        estimator = cls(*args, **kwargs)
        estimator._set_attached_dataset(data)
        estimator._set_lru(datetime.now())
        cls._maintain_cache(estimator)
        return estimator


    @classmethod
    def _maintain_cache(cls, estimator: CachedEstimatorMixin):
        """Insert an estimator and evict excess live cache entries.

        Parameters
        ----------
        estimator : CachedEstimatorMixin
            Newly constructed estimator to add to its argument-hash bucket.

        Returns
        -------
        None
            The cache is updated in place. At most ``_cache_limit`` live
            estimator references remain after maintenance.

        Raises
        ------
        RuntimeError
            If an estimator selected for eviction has no corresponding cache
            bucket, indicating that the cache changed unexpectedly.
        """
        size = 0
        cached_estimators: list[CachedEstimatorMixin] = list()
        cache = cls._update_cache(estimator)

        if cache is None:
            raise RuntimeError("Estimator hash not found during cache maintenance.")

        # Materialize the currently live objects so their total can be bounded
        # and they remain alive for the duration of this maintenance pass.
        for cached_hash, estimators in tuple(cache.items()):
            n_est = len(estimators)

            if n_est == 0:
                cache.pop(cached_hash, None)
                continue

            cached_estimators.extend(estimators)
            size += n_est

        excess = size - cls._cache_limit

        if excess <= 0:
            return

        # Newest estimators sort first; removal proceeds from the oldest end.
        cached_estimators.sort(
            key=lambda x: x._get_lru(),
            reverse=True)

        for i in range(excess):
            lru_estimator = cached_estimators.pop()
            lru_hash = lru_estimator._get_hash()

            if lru_hash is None:
                continue

            estimator_set = cache.get(lru_hash)

            if estimator_set is None:
                raise RuntimeError(f"Estimator cache for hash {lru_hash} "
                                   "not found during cache maintenance.")

            # WeakSet removal uses the estimator's hash/equality semantics.
            estimator_set.remove(lru_estimator)

            if not estimator_set:
                cache.pop(lru_hash, None)

    @staticmethod
    def _updates_lru(func: Callable[Concatenate[_T_cache_enabled, _P], _R]
        ) -> Callable[Concatenate[_T_cache_enabled, _P], _R]:
        """
        Decorate an instance method to update its estimator's use time.

        Parameters
        ----------
        func : collections.abc.Callable
            Instance method whose invocation counts as estimator use.

        Returns
        -------
        collections.abc.Callable
            Wrapper that records the current local date and time immediately
            before calling ``func``.
        """
        @wraps(func)
        def wrapper(self: _T_cache_enabled, /, *args: _P.args, **kwargs: _P.kwargs) -> _R:
            # Update recency before executing the wrapped operation so cache
            # maintenance can prefer recently requested estimators.
            self._set_lru(datetime.now())
            return func(self, *args, **kwargs)

        return wrapper
