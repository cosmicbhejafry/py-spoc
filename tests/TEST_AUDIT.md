# Test-suite audit

Reviewed against the working tree on 2026-10-22. This is a snapshot of a library
under active refactoring, not a list of tests to skip.

## Running the suite

```powershell
& 'D:\Python-venvs\pyspoc-venv\Scripts\python.exe' -m pytest tests -q
```

Last full run: **602 passed, 67 failed, zero collection errors** (669 cases,
approximately 26 seconds; six dependency deprecation warnings).

No production modules were changed for this audit. Valid regression failures
remain ordinary failures: they have not been skipped, marked xfail, or rewritten
to expect broken behaviour. Import and constructor failures can prevent later
assertions in those individual tests from running; fixing the first failure may
therefore uncover additional problems.

## Outdated tests replaced or updated

- Exception imports now use `pyspoc._exceptions`; rank statistics are imported
  from their defining module rather than an unavailable eager re-export.
- Component test doubles implement the current abstract metadata-resource
  hooks. Metadata tests exercise `get_identifier()` / `get_tags()` and the
  missing-property policy, not removed constructor-time property checks.
- Component configuration uses `_set_config()`. Core constructor calls provide
  full name, short name, and labels. Constructor-depth cleanup checks instance
  storage rather than `hasattr()`, which also sees the class-level default.
- Core Statistic tests accept integer-valued array outputs and NumPy/embedded
  scalars; complex and incorrectly shaped outputs remain rejection cases.
  Input-validation tests explicitly enable `result_validation="both"`.
  Concrete cache test doubles implement the required output-validation hook.
- Estimator-cache tests use the current `_estimators` module and
  `max_estimator_cache_results`; seed-sensitive test doubles explicitly include
  the random-seed mixin.
- Pairwise and HSIC helper tests unpack `(matrix, exceptions)`. Added tests for
  default error collection in serial and concurrent execution, and retained the
  separate first-error propagation regression. HSIC executor stubs accept the
  current execution keywords and return the current tuple result.
- HSIC tests use the public `compute()` boundary for dimension selection and
  float32-appropriate comparison tolerances for its default downsampling.
  Bias is checked in constructor configuration rather than an obsolete
  instance-specific identifier suffix.
- Partial-correlation tests use supported pseudoinverse aliases, enum metadata,
  current constructor options, and explicit row-omission policies. Dimension
  and omission checks run through `compute()`, not the private numerical kernel.
- Graph tests target currently exported classes, numeric graph definitions,
  thresholding, topology, and constructor validation. Removed graph classes
  that have moved into general association statistics are no longer imported.
- Clustering tests use current module paths and small deterministic inputs;
  fitting is no longer done at import time. Importability, invalid parameters,
  fitted-centre/medoid similarity definitions, and configuration are tested.
- Orthogonal PCAE tests use the current model bottleneck protocol, training
  arguments, metadata accessors, and incremental explained-variance definition.
- Fractal tests use current scale keywords and density-estimation hooks. The
  fine-scale entropy regression no longer depends on removed saturation/grid
  shortcuts. Rank tests now actually instantiate the parametrized ranker, with
  deterministic random column scaling.
- Calculator traceback-location comparisons are case-insensitive on Windows.

## Remaining library issues exposed by valid tests

| Area | Failing cases | First observable issue(s) |
| --- | ---: | --- |
| Partial correlation | 5 | Covariance adapters forward unsupported sklearn keywords (`store_precision`, `assumed_centered`); YAML tag precedence hides instance SIGNED removal; `compute()` discards transposed/NaN-filtered data. |
| KMeans / KMedoids | 33 | `kmeans/statistics.py` imports nonexistent `KMeansSweepStatistic`. The parent package imports KMeans, so KMedoids imports fail too. Existing kernel-definition and edge-case coverage is retained behind this import blocker. |
| Orthogonal PCAE | 13 | The mixin calls a required-argument Statistic initializer without `dim`/`nan_policy`; estimator caching raises `NameError: CachedEstimatorMixin`; an unfitted model accessor reads an uninitialized `_model_`. |
| HSIC dimension selection | 1 | Public `compute(dim="n")` still supplies the original matrix to the numerical kernel. |
| Graph adapters | 7 | Several constructors pass `ensure_adjacency_type` where the base expects `enforce_adjacency_type`; DistributionGraph unconditionally asserts histograms exist even for Wasserstein. |
| Core output validation | 3 | SPSD semantic validation is named `validate_output` but the dispatcher calls `_validate_output`; `NumpyRealScalar` repeats `np.floating` and omits `np.integer`. |
| Config dependency errors | 2 | Module discovery catches every `ModuleNotFoundError`, including errors raised by dependencies of an existing module. |
| Execution / memory | 3 | Default memory-fraction path leaves `checked_fraction` unassigned; first-error handling reads unassigned `value` in `finally`, masking the original exception; barrier execution retains four prepared values where its documented bound is three. |

Clustering imports occur inside tests deliberately: an import regression should
be visible without aborting collection of the rest of the suite. This does not
work around or repair the broken library import.

## Collection isolation

`conftest.py` redirects Numba caches to `.pytest_cache/numba` unless the caller
already set `NUMBA_CACHE_DIR`. During the initial package import only, it blocks
the `oct2py` and `jpype` executable probes so ordinary unit-test collection does
not launch Octave or a JVM. Their original `sys.modules` entries are restored
immediately afterwards; the package's import-time dependency snapshot therefore
treats these two integrations as unavailable. Tests requiring those real
external integrations should use a separate integration-test bootstrap.
