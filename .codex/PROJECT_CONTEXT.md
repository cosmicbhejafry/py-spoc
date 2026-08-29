\# pySPoC project context and development handoff



Last updated: 2026-08-08



This document is intended primarily for Codex and other coding agents working

on pySPoC. Read it before making architectural changes. It records the project

purpose, design decisions made with the maintainer, completed work, and the

current unfinished refactor. The repository remains the source of truth when

it differs from this document.



\## Project purpose



pySPoC means "Python Statistics for Point Clouds". It is an under-development

framework for computing many statistical summaries of static datasets. Its

main architectural idea is to separate:



1\. A `Statistic`, which computes an intermediate scalar, vector, or matrix

&#x20;  from a dataset.

2\. A `Reducer`, which summarizes a compatible Statistic result.



This permits reusable combinations. For example, one covariance matrix can be

reduced using eigenvalues, norms, a determinant, and other matrix reducers.

Historically, `ReducedStatistic` represented computations that produced their

own final scalar result without a separate Reducer. This area is now being

redesigned around more precise Statistic result contracts.



Users are expected to interact mainly through:



\- `Dataset`: validated/wrapped user data.

\- `Config`: a collection of requested Statistic/Reducer pipelines, commonly

&#x20; loaded from YAML.

\- `Calculator`: executes those pipelines over data.

\- Results: ultimately presented as tabular statistical summaries.



The long-term use case includes running many related Statistics, potentially

across threads. Multiple Statistics may depend on the same expensive fitted

estimator, so estimator reuse, deterministic random-number generation, and

thread safety are important cross-cutting concerns.



\## Maintainer preferences and working conventions



\- Use NumPy-style docstrings. Document parameter type, default, optionality,

&#x20; return type, raised errors, and non-obvious notes.

\- Add meaningful comments for researchers and engineers, particularly around

&#x20; numerical algorithms and lifecycle orchestration.

\- Explicitly call attention in comments/docstrings when a method overrides an

&#x20; inherited or abstract hook.

\- Prefer classes near the top of a module and supporting functions below them

&#x20; when practical.

\- A leading underscore on a package/module communicates that the area is

&#x20; private, but internal objects may still use underscores where it clarifies

&#x20; their intended scope.

\- Components are intended to be largely immutable after initialization.

&#x20; Read-only properties are preferred over publicly mutable fields. Return

&#x20; tuples or copies where exposing a mutable internal collection would be

&#x20; unsafe.

\- Model objects should normally remain private. Python privacy is advisory;

&#x20; the objective is a clear supported API rather than making access impossible.

\- Constructor APIs should remain explicit. A concrete object that supports a

&#x20; random-seed override should explicitly expose `random\_seed` in `\_\_init\_\_`.

\- Commit messages should use hierarchical bullet points grouped by topic, not

&#x20; a single prose paragraph.

\- Preserve unrelated worktree changes. Do not assume an untracked or modified

&#x20; file is disposable.



\## Important architectural work completed



\### Settings



`pyspoc/settings.py` now contains:



\- Immutable, slotted `SettingsValues` snapshots.

\- A package-wide `Settings` manager exposed as `settings`.

\- Permanent updates through `settings.configure(...)`.

\- Temporary, nestable overrides through:



&#x20; ```python

&#x20; with settings.override(verbose=True, numba\_mode="python"):

&#x20;     ...

&#x20; ```



\- Context-local overrides implemented with `ContextVar`, making asynchronous

&#x20; contexts safer. Thread-pool work may need explicit `copy\_context()`

&#x20; propagation later.



Consumers should always read `settings.current.<field>`. Current settings

include verbosity, cache limits and switches, random seed, PyTorch inference

device behavior, and Numba execution/compilation policies.



\### Runtime and semantic argument checking



`pyspoc/\_argchecking.py` replaced the earlier `\_typechecking` module and has a

wider remit:



\- `RuntimeTypeCheckedMixin` automatically wraps subclass constructors and

&#x20; configured methods/properties using Typeguard.

\- It supports public/all/explicit method-selection policies, return checking,

&#x20; properties, variadic arguments, collection strategies, async functions, and

&#x20; signature preservation with `ParamSpec`/`wraps`.

\- Type hints are resolved lazily with `get\_type\_hints()`.

\- Semantic helpers validate and optionally clip numeric constructor values.

\- Informational clipping messages use logging rather than warnings where

&#x20; appropriate.



The mixin performs structural type checks only. Range checks, matrix

properties, and relationships between values remain explicit semantic checks.



\### Component constructor capture



The new `pyspoc/\_core/component.py` contains the in-progress replacement

`Component`. It automatically captures normalized initialization arguments:



\- `\_\_init\_subclass\_\_` wraps directly declared constructors.

\- `inspect.signature(...).bind(...)` and `apply\_defaults()` normalize calls.

\- A per-instance depth counter prevents cooperative parent constructors from

&#x20; overwriting the outermost public constructor arguments.

\- Arguments are recorded only after successful initialization.

\- `params` exposes a read-only `MappingProxyType` view.



Tests for this mechanism were added in `tests/test\_core\_component.py`.



\### Estimator caching and fitting lifecycle



Shared estimator infrastructure lives in the private `pyspoc/\_estimators`

package because it now covers both caching and fitting/thread safety:



\- `CachedEstimatorMixin` records normalized constructor arguments and supports

&#x20; positional, keyword, positional-only, variadic, and defaulted parameters.

\- `get\_or\_create(data, \*args, \*\*kwargs)` performs cache lookup or construction;

&#x20; direct construction always creates a new object.

\- Cache identity uses a candidate hash followed by exact argument and dataset

&#x20; comparison, so hash collisions do not establish equivalence.

\- `\_canonicalize\_cache\_args()` can normalize/remove arguments from ordinary

&#x20; equivalence.

\- `\_get\_candidate\_args()` can deliberately create a coarser candidate bucket

&#x20; before estimator-specific `\_matches\_cache\_request()` logic.

\- Subclass resolution hooks cooperate through the MRO.

\- Estimator caches now use strong `set` references, not weak references. Weak

&#x20; references were evicted immediately when sequential Calculator Statistics

&#x20; replaced their only strong reference, defeating estimator reuse.

\- Cache access/mutation is protected by an `RLock`.

\- Bounded LRU eviction uses `settings.current.max\_cache\_results`.

\- `\_updates\_lru` marks estimator use, not merely creation.

\- `LazyFittedCachedEstimatorMixin` supplies synchronized fit-on-first-use with

&#x20; double-checked locking, attached-data validation, and an abstract

&#x20; `\_fit\_estimator()` hook.



Time-based cache eviction was explicitly deferred. Statistics using the same

estimator may not execute adjacently, so retaining the cache until size-based

eviction is currently intentional.



There is a longer-term plan for a separate Component-result cache. A future

user-facing clear-cache operation should cascade through Component, Reducer,

and estimator cache layers.



\### Random-number generation



Shared RNG behavior lives in `pyspoc/\_random`:



\- `RandomSeedMixin` resolves explicit seeds against

&#x20; `settings.current.random\_seed`.

\- Cached estimators may set `\_freeze\_random\_seed = True`, resolving the seed

&#x20; before cache lookup so the effective seed is stable and participates in

&#x20; cache identity.

\- Statistics may retain dynamic behavior and observe the active settings

&#x20; context at computation time.

\- Helpers create local NumPy and PyTorch generators.

\- PyTorch is imported lazily by the generator helper so it remains an optional

&#x20; dependency.

\- OrthogonalPCAE uses separate deterministic generator streams for DataLoader

&#x20; shuffling and model-side stochastic operations.



Do not rely on PyTorch's process-global RNG for estimator internals when a

local generator can be supplied. Optimizers such as Adam are deterministic

given gradients; stochasticity usually comes from initialization, randomized

model operations, and data ordering.



\### Optional dependencies



`pyspoc.exceptions.OptionalDependencyMissingError` is a public `ImportError`

subclass used when a requested optional feature lacks its dependency. It

records the dependency, feature, and installation hint so callers can handle

this case higher in the execution chain.



\### Orthogonal PCA autoencoder family



The OrthogonalPCAE implementation is under

`pyspoc/statistics/dimreduce/orthopcae`, with concrete reduced statistics under

`pyspoc/rstatistics/dimreduce/orthopcae`.



Major completed design work includes:



\- Separation of model (`\_model.py`), cached/lazy estimator (`\_estimator.py`),

&#x20; shared Statistic machinery (`\_mixin.py`/`\_base.py`), state, and concrete

&#x20; Statistics.

\- The old `\_module.py` name was changed to `\_model.py`; tests/imports were

&#x20; updated at the time.

\- Components selection belongs at the Statistic level. The estimator needs a

&#x20; sufficiently large `max\_bottleneck\_dim`; downstream Statistics select

&#x20; subsets of ordered latents.

\- Estimator equivalence currently requires equal effective

&#x20; `max\_bottleneck\_dim` rather than treating larger bottlenecks as compatible.

\- Thread-safe training and fitted-data validation are delegated to the shared

&#x20; lazy-fitted estimator lifecycle.

\- Burn-in applies only at the start: cumulative epoch count prevents it from

&#x20; being repeated during explicit additional training.

\- Model and DataLoader RNG streams are independent.

\- Training/inference device behavior and returning cached models to CPU were

&#x20; considered; `settings.current.torch\_estimator\_inference\_device` records the

&#x20; policy.

\- Private estimator state is exposed selectively through read-only properties;

&#x20; mutable histories return tuples/copies.

\- Estimator/model access remains private-supported API. Statistic authors may

&#x20; borrow fitted model state internally, but users should not mutate it.

\- Extensive tests exist under

&#x20; `tests/statistics/dimreduce/orthopcae` and for estimator caching.



\### K-based clustering family



The clustering family is now generalized toward K-Means and K-Medoids under

`pyspoc/statistics/clustering/kmeans`. The shared class name is

`KClusteringEstimator`; `KClustering` was preferred over an overly broad

general clustering abstraction because the APIs remain K-based.



Completed work includes:



\- Cached, lazy fitted scikit-learn/scikit-learn-extra estimator integration.

\- Library/global random seed with per-object override.

\- A concrete `KClusteringSimilarity` Statistic producing pairwise cluster

&#x20; center similarities.

\- Supported similarities include RBF, Laplacian, inverse-distance, cosine,

&#x20; correlation, and Mahalanobis variants.

\- Mahalanobis similarities whiten centers using an SPSD pseudo-inverse square

&#x20; root, intentionally supporting singular covariance matrices by ignoring the

&#x20; numerical null space.

\- Degenerate correlation cases are handled explicitly. The philosophical

&#x20; interpretation of zero/constant centered vectors was discussed, but no

&#x20; implementation change was intended.

\- Detailed documentation and tests, including edge cases, exist in

&#x20; `tests/statistics/clustering/kmeans/test\_cluster\_similarity.py`.



\### Numba dispatch and fallback behavior



Reusable Numba integration moved from the fractal area to `pyspoc/\_numba`.

The dispatch design supports:



\- `numba\_mode="auto"`: try Numba and allow Python fallback.

\- `numba\_mode="numba"`: require the compiled implementation and expose

&#x20; failures.

\- `numba\_mode="python"`: bypass Numba.

\- An explicitly supplied Python fallback, rather than always using the Python

&#x20; body of the decorated Numba function. This is useful for functions such as

&#x20; array equality, whose best Python fallback is `numpy.array\_equal`.



Names were clarified to distinguish the Python implementation/fallback from

the wrapper that performs fallback dispatch. Renyi functions continue to use

the dispatch wrapper without module-reload tricks.



\### Rényi/fractal work



The Rényi entropy/fractal-dimension implementation and tests were substantially

reworked:



\- Runtime argument checking was added to the legacy fractal classes.

\- Duplicate interval-handling logic was removed in favor of shared semantic

&#x20; argument checks.

\- Box counting stops before integer box identifiers can overflow: refinement

&#x20; terminates when occupied boxes reach the sample count and at the opposite

&#x20; saturation end when all points occupy one box.

\- Tests comparing generated fractal datasets against expected dimensions were

&#x20; restored; q=0 is expected to remain within stated bounds for all generators.

\- Elbow detection is now part of scaling-region selection and minimum retained

&#x20; region requirements were introduced.

\- `minimum\_scaling\_points` and `scale\_length` accept `None` for data-dependent

&#x20; defaults.

\- Scale length uses dimensionality, currently documented as approximately

&#x20; `max(50, ceil(100 \* log(p)))`, to compensate for degraded high-dimensional

&#x20; estimates.

\- The Renyi test workload was reduced while preserving meaningful backend and

&#x20; fractal-dimension coverage.



\## Current active work: rebuilding the core Statistic hierarchy



This is the most important unfinished area. The maintainer is rebuilding from

the base `Component` upward rather than incrementally modifying the legacy

`pyspoc/\_base.py` hierarchy.



Intended hierarchy:



```text

Statistic

├── ScalarStatistic

├── VectorStatistic

└── MatrixStatistic

&#x20;   └── SquareMatrixStatistic

&#x20;       └── SPSDMatrixStatistic

```



The purpose is to make Reducer compatibility explicit from the Statistic

result contract. Public `compute()` remains implemented by concrete

Statistics. The inherited calculation/orchestration path must validate its

result afterward so correctness does not depend on every concrete author

copying an annotation accurately.



\### Current type aliases



`pyspoc/\_core/types.py` currently defines jaxtyping aliases for:



\- `NumpyDataMatrix`

\- `NumpyRealMatrix`

\- `NumpyRealSquareMatrix`

\- `NumpyRealSPSDMatrix`

\- `NumpyRealVector`

\- `NumpyRealTensorAtMost2D`



Jaxtyping is being used for documentation and individual runtime structural

checks. Typeguard's `check\_type()` triggers jaxtyping's `ndarray`, dtype, rank,

and repeated-axis checks. For example:



```python

check\_type(result, Real\[ndarray, "size size"])

```



checks that `result` is a real NumPy array with exactly two equally sized

dimensions.



Important limitations/decisions:



\- Installed versions at the time of discussion were jaxtyping 0.3.11 and

&#x20; Typeguard 4.5.2.

\- Jaxtyping's `jaxtyped` is a decorator in this version, not a public context

&#x20; manager.

\- Jaxtyping documentation warns about compatibility of its Typeguard decorator

&#x20; integration with Typeguard 3/4. Avoid making the hierarchy depend on

&#x20; `@jaxtyped(typechecker=typechecked)` without dedicated compatibility tests.

\- Independent `check\_type()` calls do not preserve symbolic dimension

&#x20; bindings between `data` and `result`. This was tested locally: a composite

&#x20; tuple check still accepted an unrelated square result size.

\- Therefore cross-object relationships must be checked manually. If data has

&#x20; `(rows, columns)`, a SquareMatrix result intended to inherit a data dimension

&#x20; should explicitly require either `(rows, rows)` or `(columns, columns)`.

\- A repeated name within one jaxtyping annotation, such as `"size size"`, is

&#x20; reliable for enforcing that the result itself is square.

\- Ruff interprets strings in explicit `TypeAlias` declarations as forward

&#x20; references and reports `F821` for names such as `"m"`. Plain implicit aliases

&#x20; avoid this. If explicit `TypeAlias` is retained, use a narrowly scoped

&#x20; `# noqa: F821` rather than disabling undefined-name checking broadly.

\- Pylance generally displays a type alias name rather than expanding its full

&#x20; meaning in IntelliSense. Prefer descriptive alias names and detailed public

&#x20; docstrings; inline public annotations remain an option when showing the

&#x20; shape is more important than brevity.



\### Override variance decision



Overrides may narrow a return type but must not narrow an accepted parameter

type. Thus this is appropriate:



```text

Statistic.compute(...) -> scalar | vector | matrix

MatrixStatistic.compute(...) -> matrix

SquareMatrixStatistic.compute(...) -> square matrix

```



But a `\_get\_validated\_result(result)` override cannot safely narrow the type of

the `result` input parameter under ordinary substitutability rules. A robust

design is to keep the validator input broad, perform a check, and narrow only

its return annotation. An exception on invalid data is compatible with the

narrowed successful return.



\### Known incomplete/inconsistent core state



Do not treat the current `\_core/statistic` files as finished. At this handoff:



1\. `pyspoc/\_\_init\_\_.py` still imports `pyspoc.\_core.base`, which no longer

&#x20;  exists. This currently prevents an ordinary package import and blocked a

&#x20;  runtime smoke test.

2\. Several legacy modules still import old `\_core.base` paths. Imports must be

&#x20;  migrated deliberately as the hierarchy stabilizes.

3\. `pyspoc/\_core/statistic/matrix/base.py` imports

&#x20;  `pyspoc.\_core.np\_types`, but the current alias module is

&#x20;  `pyspoc.\_core.types`.

4\. `Statistic.calculate()` currently calls `\_get\_validated\_result(result)`.

&#x20;  If validation must compare output dimensions with input data, change the

&#x20;  stable hook to `\_get\_validated\_result(data, result)` throughout the

&#x20;  hierarchy.

5\. `SquareMatrixStatistic` currently validates only that its output is square.

&#x20;  The proposed additional manual relationship is:



&#x20;  ```python

&#x20;  rows, columns = data.shape

&#x20;  if result.shape not in {(rows, rows), (columns, columns)}:

&#x20;      raise ValueError(...)

&#x20;  ```



&#x20;  Decide whether every SquareMatrix Statistic should allow either orientation

&#x20;  or whether row-square and column-square contracts deserve distinct classes.

6\. `SPSDMatrixStatistic` currently defines `\_validate\_result`, while the active

&#x20;  chain is named `\_get\_validated\_result`; consequently its semantic check is

&#x20;  not yet integrated. It should override the active hook, call `super()`, then

&#x20;  check Hermitian symmetry and `eigvalsh` with a numerical tolerance.

7\. `SPSDMatrixStatistic` uses `self.\_\_name\_\_` in error text. Instances do not

&#x20;  ordinarily have `\_\_name\_\_`; use `type(self).\_\_name\_\_` or `self.name`.

8\. The maintainer decided ScalarStatistic may return either a `numbers.Real`

&#x20;  scalar or a zero-dimensional real NumPy array, with downstream normalization.

&#x20;  The current aliases/classes do not yet fully express this: the general union

&#x20;  includes `RealNumber`, vector, and matrix but omits the zero-dimensional

&#x20;  jaxtyping array alias; `ScalarStatistic` currently accepts only

&#x20;  `numbers.Real`.

9\. Some current docstrings still describe old dimensional contracts and need a

&#x20;  consistency pass after behavior settles.

10\. Tests for the new Scalar/Vector/Matrix/Square/SPSD validation hierarchy are

&#x20;   still needed.



\### Suggested next implementation sequence



Proceed one issue at a time, as requested by the maintainer:



1\. Finalize `\_core/types.py`, including the scalar-or-zero-dimensional-array

&#x20;  result type and precise naming.

2\. Decide the stable `\_get\_validated\_result` signature and implement it across

&#x20;  the hierarchy.

3\. Fix the SquareMatrix input/output shape relationship manually.

4\. Integrate the SPSD semantic validation into the same override chain.

5\. Add focused tests for accepted/rejected result type, rank, square shape,

&#x20;  data-dimension relationship, symmetry, and positive semidefiniteness.

6\. Repair `\_core` exports and stale imports only after the new contracts are

&#x20;  settled, then run the wider suite and migrate legacy Statistics.



\## Other pending or deferred work



\- Add verbose diagnostic warnings/logging for estimator cache comparison

&#x20; failures (`CachedEstimatorMixin` retains a TODO).

\- Design the Component-level result cache and a cascading public clear-cache

&#x20; API across Component, Reducer, and estimator caches.

\- Revisit time-based estimator eviction only if bounded strong-reference

&#x20; caching proves insufficient.

\- Continue making estimator/model state read-only where practical.

\- Propagate settings contexts explicitly when Calculator begins dispatching

&#x20; Statistics through worker threads.

\- Continue generalizing and completing K-Medoids support. Inspect spelling and

&#x20; argument mapping carefully (`kmedoids`/`medoids`, initializer conventions,

&#x20; and supported scikit-learn-extra API).

\- Integrate the legacy code under `pyspoc/\_base.py`, `pyspoc/statistic.py`, and

&#x20; older Statistic/Reducer modules with the new `\_core` hierarchy. Legacy code

&#x20; was written quickly and contains overlapping functionality; improve rather

&#x20; than mechanically preserve it.

\- Complete package documentation and update README architecture once the new

&#x20; core hierarchy is stable.



\## Testing and tooling



The project virtual environment is located at `.venv`. Prefer explicit

executables so commands do not depend on shell activation:



```bash

.venv/bin/python -m pytest ...

.venv/bin/ruff check ...

```



Useful focused test areas:



```text

tests/test\_core\_component.py

tests/test\_typechecking.py

tests/test\_settings.py

tests/test\_random.py

tests/test\_caching.py

tests/statistics/dimreduce/orthopcae/

tests/statistics/clustering/kmeans/

tests/rstatistics/fractal/

```



At the time this document was created, `pyproject.toml` had an uncommitted user

change reorganizing optional dependency groups. Preserve and inspect that diff

before editing the file.



Ruff is configured through `\[tool.ruff.lint]`. Jaxtyping shape strings may

require `F722` handling, and explicit aliases may produce the separate `F821`

false positive described above.



\## Environment and operational notes



\- The development machine is a personal Linux machine.

\- A previous Codex sandbox failure was caused by Ubuntu/AppArmor restricting

&#x20; unprivileged user namespaces (`kernel.apparmor\_restrict\_unprivileged\_userns`).

&#x20; The user adjusted the system so normal sandbox commands later worked.

\- VS Code file-watcher limits were adjusted separately through a sysctl file.

\- Git credentials were made available with the `store` credential helper after

&#x20; `credential.helper` was initially unset. Do not expose stored credentials.

\- The user normally performs renames through Pylance refactoring, but stale

&#x20; imports can remain. Always verify renamed modules using `rg` and tests.

\- The Codex Diff view has intermittently shown blank/"No diff available" even

&#x20; when Git diffs exist. Use `git diff`, source-control views, or mouseover diffs

&#x20; as fallbacks; do not infer that a blank Codex tab means no changes occurred.



\## Guidance for the next Codex session



1\. Read this document and inspect `git status` before doing anything.

2\. Treat the new `\_core` hierarchy as active design work and the older classes

&#x20;  as legacy compatibility code.

3\. Validate assumptions against current files because the maintainer frequently

&#x20;  makes small changes between sessions.

4\. Explain variance, MRO, threading, and numerical tradeoffs concretely; the

&#x20;  maintainer prefers understanding the design rather than receiving opaque

&#x20;  patches.

5\. For requested implementation work, make focused changes, add documentation

&#x20;  and comments, and run tests proportional to risk.

6\. Report unrelated blockers separately and do not repair broad legacy areas

&#x20;  without authorization.



