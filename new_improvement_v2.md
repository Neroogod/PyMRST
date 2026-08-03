# Proposed improvements v2: a trust architecture for PyMRST

## Executive position

PyMRST's opportunity is larger than reproducing MRST in Python. It can make
incorrect reservoir simulations harder to produce, easier to detect, and much
easier to diagnose and reproduce.

The original `new_improvements.md` identifies the right defect classes:

- independent formulations can expose incorrect reference data;
- units, index bases and composition bases create silent errors;
- physical invariants are not checked consistently;
- parsed deck keywords can disappear before reaching a model;
- solver failures retain too little evidence;
- long simulations are difficult to reproduce or resume;
- performance work needs validation evidence before numerical backends change.

The principal improvement in this v2 is architectural. Cross-formulation
testing, convergence provenance, structured logging, invariant checks, failure
forensics and reproducibility should not become separate subsystems that each
collect similar data. They should be consumers of one shared simulation
evidence model.

The recommended first initiative is therefore:

> **PyMRST Simulation Evidence and Trust Layer**

Cross-formulation comparison remains the first major user-facing capability,
but it should be built on this common evidence foundation.

## 1. The common evidence model

Every simulator should produce states and a structured report:

```python
states, report = simulate_schedule(...)
```

The report should have a stable, versioned schema:

```text
SimulationReport
├── schema_version
├── manifest
│   ├── input and dataset hashes
│   ├── PyMRST version and git revision
│   ├── Python and dependency versions
│   ├── model/formulation/backend identity
│   └── solver and tolerance settings
├── timesteps
│   ├── requested and accepted timestep
│   ├── attempts and cuts
│   ├── nonlinear and linear iterations
│   └── wall-clock timings
├── convergence
│   ├── requested tolerances
│   ├── achieved residuals
│   ├── residual histories
│   └── convergence classification
├── conservation
│   ├── component inventories
│   ├── boundary/source/well contributions
│   └── numerical imbalance
├── invariants
├── deck_coverage
├── warnings
├── failures
└── comparisons
```

This object becomes the shared source for:

- structured logging;
- progress reporting;
- Newton failure forensics;
- reference-export qualification;
- reproducibility manifests;
- cross-formulation comparison;
- OPM/MRST comparisons;
- dashboards and notebook displays;
- checkpoint/restart metadata.

The report should be serializable to JSON for summaries and to a chunked
scientific format when it includes large histories.

## 2. Cross-formulation validation as a graph

Any problem that PyMRST can solve in two valid ways should be solved both ways.
Disagreement is evidence that at least one path, convention or convergence
criterion is wrong. This does not require stored reference data and can expose
errors in the reference itself.

Important pairs include:

| Pair | Defects it can expose |
|---|---|
| Overall composition vs natural variables | Flash coupling and primary-variable errors |
| Fully implicit vs sequential | Splitting, transport and convergence errors |
| AD Jacobian vs finite differences | Derivative and scaling errors |
| Direct vs iterative solve | Linear-solver, preconditioner and scaling faults |
| PyMRST vs MRST | Port and convention errors |
| PyMRST vs OPM Flow | Independent deck/simulator disagreements |

The initial API can remain simple:

```python
report = compare_formulations(
    setup,
    formulations=["natural", "overall"],
    fields=["pressure", "saturation", "well_rates"],
    atol={"pressure": 10.0},
    rtol={"saturation": 1e-6},
)
report.assert_passed()
```

Internally, comparisons should form a validation graph rather than an
unstructured collection of pairwise tests. Each edge records:

- compatible cases;
- field and unit mappings;
- cell, phase, component and well ordering;
- report-time alignment;
- expected tolerance by quantity;
- achieved convergence quality on both sides;
- known convention differences;
- whether the implementations share numerical kernels.

Shared implementations agreeing is valuable but not proof of correctness.
Independent simulators, analytic solutions, physical invariants, conservation
and finite-difference checks are different evidence classes and should be
reported separately.

## 3. Metamorphic and property-based testing

Property-based tests should generate physically valid numerical inputs and
verify mathematical properties rather than stored output arrays.

Examples include:

- a two-phase flash either satisfies the Rachford-Rice endpoint conditions or
  returns a vapour fraction that zeros the residual;
- phase fugacities agree at equilibrium;
- grid topology has monotone position arrays and valid distinct neighbours;
- unit conversions round-trip;
- interpolation respects endpoints and monotonicity where required.

Metamorphic tests should transform complete valid cases in ways with known
physical effects:

- permute cell numbering and recover the same physical result;
- reorder wells or components and recover the same named outputs;
- convert SI to field units and back;
- convert mole to mass fractions and back;
- reverse an internal-face orientation without changing physical flux;
- rotate an isotropic mechanics problem and rotate its solution back;
- divide a timestep and approach the same solution;
- apply consistent length/permeability/time scaling;
- compare zero gravity with a mathematically equivalent zero-density term.

These tests target indexing, ordering, unit and convention defects particularly
well.

## 4. Runtime physical invariants

Runtime checks should be controlled by a validation profile:

| Profile | Behavior |
|---|---|
| `off` | Disable optional checks |
| `observe` | Record violations without interrupting |
| `strict` | Stop on definite physical impossibilities |
| `reference` | Strict checks plus export-quality convergence requirements |

Initial invariants should cover:

- finite pressures, rates, compositions and displacements;
- saturation bounds and unit sum;
- non-negative normalized compositions;
- vapour-fraction bounds and phase-state consistency;
- component mass conservation;
- pore-volume positivity;
- valid porosity and permeability;
- well controls and limits;
- mechanics equilibrium and constrained rigid-body modes.

Every violation needs scale, location and severity:

```text
MassBalanceViolation
step=17
component=water
relative_error=2.1e-5
configured_limit=5e-6
worst_region=FAULT_BLOCK_2
```

Tiny floating-point excursions should not be treated like large physical
violations. Invariant tolerances should be scale-aware and derived from the
configured nonlinear tolerances where appropriate.

## 5. Convergence provenance and failure forensics

Every accepted state should record:

- requested tolerance per equation group;
- residual actually achieved;
- nonlinear and linear iterations;
- relaxation and line-search activity;
- control switches;
- timestep attempts and cuts;
- termination reason.

Reference export should reject insufficiently converged states by default.
The reference header must record solver settings, achieved residuals, units,
software versions, dataset revision and generation command.

Failure analysis should classify residual behavior:

- converging;
- stalled at an estimated numerical floor;
- oscillating or control-chattering;
- diverging;
- non-finite;
- linear-solver failure;
- constraint/control failure.

An actionable diagnostic should look like:

```text
Well residual stalled near 2.06e-7 for eight iterations.
Requested tolerance: 1e-9.
All cell residuals passed.
The well-control equation alternated between two states.
Suggested diagnostic tolerance: at least 5e-7.
```

PyMRST may recommend a tolerance but should not silently relax it. Tolerance
selection is an engineering acceptance decision.

## 6. Structured logging and progress

Replace direct simulator `print()` calls with structured events. Human-readable
console output should be one handler, not the source of truth.

Useful event types include:

- run started/completed;
- report step started/accepted/rejected;
- nonlinear iteration;
- linear solve;
- timestep cut;
- well/control switch;
- invariant warning;
- checkpoint written;
- failure bundle written.

Each event should carry identifiers and structured numerical values. Handlers
can provide:

- compact terminal progress bars;
- verbose engineering tables;
- JSON Lines logs;
- live notebook plots;
- browser dashboards;
- retained post-run convergence analysis.

## 7. Deck coverage and property lineage

A deck keyword needs more than known/unknown status. Use five states:

| State | Meaning |
|---|---|
| Unknown | No parser exists |
| Parsed | Syntax was recognized |
| Normalized | Units, defaults, boxes and operators were resolved |
| Applied | Data reached model construction |
| Verified | A model-level check confirmed its effect |

The highest-risk state is parsed but not applied.

Applied properties should retain source lineage where practical:

```text
rock.perm[cell 100:200]
← MULTIPLY PERMX 0.1
← BOX I=1:10 J=2:5 K=3
← CASE.DATA line 418
```

The loader should print a short coverage summary and attach the full report to
the simulation evidence. Strict/reference profiles should reject parsed-but-
unapplied keywords unless they have an explicit accepted disposition.

## 8. Boundary-safe units, indices and composition bases

An opt-in dimensioned quantity layer remains valuable, but it should be applied
first at external boundaries rather than throughout numerical hot loops.

Important semantic tags include:

- pressure, time, permeability and transmissibility dimensions;
- one-based versus zero-based indices;
- mole, mass and volume composition bases;
- surface versus reservoir rates;
- absolute versus gauge pressure;
- depth-positive versus elevation-positive coordinates.

The default internal representation can remain NumPy arrays and SI floats. The
boundary layer validates and strips tags after conversion, which limits runtime
and compatibility cost.

Claims of zero disabled-mode overhead should be avoided until measured. Even
boundary wrappers and dispatch can carry a small cost.

## 9. Checkpointing and resumable simulation

Checkpointing should be a high-priority operational feature. A long simulation
must not lose all accepted report steps because a final check, export or later
timestep fails.

Persist after each accepted report step:

- reservoir and well state;
- model-control state;
- schedule position and elapsed time;
- solver and convergence report;
- input fingerprint;
- deterministic/replay settings;
- random state when relevant.

Proposed interface:

```bash
pymrst run case.py --checkpoint results/run.zarr
pymrst resume results/run.zarr
```

Resume should reject incompatible code, model, dataset or input fingerprints
unless the user explicitly requests a migration.

## 10. Minimal failure bundles

When a simulation fails, automatically create a compact reproducible bundle:

- previous accepted and failing trial states;
- failing timestep and control;
- residual histories and convergence classification;
- Jacobian shape, block norms and conditioning diagnostics;
- implicated well or local grid neighborhood;
- input/software manifest;
- exact rerun command.

This allows developers to reproduce a field failure without rerunning all prior
steps or receiving an entire proprietary model.

## 11. Component-wise conservation ledger

Track component conservation continuously:

```text
initial inventory
+ boundary influx
+ well injection
+ source terms
- production
= current inventory
+ numerical imbalance
```

Provide global, regional and well-level ledgers. This distinguishes nonlinear
error from incorrect rate signs, separator conventions, boundary conditions,
PVT conversions and stale pore-volume terms.

## 12. Reproducibility and deterministic replay

Every result should include:

- input and dataset hashes;
- PyMRST version and git revision;
- Python and dependency versions;
- solver/backend settings;
- thread count and platform information;
- random seeds;
- deterministic-mode status.

Deterministic replay should pin ordering, sparse-solver options, timestep
decisions and well-control switching where supported. Reports should distinguish:

- bitwise reproducible;
- numerically reproducible within tolerance;
- physically equivalent within engineering criteria.

## 13. Scenario comparison tooling

Reservoir engineers need a first-class run comparator:

```python
comparison = compare_runs(base, candidate)
```

It should align report times and named wells, then compare:

- pressure and saturation maps;
- production and injection histories;
- cumulative volumes;
- breakthrough times;
- uplift/subsidence or other coupled quantities;
- convergence and runtime;
- conservation;
- changed inputs, schedule controls and model settings.

The output should support JSON, Markdown, plots and an interactive dashboard.

## 14. Fault-injection testing

The trust layer itself needs tests. Intentionally inject common defects and
verify that the correct guardrail detects them:

- one-based index passed to a zero-based API;
- permeability scaled by the wrong unit;
- mole fractions labeled as mass fractions;
- parsed deck multiplier dropped before model construction;
- face orientation reversed incorrectly;
- invalid saturation or vapour fraction;
- well-rate sign reversed;
- old-state pore volume evaluated with new linearization terms.

This verifies diagnostic coverage rather than only numerical code.

## 15. Sparse assembly and repeated-solve diagnostics

Performance work should begin with measurement. Add instrumentation for:

- sparse allocations per equation assembly;
- time spent constructing, canonicalizing and filling sparse matrices;
- sparsity-pattern changes;
- reusable operators and Jacobian blocks;
- repeated symbolic or numerical factorizations;
- linear-solve time and convergence;
- peak memory.

PyMRST has already demonstrated the value of reuse: caching the immutable VEM
stiffness factorization changed the 7,392-cell fixed-stress mechanics workflow
from repeated expensive refactorization to fast repeated substitutions.

The measured optimization sequence should be:

1. reuse immutable operators and factorizations;
2. preallocate stable sparsity patterns and fill values in place;
3. reduce transient sparse-object creation;
4. profile representative field cases;
5. then select PETSc, hypre, AMGCL, AMGX or GPU linear solvers where the solve
   is demonstrably dominant.

## 16. Result storage, export and streaming

Result handling should support:

- VTK for ParaView;
- HDF5 or Zarr for chunked simulation histories;
- xarray for labeled multidimensional analysis;
- Parquet for well and summary tables;
- lazy loading of selected variables, cells and timesteps;
- streaming accepted states to disk.

Out-of-core storage removes the assumption that every state in a large SPE10 or
field-scale run fits in memory. It also supplies the persistence mechanism for
checkpointing and browser/notebook inspection.

## 17. Parallel ensemble execution

The ensemble driver should expose bounded parallel execution:

```python
result = run_ensemble(cases, n_jobs=8, backend="process")
```

Requirements include:

- deterministic member identifiers and seeds;
- per-member evidence and failure bundles;
- bounded memory and worker recycling;
- continue-on-error behavior;
- resumable completed members;
- aggregation without loading every full state history;
- safe handling of non-picklable model objects.

Local process pools should come before distributed execution. Dask, Ray, Slurm
or cloud adapters can follow behind a stable member-execution contract.

## 18. Independent OPM Flow validation

Promote the existing ECLIPSE result readers from example utilities into a
library-level comparison package. Extend support to the result and grid files
needed to align:

- active-cell mappings;
- pressure and saturation histories;
- well BHP and phase rates;
- restart states;
- summary vectors;
- aquifer histories.

OPM Flow provides independent evidence for the overlapping ECLIPSE-deck
workflow. It is not an oracle for every PyMRST capability, particularly VEM
mechanics and specialized compositional formulations. The validation graph
must state the supported overlap and known convention differences explicitly.

## 19. Benchmark and acceptance suite

Package SPE1, SPE3, SPE9, SPE10, Egg, Norne and relevant CO2-storage cases as a
curated benchmark suite. Each benchmark should declare:

- dataset revision and hash;
- supported models/formulations;
- expected engineering outputs;
- accepted tolerances;
- convergence and conservation limits;
- reference simulators;
- expected runtime/memory class;
- quick, standard and field-scale profiles.

The suite should produce publishable Markdown/JSON reports and trend results
over time.

## 20. Versioned model and state contracts

Flexible dictionary states are useful internally but risky at boundaries.
Define optional versioned schemas for:

- reservoir state;
- well solution;
- schedule and controls;
- simulation reports;
- result archives;
- surrogate inputs and outputs.

Validation can remain disabled inside hot numerical loops and mandatory during
model construction, serialization, restart, comparison and external API calls.

A capability registry should let models declare supported features:

```python
model.capabilities = {
    "phases": {"water", "oil", "gas"},
    "disgas": True,
    "adjoint": True,
    "restart": True,
    "surrogate_flash": True,
    "gpu_linear_solve": False,
}
```

Invalid combinations can then fail before simulation begins.

## 21. Surrogate-component contract

Generalize the Victory flash experiment into a contract for learned components:

- declared input/output schema and units;
- validity domain and training-data fingerprint;
- uncertainty or out-of-distribution score;
- physical invariant checks;
- physics fallback;
- accuracy guard against sampled conventional evaluations;
- batch execution and optional GPU placement;
- complete provenance in the simulation report.

Candidate components include flash, stability, PVT, relative permeability and
selected well correlations. A learned component must remain an explicit model
choice, not a silent replacement.

## 22. Differentiable and accelerator backends

GPU or differentiable work should be divided by numerical role:

- keep PyTorch/JAX for dense batched learned surrogates;
- use implicit differentiation or existing adjoints for converged simulator
  solutions;
- put GPU acceleration behind sparse assembly or the linear-solver boundary;
- prefer CuPy/PETSc/AMGX-style sparse integration over dense torch solves for
  the simulator core.

Small-case profiling does not determine the field-scale bottleneck. Profile
large SPE10 and Norne cases before selecting a solver backend. Hardware claims
about double precision must be tied to specific devices; consumer and data-
center GPUs have very different FP64 capability.

## 23. Notebook and browser experience

Rich representations should summarize grids, schedules, states, wells and
reports without opening a plotting window. Useful features include:

- `_repr_html_` summaries;
- progress bars backed by structured events;
- live residual and conservation plots;
- lazy field selection;
- engineering unit controls;
- side-by-side scenario comparison;
- downloadable evidence and failure bundles.

Browser execution through Pyodide is a roadmap item. A smaller teaching kernel
with explicit grid/cell limits is more realistic than promising field-scale
simulation in WebAssembly.

## 24. Automated tolerance calibration

Provide an explicit calibration tool rather than silently changing tolerances.
For a representative case, sweep selected nonlinear/well/splitting tolerances
and report:

- achieved residual floor;
- output sensitivity;
- conservation error;
- runtime and iteration count;
- control switching and timestep cuts.

The user can then select an acceptance tolerance from evidence rather than
trial and error.

## 25. Recommended delivery sequence

### Phase A: evidence foundation

1. Versioned `SimulationReport` and `SimulationEvidence` schemas.
2. Structured solver events and retained convergence histories.
3. Reproducibility manifest.
4. Component-wise conservation ledger.
5. Checkpoint/restart metadata and per-report-step persistence.

### Phase B: correctness guardrails

1. Cross-formulation comparison on compositional natural/overall variables.
2. Normalized comparison of report times, units, phases, wells and active cells.
3. Runtime invariant profiles.
4. Three/five-state deck coverage and data lineage.
5. Newton failure classification and minimal failure bundles.
6. Property-based, metamorphic and fault-injection tests.
7. Index-base and composition-basis tags at external boundaries.

### Phase C: independent validation and usability

1. OPM Flow comparison adapter.
2. Curated benchmark/acceptance suite.
3. Scenario comparison reports.
4. VTK, Zarr/HDF5, xarray and Parquet exports.
5. Notebook progress and result inspection.
6. Parallel, resumable ensembles.

### Phase D: performance and research

1. Sparse-allocation and repeated-factorization profiling.
2. Stable sparsity-pattern reuse and in-place Jacobian value assembly.
3. Field-scale linear-solver benchmark.
4. PETSc/hypre/AMGX/CuPy integrations selected from measured evidence.
5. Out-of-core field cases.
6. Surrogate-component contracts.
7. Implicit-differentiation and accelerator backends.
8. Limited browser/teaching runtime.

## 26. First concrete release

The first release should be deliberately small and vertical rather than a broad
set of partially connected utilities.

### Deliverables

1. `SimulationReport` with manifest, timesteps, residuals and conservation.
2. JSON serialization and a concise terminal renderer.
3. Structured events from `NonLinearSolver`.
4. Failure classification for converged, stalled, oscillating and diverging
   histories.
5. `compare_reports` with pressure, saturation and well-rate alignment.
6. Natural-variable versus overall-composition agreement test.
7. Reference-export guard based on achieved convergence.
8. Checkpoint written after every accepted report step.

### Success criteria

- the phase-114 unconverged-reference problem is diagnosed from report evidence;
- the Norne/control residual floor is classified as a stall or oscillation;
- a failed long case can resume from its previous accepted report step;
- comparison output names the worst field, timestep, cell/well and magnitude;
- reports are sufficient to reproduce solver settings without reading console
  output;
- existing numerical results remain unchanged when evidence collection is
  enabled.

## Final recommendation

PyMRST should not pursue isolated modernization features first. It should build
an evidence spine through the simulator and attach validation, diagnostics,
reproducibility, comparison and resumability to that spine.

This sequence preserves the strongest idea in the original proposal—redundant
formulations can catch errors even when stored references are wrong—while
making it scalable beyond one compositional test. It also moves checkpointing,
conservation accounting, metamorphic testing and failure bundles into the core
roadmap, because those features directly reduce the cost and risk of real
engineering simulation.

The result would not merely be a Python version of MRST. It would be a
reservoir-simulation platform that explains what it did, how well it converged,
why it failed, whether independent formulations agree, and exactly how to
reproduce or resume the calculation.
