# PyMRST improvements v3: Research Studio and Simulation Evidence Platform

## Vision

PyMRST should become more than a Python translation of MRST. It should become
a trustworthy, reproducible and extensible reservoir-simulation research
platform where formulations, solvers, physics models and machine-learning
surrogates can be compared under one evidence system.

The objective is deliberately ambitious:

> **Every model is interchangeable, every result carries evidence, every
> failure is diagnosable, every long run is resumable, and every research
> conclusion can be reproduced.**

This is not a proposal for a small demonstration. It is a proposal for a
coherent platform that reservoir engineers can rely on for real simulation
work and researchers can use to design, validate and publish new methods.

The proposed platform is called **PyMRST Research Studio**. Its numerical and
scientific foundation is the **Simulation Evidence Engine**.

## Why this would make PyMRST distinct

MRST is valuable because researchers can inspect and modify reservoir-
simulation mathematics. PyMRST should preserve that openness while adding
capabilities that are difficult to assemble in MATLAB or traditional reservoir
simulators:

- formulation-independent experiment management;
- structured numerical evidence attached to every result;
- automatic conservation and physical-invariant checks;
- comparison with independent formulations and simulators;
- actionable nonlinear-solver failure diagnosis;
- interruption-safe checkpointing and branching;
- physics/ML component substitution with accuracy guards;
- scalable ensemble and optimization workflows;
- open scientific result formats and lazy data access;
- reproducible, publication-ready validation reports;
- notebook and browser-native engineering interaction.

The distinguishing product is not one solver or one user interface. It is the
integration of simulation, evidence, comparison, experimentation and
reproducibility.

## Platform architecture

```text
PyMRST Research Studio
│
├── Simulation Runtime
│   ├── schedules and controls
│   ├── nonlinear and linear solvers
│   ├── timestep management
│   ├── checkpoint and restart
│   └── streaming state storage
│
├── Simulation Evidence Engine
│   ├── convergence provenance
│   ├── conservation ledger
│   ├── physical invariants
│   ├── failure diagnosis
│   ├── deck/property lineage
│   └── reproducibility manifest
│
├── Validation Laboratory
│   ├── cross-formulation agreement
│   ├── MRST comparison
│   ├── OPM Flow comparison
│   ├── analytic solutions
│   ├── finite-difference derivatives
│   ├── property and metamorphic tests
│   └── benchmark acceptance suites
│
├── Experiment Engine
│   ├── parameter sweeps
│   ├── uncertainty ensembles
│   ├── sensitivity analysis
│   ├── history matching
│   ├── optimization
│   └── experiment branching
│
├── Component Registry
│   ├── physical models
│   ├── formulations
│   ├── linear and nonlinear solvers
│   ├── learned surrogates
│   └── external simulators
│
├── Result Platform
│   ├── HDF5, Zarr and xarray
│   ├── VTK and Parquet export
│   ├── lazy and out-of-core access
│   ├── scenario comparison
│   └── publication reports
│
└── Research Studio Interface
    ├── browser dashboard
    ├── notebook integration
    ├── live convergence and conservation
    ├── reservoir field visualization
    └── experiment and validation workspace
```

## 1. Unified simulation runtime

PyMRST currently exposes several schedule, packed-problem, adjoint,
optimization and example-specific simulation paths. Research Studio requires a
shared runtime contract without erasing specialized numerical workflows.

All simulation paths should participate in a common session:

```python
session = SimulationSession(
    model=model,
    state0=state0,
    schedule=schedule,
    solver=solver,
    store="results/norne.zarr",
)

result = session.run()
```

The session should coordinate:

- report and ministep progression;
- nonlinear and linear solver events;
- schedule/control changes;
- accepted and rejected state transitions;
- checkpoint persistence;
- evidence collection;
- progress and cancellation;
- result streaming;
- deterministic decision recording.

Existing simulation functions can remain as compatibility entry points that
construct or delegate to a session. Specialized adjoint and optimization
drivers should attach to the same session/evidence protocols rather than being
forced into an inappropriate generic numerical kernel.

### Runtime contract requirements

- Existing numerical results remain within their certified tolerances.
- Evidence collection does not alter solver decisions unless a strict
  validation profile explicitly requests termination.
- Every accepted report step is persistable.
- Every failure retains the last accepted state and the failed attempt.
- All execution paths have stable run, case, step and attempt identifiers.
- Long-running work can be monitored, cancelled and resumed safely.

## 2. Simulation Evidence Engine

Every simulation should produce a structured, versioned report:

```python
result.states
result.report
result.evidence
result.store
```

The evidence model should include:

```text
SimulationEvidence
├── schema_version
├── manifest
├── input_lineage
├── timestep_reports
├── convergence
├── conservation
├── invariants
├── deck_coverage
├── warnings
├── failures
├── timings
├── comparisons
└── checkpoints
```

This is the shared source for terminal output, structured logs, notebook
widgets, browser dashboards, failure diagnosis, validation comparisons and
published reports. Console text must be treated as a renderer of evidence, not
as the evidence itself.

## 3. Reproducibility manifest

Every result should record:

- input, schedule and dataset hashes;
- PyMRST version and git revision;
- Python and dependency versions;
- operating system, architecture and hardware summary;
- model, formulation and component identities;
- nonlinear and linear solver settings;
- requested tolerances;
- unit/convention profile;
- random seeds;
- thread and process settings;
- external simulator versions;
- checkpoint and result-schema versions.

The report should distinguish:

- deterministic decision replay;
- numerical reproducibility within tolerance;
- physical equivalence within documented engineering criteria.

Bitwise reproducibility should only be claimed for controlled kernels and
environments where it has been demonstrated. Threaded sparse factorizations
and pivoting make it an inappropriate general promise.

## 4. Component-wise conservation engine

Conservation should be a central service rather than a collection of local
post-processing checks.

The engine should support:

- water, oil and gas phases;
- dissolved gas and vaporized oil;
- compositional components;
- wells and completion fluxes;
- aquifers;
- boundaries and distributed sources;
- separators and surface networks;
- compressible rock;
- poromechanical pore-volume changes;
- regions, fault blocks and arbitrary cell groups.

For every component and reporting region, calculate:

```text
initial inventory
+ boundary influx
+ well injection
+ source terms
- production
= final inventory
+ numerical imbalance
```

An engineering report should identify the scale and location of imbalance:

```text
Water balance — report 17
Initial inventory        7.3221e7 m3
Injection                1.2600e5 m3
Production              -2.8914e4 m3
Boundary influx          0.0000e0 m3
Current inventory        7.3318e7 m3
Numerical imbalance      4.13e-7 relative
Worst region             Region 4
```

Each physical model must implement an explicit conservation adapter defining
its component basis, inventory, flux/source terms and surface/reservoir
conventions. A generic ledger must never guess these semantics.

## 5. Runtime physics invariants

Validation profiles should control runtime checks:

| Profile | Behavior |
|---|---|
| `off` | Disable optional checks |
| `observe` | Record violations and continue |
| `strict` | Stop on definite physical impossibilities |
| `reference` | Strict checks plus export-quality convergence requirements |

Checks should include:

- finite pressure, rate, composition and displacement values;
- saturation bounds and unit sum;
- non-negative normalized compositions;
- phase-state and flash consistency;
- vapour-fraction bounds;
- positive pore volume;
- valid porosity and permeability;
- well controls and limits;
- mass/component conservation;
- mechanics equilibrium and rigid-body constraints.

Violations must report severity, tolerance, location, physical basis and the
relevant state fields. Tolerances should be scale-aware and linked to achieved
solver accuracy where appropriate.

## 6. Convergence provenance

Every accepted or rejected step should retain:

- requested tolerances per equation group;
- achieved residuals;
- complete residual histories;
- nonlinear and linear iterations;
- timestep attempts and cuts;
- relaxation and line-search activity;
- solver/preconditioner status;
- well and group-control switches;
- convergence/failure reason;
- time and memory measurements.

Reference export should reject results that do not meet configured reference-
quality criteria. Exported fixtures should retain achieved convergence,
software versions, dataset hashes, units and generation commands.

## 7. Intelligent failure diagnosis

The solver should classify failures as:

- converging but iteration-limited;
- stalled near a numerical floor;
- oscillating;
- well/control chattering;
- diverging;
- non-finite;
- linear-solver/preconditioner failure;
- timestep or update-limit failure;
- physical-invariant failure.

It should identify:

- the worst residual and equation group;
- owning cell, face, component, segment or well;
- residual trend and estimated floor;
- relevant control switches;
- timestep sensitivity;
- conservation impact;
- suspected failure family;
- recommended diagnostics.

Example:

```text
Failure classification: well-control chattering

Location
  Well: INJ-3
  Report step: 18
  Ministep: 2.5 days

Evidence
  CNV reduced from 2.7e-2 to 1.8e-8
  MB passed
  Well residual stalled near 2.1e-7
  Control residual alternated between two states
  Eight iterations repeated the same cycle

Suggested actions
  Inspect the BHP/rate limit intersection.
  Enable safeguarded line search or control hysteresis.
  Compare the requested well tolerance with the observed floor.
```

The system may recommend a tolerance but must not silently relax engineering
acceptance criteria.

## 8. Structured events, logging and progress

Simulation activity should be emitted as structured events:

- run started/completed;
- report/ministep started;
- nonlinear iteration;
- linear solve;
- step accepted/rejected;
- timestep cut;
- control switch;
- invariant violation;
- checkpoint written;
- failure bundle written.

Handlers can render:

- compact terminal progress bars;
- detailed engineering tables;
- JSON Lines logs;
- live residual/conservation plots;
- browser and notebook dashboards;
- retained post-run diagnostics.

## 9. Deck coverage and data lineage

Every deck keyword should pass through explicit states:

| State | Meaning |
|---|---|
| Unknown | No parser exists |
| Parsed | Syntax recognized |
| Normalized | Units, defaults, boxes and operators resolved |
| Applied | Data reached model construction |
| Verified | Model-level validation confirmed the effect |

Parsed-but-not-applied keywords are the highest-risk class and should be
visible during loading. Strict/reference modes should reject them unless an
explicit accepted disposition exists.

Property lineage should preserve transformations:

```text
rock.perm[cell 100:200]
← MULTIPLY PERMX 0.1
← BOX I=1:10 J=2:5 K=3
← CASE.DATA line 418
```

The same principle should cover schedule controls, regions, PVT tables, grid
activity and well completions.

## 10. Semantic boundary types

The hot numerical core can continue using NumPy arrays and SI floats, while
external boundaries use semantic tags for:

- physical dimensions;
- one-based and zero-based indices;
- mole, mass and volume fractions;
- surface and reservoir rates;
- absolute and gauge pressure;
- depth-positive and elevation-positive coordinates;
- phase and component ordering.

Tags should be validated and removed after conversion. Disabled-mode overhead
must be measured rather than assumed to be zero.

## 11. Checkpoint, restart and experiment branching

Checkpoint after configurable accepted ministeps or report steps. Persist:

- reservoir and facility state;
- model-control state;
- schedule position and elapsed time;
- solver/convergence state;
- evidence accumulated so far;
- input and code fingerprints;
- deterministic decision record;
- random state where relevant.

Requirements include:

- interruption-safe transactional writes;
- restart from any compatible checkpoint;
- schedule extension;
- branch-from-checkpoint experiments;
- lazy state loading;
- compatibility and migration reporting;
- retention of failed trial states separately from accepted states.

Example research workflow:

```python
base = Simulation.open("norne_base")
branch = base.branch(step=15)
branch.modify_controls(...)
branch.run()
branch.compare_with(base)
```

Initial storage may use existing HDF5 support. Zarr and xarray should be added
as the result platform matures rather than assumed to exist today.

## 12. Minimal reproducible failure bundles

On failure, write a compact bundle containing:

- previous accepted and failed trial states;
- failing timestep and controls;
- residual and update histories;
- Jacobian block shapes, norms and conditioning diagnostics;
- implicated well or local grid neighborhood;
- model/input/software manifest;
- exact reproduction command or Python script.

This should let a developer reproduce a field failure without receiving the
complete proprietary model or rerunning all preceding report steps.

## 13. Multi-formulation experiment laboratory

Researchers should define one physical case and run multiple formulations,
solvers and components:

```python
experiment = Experiment(
    case="SPE1",
    variants={
        "formulation": ["natural", "overall"],
        "flash": ["physics", "victory_ensemble"],
        "linear_solver": ["direct", "cpr"],
    },
)

results = experiment.run(parallel=True)
results.compare()
results.publish("results/spe1-study")
```

Variant dimensions should include:

- natural and overall compositional variables;
- fully implicit, sequential and AIM integration;
- fully coupled and fixed-stress mechanics;
- physics and learned components;
- direct, CPR, AMG and external linear solvers;
- PyMRST, MRST and OPM Flow where supported;
- tolerance and timestep strategies;
- grid/property/control scenarios.

The platform should compare state and production histories, convergence,
conservation, derivatives, runtime, memory and surrogate error.

## 14. Simulation evidence graph

Validation should be represented as a graph of evidence relationships:

```text
                         Analytic solution
                                │
                     physical invariants
                                │
MRST reference ── PyMRST formulation A ── PyMRST formulation B
       │                    │                    │
       └────────────── OPM Flow ────────────────┘
                                │
                     finite-difference Jacobian
```

Each edge defines:

- compatible cases;
- report-time alignment;
- active-cell, well, phase and component mappings;
- unit conversions;
- expected tolerance by quantity;
- achieved convergence on both endpoints;
- known convention differences;
- implementation independence.

Reports should separate:

- internal formulation agreement;
- independent simulator agreement;
- analytic agreement;
- invariant satisfaction;
- conservation quality;
- derivative validation;
- convergence qualification.

Example summary:

```text
Confidence summary
------------------
Conservation:            passed
Physics invariants:      passed
Natural/overall:         passed
PyMRST/MRST:             passed
PyMRST/OPM Flow:         passed
AD/finite differences:   passed
Reference quality:       residual-qualified
Overall evidence:        strong
```

Agreement between implementations that share kernels is valuable but not
independent proof. The evidence graph must make that distinction visible.

## 15. Property, metamorphic and fault-injection testing

### Property tests

- Rachford-Rice endpoint/residual conditions;
- phase fugacity equality at equilibrium;
- grid topology consistency;
- positive/normalized physical quantities;
- unit conversion round-trips;
- interpolation and tabular-property contracts.

### Metamorphic tests

- permute cell numbering;
- reorder wells, phases or components;
- reverse internal-face orientation consistently;
- convert SI to field units and back;
- convert composition bases and back;
- rotate isotropic mechanics cases;
- refine or split timesteps;
- apply consistent physical scaling.

### Fault-injection tests

- pass one-based indices to a zero-based API;
- apply an incorrect permeability unit factor;
- label mole fractions as mass fractions;
- parse but drop a deck multiplier;
- reverse a rate sign;
- corrupt a phase-state flag;
- evaluate old accumulation with new linearization terms;
- introduce well-control chattering.

Fault injection verifies the trust system itself rather than only numerical
answers.

## 16. Independent MRST and OPM Flow validation

MRST comparison remains essential for port parity. OPM Flow supplies an
independent implementation for overlapping ECLIPSE-deck workflows.

The external comparison layer should support:

- EGRID/INIT active-cell and property mapping;
- UNRST restart states;
- UNSMRY summary histories;
- pressure and phase state;
- well BHP and phase rates;
- aquifer histories;
- control/schedule changes;
- known sign, unit and convention mappings.

OPM Flow is not an oracle for every PyMRST capability, especially VEM
mechanics or specialized compositional research models. Supported overlap and
independence must be explicit in the evidence graph.

## 17. Scenario comparison workspace

A reservoir-aware comparison tool should align report times, active cells,
regions and named wells, then compare:

- pressure and saturation maps;
- production/injection histories;
- cumulative recovery;
- breakthrough times;
- well controls and limits;
- aquifer behavior;
- uplift, subsidence and stress;
- convergence and timestep cuts;
- runtime and memory;
- conservation and evidence confidence;
- changed model inputs and schedule controls.

Outputs should include Markdown/PDF reports, JSON, Parquet/CSV tables, static
plots and an interactive browser view.

## 18. Experiment registry and reproducible research packages

Every experiment should capture:

- hypothesis and description;
- base case and variant definitions;
- datasets and hashes;
- software environment;
- solver/component configurations;
- checkpoints and outputs;
- validation evidence;
- figures, tables and conclusions.

Published package layout:

```text
experiment/
├── manifest.json
├── environment.lock
├── cases/
├── checkpoints/
├── results.zarr
├── evidence/
├── comparisons/
├── figures/
├── tables/
├── report.md
└── reproduce.sh
```

This package should be suitable for archival with a paper or dataset.

## 19. Ensemble, uncertainty and history matching

The experiment engine should support:

- bounded local process parallelism;
- distributed adapters after a stable execution contract exists;
- deterministic member identifiers and seeds;
- Latin hypercube and Sobol sampling;
- parameter distributions and correlations;
- ensemble statistics and confidence intervals;
- sensitivity indices;
- failed-member isolation and diagnosis;
- checkpoint reuse and resumable members;
- gradient and adjoint optimization;
- ensemble and surrogate-assisted history matching.

Aggregation should not require all complete field histories in memory.

## 20. Hybrid physics and machine-learning component laboratory

Generalize the Victory flash work into a component contract for learned:

- flash and phase stability;
- PVT properties;
- relative permeability;
- well and facility correlations;
- selected constitutive laws.

Every surrogate should declare:

- versioned input/output schema;
- units and composition basis;
- training-data hash;
- validity/training domain;
- uncertainty or out-of-distribution method;
- physical invariants;
- conventional physics fallback;
- reference sampling policy;
- batching and device support.

Runtime workflow:

1. assess whether the input is in distribution;
2. evaluate the surrogate;
3. check physical invariants;
4. sample conventional evaluations for monitoring;
5. fall back to physics when uncertain or invalid;
6. record fallback location and reason;
7. quantify local and accumulated simulation error;
8. report speed, accuracy and coverage.

Learned components must be explicit experimental choices, never silent
replacements.

## 21. Versioned schemas and capability registry

Define schemas for:

- reservoir and facility states;
- schedule/control data;
- simulation evidence;
- checkpoint and result storage;
- experiment definitions;
- surrogate component inputs/outputs.

Schema validation can remain outside numerical hot loops and be mandatory at
construction, serialization, restart and external interfaces.

Models and components should declare capabilities:

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

Invalid combinations should fail before simulation begins.

## 22. Result platform and out-of-core state handling

Support:

- HDF5 and Zarr chunked histories;
- xarray labeled analysis;
- VTK export for ParaView;
- Parquet for well and summary tables;
- lazy field/timestep/region access;
- streaming accepted states;
- transactional checkpoint writes;
- archive and publication packages.

Large SPE10 and field cases should not require every state to remain in memory.

## 23. Sparse assembly and solver performance program

Measure:

- sparse allocations per equation assembly;
- time spent allocating, canonicalizing and filling matrices;
- stable versus changing sparsity patterns;
- repeated symbolic/numerical factorizations;
- reusable operators and blocks;
- linear-solve iterations and time;
- peak memory and data movement.

Optimization sequence:

1. reuse immutable operators and factorizations;
2. preallocate stable sparsity patterns;
3. fill Jacobian values in place;
4. reduce temporary AD and sparse objects;
5. profile field-scale cases;
6. choose PETSc, hypre, AMGCL, AMGX or GPU solvers based on measured needs.

Caching the immutable VEM mechanics factorization has already demonstrated the
value of structural reuse on the 7,392-cell Norne workflow.

## 24. Differentiable and accelerator research

Separate numerical roles:

- PyTorch/JAX for dense batched learned components;
- existing adjoints or implicit differentiation for converged simulator
  solutions;
- CuPy/PETSc/AMGX-style sparse acceleration behind assembly or linear-solver
  boundaries;
- no dense torch solve as a field-scale reservoir-simulator strategy.

Profile large SPE10/Norne workloads before choosing backends. FP64 performance
claims must be tied to specific hardware rather than generalized across GPUs.

## 25. Curated benchmark and publication center

Package:

- SPE1, SPE3, SPE9 and SPE10;
- SPE11 CO2-storage cases;
- Egg and Norne;
- compositional benchmark cases;
- aquifer and facility cases;
- poromechanics cases;
- unstructured/polyhedral grid cases;
- ML-surrogate comparison cases.

Each benchmark should declare:

- dataset revision/hash;
- supported formulations and components;
- expected engineering outputs;
- quantity-specific tolerances;
- convergence and conservation limits;
- reference simulators;
- expected runtime and memory class;
- quick, standard and field-scale profiles.

Every release should publish a cross-validation report containing:

- MRST and OPM comparisons where applicable;
- cross-formulation agreement;
- analytic/invariant evidence;
- conservation;
- convergence;
- derivative validation;
- runtime and memory;
- hardware/environment;
- dataset/code hashes;
- known exceptions and convention differences.

This report is the citable public artifact that allows other researchers to
trust and build on PyMRST.

## 26. Research Studio user experience

### Notebook experience

- rich grid, model, state and well summaries;
- live progress, convergence and conservation;
- lazy field selection;
- experiment creation and comparison;
- inline evidence/confidence summaries;
- exportable publication figures and tables.

### Browser workspace

- project/experiment browser;
- case construction and deck-coverage review;
- live simulation monitoring;
- map, slice and well visualization;
- scenario comparison;
- solver failure diagnosis;
- evidence graph exploration;
- report and reproducibility-package export.

Browser execution through Pyodide should target bounded teaching cases rather
than promise field-scale simulation in WebAssembly.

## 27. Automated tolerance calibration

Provide an explicit study tool that sweeps selected nonlinear, well and
splitting tolerances and reports:

- achieved numerical floor;
- output sensitivity;
- conservation error;
- runtime and iterations;
- control switching;
- timestep cuts;
- cross-method agreement.

The engineer or researcher chooses acceptance settings from evidence. PyMRST
does not silently change them.

## 28. Ordinary test-coverage program

Platform work does not replace ordinary tests. Maintain a continuous program
that:

- inventories modules without focused tests;
- ranks them by numerical risk, size, dependency centrality and use frequency;
- adds characterization tests before architectural refactoring;
- records defects found and their failure class;
- converts discovered defects into property, metamorphic or fault-injection
  regressions where possible;
- publishes test and validation coverage separately from line coverage.

The observed defect yield from first-time tests on previously untested modules
is strong evidence that this program must remain a first-class workstream.

## Development posture: private architecture-first hardening

PyMRST does not need to optimize this roadmap around an early public release.
The current stage should be treated as a private platform-development and
numerical-hardening period. This creates room to establish durable runtime,
storage, evidence and experiment contracts before external compatibility
expectations make those contracts expensive to change.

The priorities during this period are:

1. establish the intended architecture and ownership boundaries;
2. characterize existing numerical behavior before changing execution paths;
3. unify fragmented runtime behavior without flattening specialized methods;
4. make long simulations persistent, resumable and inspectable;
5. build evidence, conservation and diagnostics directly into the runtime;
6. create the research experiment and component systems;
7. harden the platform continuously through internal benchmarks;
8. design the Research Studio interface after the underlying contracts are
   stable;
9. publish only when the numerical and research workflows form one coherent
   system.

Validation remains mandatory, but its initial purpose is internal engineering
safety rather than public proof. Internal evidence should answer:

- did numerical results or solver decisions change;
- do conservation and physical invariants still hold;
- do independent formulations remain within their certified tolerances;
- can old checkpoints and result stores still be read;
- did runtime, memory or nonlinear robustness regress;
- are failures now easier to reproduce and diagnose.

Reports may remain under an internal validation store until PyMRST is ready for
external release. When a public benchmark report is eventually produced, it
should be generated from the same accumulated structured evidence rather than
assembled separately for publication.

## Robust implementation milestones

The destination should not be reduced to a small vertical slice. Robustness
comes from completing integrated platform layers with explicit acceptance
criteria. Before the five platform milestones, complete a private architecture
and characterization foundation.

### Foundation 0: architecture and characterization

Deliver:

- an inventory and call graph of schedule, packed-problem, adjoint,
  optimization and example-specific simulation entry points;
- a classification of which paths share nonlinear/timestep machinery and
  which require specialized protocols;
- characterization tests around current state transitions, timestep choices,
  controls, convergence reports and return contracts;
- architecture decision records for runtime, evidence, storage, schemas,
  compatibility and component ownership;
- versioning and migration policies for reports, checkpoints and result
  stores;
- an internal benchmark profile covering representative water, black-oil,
  compositional and poromechanics workflows;
- a baseline of numerical outputs, solver decisions, conservation, runtime and
  memory before cross-cutting changes begin;
- a risk-ranked ordinary module-coverage campaign running in parallel.

Acceptance:

- every supported execution path has a documented owner and call graph;
- core numerical behavior is protected by characterization or parity evidence;
- specialized adjoint/optimization paths have explicit integration contracts;
- architecture decisions identify compatibility guarantees and intentional
  breaking-change windows;
- baseline artifacts can be regenerated from recorded commands and inputs;
- no platform implementation begins with unresolved ownership of state,
  schedule, solver report or persistent storage schemas.

Foundation 0 is not a public adoption release and does not reduce the platform
scope. It is the internal regression and design baseline required to implement
the deeper architecture confidently.

### Milestone 1: unified runtime and evidence engine

Deliver:

- inventory and call graph of simulation entry points;
- compatibility-preserving `SimulationSession` integration;
- versioned evidence/report schemas;
- structured events and progress;
- convergence provenance and failure classification;
- model-aware conservation adapters;
- invariant profiles;
- checkpoint/streaming storage;
- reproducibility manifests;
- failure bundles.

Acceptance:

- all supported simulation entry points emit compatible evidence;
- parity/characterization results remain within certified tolerances;
- accepted report steps survive later failure or interruption;
- known phase-114 and Norne failure modes are diagnosed automatically;
- conservation is available for all primary fluid families and mechanics
  coupling;
- report/checkpoint schemas pass migration and corruption tests.

### Milestone 2: validation laboratory

Deliver:

- validation evidence graph;
- natural/overall, FI/sequential and FC/fixed-stress comparisons;
- AD/finite-difference validation;
- MRST and OPM adapters;
- analytic-solution registry;
- property, metamorphic and fault-injection suites;
- deck coverage and lineage;
- first published cross-validation report.

Acceptance:

- every supported comparison records convergence quality and independence;
- active-cell, unit, phase, component and well mappings are explicit;
- known reference-quality defects are detected;
- guardrails detect the injected unit/index/convention/topology faults;
- benchmark reports are reproducible from their manifests.

### Milestone 3: experiment and result platform

Deliver:

- experiment registry and variant matrix;
- parallel/resumable ensembles;
- scenario comparison;
- HDF5/Zarr/xarray/VTK/Parquet result interfaces;
- out-of-core state histories;
- publication packages;
- notebook experiment interface.

Acceptance:

- interrupted ensembles resume without repeating accepted members;
- experiment packages reproduce figures/tables from manifests;
- large cases operate without retaining all states in memory;
- scenario comparison aligns wells, regions and report times correctly;
- result-format round trips preserve declared numerical accuracy.

### Milestone 4: hybrid research platform

Deliver:

- component/capability registry;
- surrogate-component contracts;
- uncertainty and out-of-distribution guards;
- physics fallback and accuracy monitoring;
- adjoint/optimization integration;
- differentiable research interfaces;
- solver/backend performance laboratory.

Acceptance:

- learned components never execute outside their validity policy silently;
- accumulated simulation error and fallback coverage are reported;
- physics and surrogate variants run under identical experiment definitions;
- gradient results pass finite-difference or adjoint validation;
- backend speed claims include accuracy, hardware and field-scale evidence.

### Milestone 5: Research Studio and benchmark portal

Deliver:

- browser engineering/research workspace;
- live monitoring and failure diagnosis;
- evidence graph and scenario comparison views;
- benchmark publication portal;
- automated research reports;
- bounded browser/teaching runtime.

Acceptance:

- engineers can load, run, monitor, diagnose, resume and compare cases without
  reconstructing console output manually;
- researchers can define, execute and publish reproducible formulation or
  surrogate studies;
- release benchmark reports are accessible, versioned and independently
  reproducible;
- the interface exposes numerical evidence instead of hiding solver details.

## Program governance

To keep the platform robust rather than broad but shallow:

- each milestone must be completed and accepted internally end-to-end before
  the next is declared complete;
- schemas and public contracts require versioning and migration tests;
- every new component supplies invariants, conservation semantics and evidence
  hooks;
- every performance claim includes accuracy and field-scale profiles;
- every external comparison records version, mappings and known differences;
- every new research feature is represented in the benchmark suite;
- platform coverage and ordinary module coverage progress in parallel;
- incomplete capabilities remain explicitly marked experimental.

## Final position

The narrower-scope recommendation is useful for project management, but it should not
reduce PyMRST's destination. A small isolated feature would not create a reason
for reservoir engineers and researchers to move their work to PyMRST.

Nor does PyMRST need to rush toward public adoption before that destination is
coherent. The correct response is a strong unifying architecture, continuous
private numerical hardening and rigorous internal milestone acceptance.
Research Studio should combine trustworthy numerical simulation, independent
validation, reproducible experiments, resumable field workflows, hybrid
physics/ML research and engineer-friendly diagnosis in one open platform.

Public benchmarks, documentation and release artifacts should follow from the
same internal evidence system when the runtime, persistence, validation and
research workflows are mature. They should not dictate premature architecture
or force partial capabilities into externally stable contracts.

That would make PyMRST distinctive not because it is written in Python, but
because it can explain what it calculated, demonstrate why the result is
credible, diagnose why a run failed, compare alternative scientific methods,
and package the complete evidence needed for another researcher to reproduce
the conclusion.
