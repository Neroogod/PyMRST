# Research Studio package layout

Research Studio is an additive layer. MRST-compatible numerical owners remain
under their existing `pymrst` modules and must not import this package at module
load time.

## Canonical package boundaries

```text
pymrst/research_studio/
  runtime/       execution sessions, identity and cancellation
  evidence/      reports, events, manifests, invariants and failure evidence
  storage/       result-store protocol, HDF5 storage and checkpoints
  experiments/   definitions, variants, bounded execution and comparison
  case/          portable declarative case definitions and construction
  components/    capability, surrogate and guarded-fallback contracts
  optimization/  objectives, constraints and gradient evidence
  performance/   backend and solver qualification laboratories
  studio/        service, notebook, browser, benchmark and teaching surfaces
  validation/
    analytic/    benchmark contracts and registry types
    benchmarks/  executable analytic and manufactured benchmark adapters
    matlab/      reproducible MATLAB/MRST execution and fixture provenance
  eclipse/       ECLIPSE binary-result ingestion and normalized case records
  inspection/    grid, geometry and transmissibility diagnostics
  export/        normalized xarray/Zarr, tabular and VTK result formats
  units/         dimensional validation at the Research Studio boundary
```

Empty directories are not architecture: they create import surfaces without
owning behavior. The same rule applies to future validation adapters such as
`matlab/`, `opm/`, `properties/`, `metamorphic/` and `fault_injection/`.

The four additional top-level domains are deliberate refinements of the early
blueprint:

- `eclipse/` owns file-format ingestion, not validation against OPM;
- `inspection/` is usable without executing a qualification comparison;
- `export/` owns outbound formats independently of evidence rendering;
- `units/` enforces dimensional safety for every Research Studio boundary.

## Validation layout

Registry contracts and executable benchmarks are separated so importing
`pymrst.research_studio.validation.analytic` loads no simulator or benchmark.
The public validation package re-exports the stable names:

```text
validation/
  analytic/
    __init__.py
    contracts.py
  benchmarks/
    __init__.py
    foundation.py
  matlab/
    adapter.py
    execution.py
    fixtures.py
    mappings.py
    provenance.py
    schemas.py
  comparison.py
  derivatives.py
  graph.py
  mappings.py
```

The implemented experiment boundary is:

```text
experiments/
  definition.py   canonical ExperimentDefinition v1 and content identity
  variants.py     explicit, Cartesian and sampled matrices with deterministic IDs
  registry.py     collision-safe versioned persistence
  execution.py    independent member stores, checkpoints and lazy array access
  ensemble.py     bounded memory-aware local processes, resume and aggregation
  comparison.py   aligned field, region, well, solver and performance comparison
  runners.py      repository-owned executable member adapters
  branching.py    checkpoint ancestry, changed inputs and compatibility refusal
  publishing.py   verified research packages and reproduction recipes

export/
  dataset.py      normalized xarray datasets and Zarr round trips
  tabular.py      long-form CSV and Parquet summaries
  vtk.py          ParaView VTK collections, fields and well trajectories

performance/
  laboratory.py   cold/warm timing, allocation and numerical drift evidence
  patterns.py     immutable CSR structures, explicit cache invalidation and batches
  solvers.py      reference and availability-aware linear-solver evidence

studio/
  service.py      UI-independent project, workflow, run and resource lifecycle
  metadata.py     transactional SQLite/WAL project, run and event metadata
  processes.py    import-owned subprocess execution and hard timeout boundary
  results.py      root-confined lazy member-store and HTML-viewer catalog
  web.py          authenticated HTTP contract and engineer workspace
  notebook.py     safe rich notebook summaries over service objects
  benchmarks.py   versioned publication-state benchmark portal
  teaching.py     honestly bounded instructional execution policy
```

`foundation.py` holds the nine tightly coupled M2.4 foundation cases and their
single registry factory. It should split by physical family only when a second
registry or independent benchmark lifecycle makes that separation useful.

## Dependency direction

```text
studio (future) / experiments
          |
          v
runtime -> evidence -> storage
   |          |
   +----------+----> validation
                         |
                         v
              existing PyMRST numerical owners
```

Numerical owners never depend upward on Research Studio. Validation adapters
may call them, but must record assumptions, mappings, tolerances and evidence
independence explicitly.
