# PyMRST shale module: end-to-end implementation plan

## 1. Objective

Port the MRST 2024a `modules/shale` module into
`pymrst/modules/shale` as a native Python implementation while preserving the
MRST owner layout, numerical meaning, examples, datasets, and validation
workflows.

This is not achieved by exposing generic HFM functions under shale names. A
shale owner is complete only when its Python implementation:

1. accepts the corresponding MRST inputs and options;
2. produces the same grid, rock, well, operator, equation, or result semantics;
3. has focused Python tests;
4. has direct MATLAB fixture comparison where deterministic comparison is
   possible; and
5. participates successfully in at least one complete shale simulation.

## 2. Audit baseline

The reference tree is `mrst-2024a-Geomech/modules/shale`.

| Item | MRST reference | Current PyMRST |
|---|---:|---:|
| Total reference files | 56 | 1 source file |
| MATLAB files | 48 (47 executable owners plus `examples/Contents.m`) | 0 native ports |
| Bundled MAT datasets | 5 | 0 |
| Bundled CSV references | 1 | 0 |
| Dedicated shale tests | N/A | 0 |

The fracture-framework dependency for the shale port is the consolidated
`pymrst.modules.hfm` owner. The former `pymrst.modules.hfm_v2` package contained
only re-exports of these exact objects and was removed; this consolidation does
not claim that the wider MRST `hfm_v2` module has been ported. The current
`pymrst/modules/shale/__init__.py` defines `build_shale_edfm` and
`build_shale_edfm_3d` as direct calls to the generic builders.
`validate_phase106.py` checks only that the import surface exists.
This is useful scaffolding, but it does not implement MRST shale behavior.

Existing PyMRST `hfm` and compositional code should be reused as lower-level
infrastructure where its behavior is demonstrated equivalent. Shale-specific
algorithms must live in shale-specific owners and must not be replaced with
aliases. Any additional fracture behavior needed by shale must be ported into
the consolidated HFM layout rather than recreated behind another forwarding
package.

### Implementation progress

WP1 is in progress. The MRST-shaped package directories now exist, all five MAT
assets and the CMG CSV are packaged unchanged, and native owners have been added
for `makeShaleRock`, `SorbedCompositionalFluid`, `Pts2CellIdx`, `gangiFn`,
`createMultiStageHFs`, `meshHFsystem`, and `processStochFracs`. Focused Python
tests cover their current contracts. Initial WP2/WP3 infrastructure is also in
place: typed global NNC/operator contracts, HFM fracture/matrix metadata,
global cell assembly, `transmultpEDFM`, and pEDFM TPFA connection/divergence
assembly. Native `setupShaleEDFMOpsTPFA` and pEDFM operators now expose pore
volume, adsorption grain volume, matrix masks, gradient, averaging, upwind,
divergence and accumulation-divergence with AD propagation. A direct MRST 2024a
fixture now certifies `makeShaleRock`,
`Pts2CellIdx`, `gangiFn`, and `createMultiStageHFs`. The complete projected-NNC
owner is implemented for six-face hexahedral cells and its centered,
axis-aligned topology, areas, directions and transmissibilities match a direct
MRST fixture. Exact projected polygon union now covers partial overlap,
containment and disjoint regions, and affine-skew six-face cells are tested.
Additional MATLAB fixtures for the remaining owners are outstanding.
Native `EDFMshalegrid` preprocessing now converts rectangular `FracturePlane`
objects into HFM projections, global fracture cells, exact fracture/matrix
NNC metadata and simulation-ready operators while retaining processed and
fracture-grid mapping metadata. Per-fracture aperture, permeability and porosity
are preserved, and repeated calls add newly selected planes with stable source
ordering through a deterministic rebuild. Convex polygonal fractures are
conservatively clipped across cells, including exact oblique plane/AABB
cross-sections, with area and centroid conservation tests. The projection now
uses each matrix cell's actual face-edge graph, with a tetrahedral fixture
certifying exact general-polyhedral clipping. Physical fracture
node/face/cell topology is now appended to a checked global `Grid`, including
deduplicated shared fracture edges, neighbors, geometry and explicit global
maps. Direct MATLAB topology certification remains.
Finite convex fracture-plane intersection and cross-fracture NNC creation now
segment long intersection lines across all participating fracture-cell pairs.
Tests certify conserved line coverage and two-sided transmissibility for mixed
aperture/permeability. Direct MATLAB fracture-fracture certification remains.
The shale well stack now includes finite well/plane intersection mapping,
rectangular fracture-cell WI, natural-fracture completion disabling, and a
standard PyMRST `Well` assembler. Direct MRST fixtures certify the intersection
point/plane/cell and WI. The owner now also accepts ordinary cell vectors,
logical masks, matrix-only and mixed matrix/fracture completions, directional
WI, partial negative-WI recomputation, Kh/cell-dimension overrides,
representative-radius control, VFP index and the legacy `c` payload. A direct
MRST fracture-completion fixture certifies cell mapping, WI, representative
radius, depth, sign correction, VFP index and legacy payload.
Fracture/matrix NNC assembly now reproduces the MRST/Tene surface adjustment:
interior fracture cells receive the two-surface multiplier, while a fracture on
a shared matrix face remains one fracture cell connected to both matrix
neighbors. Repeated processing merges without duplicating existing pairs.
The two MRST-shaped compositional registries now have their proper boundaries.
The `compositional_fns` owner supplies its 18 cases and the `utils` owner adds
seven oil, SlotDrill and modified-Barnett cases while reusing the common main
definitions. Together they cover delegated base mixtures, table-backed
diagnostic and tracer mixtures, Barnett
sorption/diffusion variants, oil/N2 variants, C1-C2-C3, Bakken, Eagle Ford and
both Bakken-light conditions. Exact component ordering, outward aliases,
streams, pressure/temperature, BIC, isotherm, diffusion and Parachor metadata
are covered by focused tests. A reproducible direct MRST fixture now certifies
all returned thermodynamic arrays and metadata for both catalogs. The fixture
also records MRST's malformed eight-entry Eagle Ford injection stream; PyMRST
intentionally provides the dimensionally valid six-entry equivalent.
Both fracture-system plotting owners are now native PyVista implementations.
They retain fracture subset ordering and labels, distinguish hydraulic and
natural fractures, render well trajectories/intersection points, resolve
global fracture completion IDs to cell centroids, share interactive plotters,
and return the three MRST handle groups alongside richer typed actor metadata.
Off-screen structural regressions cover these engineering display contracts;
direct visual MATLAB certification remains.
The first two reference examples, `gangiPlot` and `langmuirPlot`, are ported as
reusable workflows rather than stateful scripts. Each returns its complete
reference curve for research and regression use, while optional Matplotlib
rendering preserves the MRST axes and presentation. Their default sample
counts, physical parameters, domains and numerical values are tested.
The complete Eagle Ford construction/simulation implementation now lives in
the MRST-named public `examples/eagle_ford.py` owner, and the complete
sorption/mechanism implementation lives in `examples/sorption.py`. The former
private catch-all files `_eagle_ford.py` and `_mechanism_comparison.py` were
removed. EDFM, full-field, diffusion, Gangi and accelerated owners reuse these
native construction kernels only for the calculations that are genuinely
identical between the corresponding MRST scripts; their discretization,
field-size, mechanism and backend choices remain explicit in their own public
owners. No example forwards to an external implementation or compatibility
placeholder.
`examples/eagle_ford_edfm.py` is additionally a self-contained owner-level
port rather than a method-selection facade: it explicitly constructs the MRST
physical parameters, hydraulic/natural fractures, fracture-aligned matrix
grid, EDFM and fracture-fracture NNCs, well intersections and completion
indices, shale EDFM TPFA operators, Eagle Ford compositional model, initial
state, 15-year schedule and nonlinear simulation.
`examples/eagle_ford_full.py` is likewise a self-contained seven-stage pEDFM
port. It owns the 1,200 x 600 x 76.2 m field parameters, 21 hydraulic
fractures, 4,200-fracture bundled realization, 920 m producer, full EDFM/NNC
preprocessing, projected connections and multipliers, pEDFM TPFA operators,
Eagle Ford compositional initialization and 15-year schedule.
`examples/diffusion.py` is also a self-contained owner-level port. It directly
owns the Barnett grid and petrophysics, ten-stage/full and bounded regression
fracture systems, well/fracture completion mapping, diffusion coefficients and
tortuosity, shale EDFM operators, diffusion-enabled `NatVarsShaleModel`,
disabled reference model, initial composition gradient, paired schedules and
nonlinear executions.
`examples/gangi.py` is a self-contained owner-level port as well. It explicitly
owns the Barnett grid/fractures/well/operators, MRST confining and closure
pressures, Biot coefficient, roughness exponent and reference permeability,
the pressure-dependent Gangi multiplier, enabled/disabled compositional
models, initial states, diagnostics and paired nonlinear schedules.
`examples/eagle_ford_accelerated.py` is a self-contained accelerated pEDFM
port, not a backend-selection wrapper. It directly owns the short Eagle Ford
geometry/fractures/well, EDFM and projected-NNC assembly, pEDFM operators,
fluid/model/state/schedule, and explicitly installs the compiled row-major
`DiagonalAutoDiffBackend` corresponding to the MRST MEX workflow.
`examples/eagle_ford_edfm_accelerated.py` independently owns the corresponding
accelerated EDFM workflow: its short-field inputs, fracture topology/NNCs,
shale EDFM operators, well/model/state/schedule and compiled row-major diagonal
backend are assembled directly in the MRST-named owner.
`examples/eagle_ford_edfm_full.py` is the corresponding self-contained
seven-stage full-field EDFM port. It directly owns the 21-HF/4,200-NF inputs,
920 m producer, complete EDFM/NNC preprocessing, shale EDFM TPFA operators,
compositional model/state and 15-year schedule.
All four deterministic SPE-201243-PA geometry owners are now native and
directly MATLAB-certified: inclined multistage fracture construction,
fracture-aligned structured grids, and the explicit single-/three-fracture
stencils. Certification covers complete coordinate vectors, grid dimensions,
fracture vertices/centroids, fracture-cell IDs and well IDs with index-base
conversion only at the fixture boundary. Their builders return grids and
metadata without the reference scripts' unconditional plotting side effects.
The shared `geomspace` root selection was also hardened to reject the
extraneous algebraic q=1 root when roundoff places it microscopically above one.
The sorption, diffusion and Gangi comparison examples now build the actual
shale stack end-to-end: matrix grid and shale rock, staged fracture polygons,
global fracture topology/NNCs, horizontal well intersections, fracture
completions, shale TPFA operators, Barnett mixture, natural-variable models,
initial states and paired enabled/disabled schedules. Compact acceptance cases
run both nonlinear models and expose research-friendly diagnostics for sorbed
storage, component gradients and Gangi multipliers. `full_size=True` restores
the MRST 61x11x4 domain, ten stages, original petrophysics, BHP producer and
15-year timesteps; recorded full-size MATLAB time histories remain outstanding.
The general `compositional3D` waterflood is also native. It preserves the three
MRST planes (including the displaced slanted plane), their heterogeneous
global fracture cells, fracture-matrix and fracture-fracture NNCs, two-component
fluid, vertical water injector, BHP producer and ramped five-year schedule.
The reduced acceptance case advances a complete nonlinear report step with
positive water injection and negative oil production; `full_size=True`
restores the 25x10x5 reference grid and 51-step schedule. Direct full-size
MATLAB production/state histories remain outstanding.
`stochasticFracs` now loads the unchanged MRST realization into 350 hydraulic
and 350 natural typed polygons, removes MATLAB closing rows at the typed-plane
boundary, preserves source indices/set membership, and reports planar versus
non-planar classification and areas. A seeded external-generator contract
supports ADFNE or research alternatives, and callers may render the two sets
or preprocess a bounded/full selection into the live shale EDFM topology.
The four non-accelerated Eagle Ford owners now share a native typed numerical
workflow while retaining distinct MRST-layout entry modules. The short case
reconstructs its 1,976-cell fracture-aligned matrix, three hydraulic fractures,
150 bundled natural fractures and horizontal producer; the full case exposes
seven stages, 21 hydraulic fractures and all 4,200 bundled natural fractures.
Both EDFM and pEDFM branches build global fracture topology, fracture-fracture
connections, well/fracture completions, their distinct TPFA operators, the
Eagle Ford compositional model, initial state and 15-year schedule. A bounded
natural-fracture option makes preprocessing testable without changing the
full-field input contract. Well equivalent radii are derived from each actual
fracture-cell polygon, matching MRST's geometric WI calculation rather than an
arbitrary constant. Reduced preprocessing/operator tests and a live BHP
production step pass. The original first 79.1-second EDFM report step also
converges after one cut into two 39.55-second mini-steps. Complete histories,
pEDFM acceptance and direct MATLAB certification remain open.
The two MEX-labelled Eagle Ford owners are also present in their own reference
layout modules. They select PyMRST's row-major diagonal AD backend and its
compiled Numba kernels, which is the supported Python analogue of MRST's
`DiagonalAutoDiffBackend(useMex=true)`. Tests verify backend availability and
every requested compiled kernel. A real nonlinear assembly reaches this path,
but its cold JIT/diagonal solve was slower than the sparse path on the reduced
2,000-cell case; performance claims therefore remain gated on warm-run timing
and complete convergence evidence.
The SPE Figures 10--13 sealing and conductive Tene cross cases are now native
owners. Each constructs the exact 27x27x3 matrix and two perpendicular finite
fractures, separate EDFM and projected-EDFM operators, the compressible-water
boundary-value problem, and an aperture-resolved explicit reference grid. The
full 225x225x2 reference is retained while a 45x45x2 option supports routine
acceptance. Both permeability extremes advance all three discretizations in
focused live solves. This work also corrected the shared pEDFM operator owner:
callers can now supply physical global fracture-cell volumes, preventing zero
fracture pore volume and invalid CNV scaling.
Both SPE validation Case 2 owners are now native. The graded script preserves
its 6,405-cell Cartesian matrix while the structured script preserves its
distinct 6,615-cell matrix; both embed the two nanodarcy barriers and central
50-darcy fracture, map the short horizontal well only to that central plane,
and build pEDFM, EDFM and aperture-resolved explicit compositional variants.
The original 75-entry CMG report schedule is packaged and attached. Focused
three-method nonlinear acceptance passes; complete CMG report histories and
direct MATLAB rate/pressure profiles remain the certification gate.
The final two missing executable owners, the Bakken EDFM and pEDFM paper
workflows, are now present. They preserve the 4,004-cell structured matrix,
eight stages and 24 hydraulic fractures, two 500-plane natural-fracture
families, 910 m horizontal producer, Bakken fluid, initial saturations and
30-year ramp. MRST generates the DFN through external ADFNE and distributes no
realization; PyMRST therefore provides a deterministic bounded-lognormal
two-family generator matching the published count, support, azimuth and dip,
plus an injection point for an exact external realization. Bounded/full DFN
preprocessing is selectable. Both distinct operator stacks build with positive
fracture PV, and the hydraulic-network EDFM nonlinear solve plus original first
report steps for both branches have been exercised successfully. Full
1,000-plane preprocessing and 30-year MATLAB histories remain acceptance work.
Direct MRST operator-pipeline certification now runs the reference
`EDFMshalegrid`, fracture/matrix NNC, pEDFM projection and both TPFA assemblers
on the same 3x3x3 fractured grid. It exposed and corrected two material issues:
the Python assemblers had omitted 12 internal fracture-cell connections, and
fracture/matrix CI transmissibilities lacked MRST shale's two-surface factor.
The corrected EDFM graph matches all 75 MRST active pairs and values. pEDFM
matches its 96-edge topology, projected matrix/fracture and matrix/matrix rows,
projected areas, and pore volumes. The direct fixture also exposes a reference
defect in MRST 2024a: projected `G.nnc.T` rows are inconsistent with the saved
projected areas, including zero transmissibility for positive area and nonzero
transmissibility for zero area. PyMRST retains the physical invariant that
projected transmissibility is positive exactly when projected area is positive.
This intentional correction gives a documented 2.7% sorted transmissibility-
vector difference; the owner remains Implemented rather than being falsely
marked as bitwise Validated.
The `NatVarsShaleModel` owner and independent competitive-Langmuir, gas
diffusion, Gangi-velocity and Forchheimer/non-Darcy kernels now exist. They are
connected through backward-compatible natural-variable assembly hooks:
competitive Langmuir modifies component accumulation, diffusion modifies gas
component flux, and Gangi/non-Darcy modify phase velocities before mass fluxes
are formed. Pressure/composition AD propagation and matrix/fracture masking are
tested. Direct MATLAB residual/Jacobian and conservation fixtures remain.

## 3. Required MRST-shaped Python layout

Python filenames use snake case, but each file has one documented MRST owner.
The directory organization mirrors the reference module.

```text
pymrst/modules/shale/
├── __init__.py
├── README.md
├── spe201243pa/
│   ├── data/cmg_timestep2.csv
│   ├── multistage_planar_nonplanar_hf.py
│   ├── structured_grid_fracture_on_face.py
│   ├── explicit_stencil.py
│   ├── explicit_stencil_3_fracs.py
│   └── fig_*.py
├── compositional_fns/
│   ├── nat_vars_shale_model.py
│   ├── sorbed_compositional_fluid.py
│   ├── equations_shale_natural_vars.py
│   └── get_shale_comp_fluid_case.py
├── hfm_fns/
│   ├── edfm_shale_grid.py
│   ├── assemble_shale_global_grid.py
│   ├── fracture_matrix_shale_nncs_3d.py
│   ├── fracture_fracture_shale_nncs_3d.py
│   └── setup_shale_edfm_operators_tpfa.py
├── other_mrst_fns/
│   ├── add_well_shale.py
│   ├── compute_wi_shale.py
│   ├── make_shale_rock.py
│   └── well_fracture_nncs_3d.py
├── pedfm_nnc/
│   ├── projected_matrix_fracture_nncs_3d.py
│   ├── setup_pedfm_operators_tpfa.py
│   └── transmissibility_multiplier_pedfm.py
├── utils/
│   ├── points_to_cell_indices.py
│   ├── create_multistage_hfs.py
│   ├── gangi.py
│   ├── get_shale_comp_fluid_case.py
│   ├── mesh_hf_system.py
│   ├── plot_frac_system.py
│   ├── plot_frac_system_nf.py
│   └── process_stochastic_fractures.py
└── examples/
    ├── data/*.mat
    ├── compositional_3d.py
    ├── diffusion.py
    ├── eagle_ford*.py
    ├── gangi.py
    ├── gangi_plot.py
    ├── langmuir_plot.py
    ├── sorption.py
    └── stochastic_fractures.py
```

Compatibility exports may be added in `shale/__init__.py` after the actual
owners exist. They must import those owners rather than implement parallel
kernels or redirect shale functions to generic HFM.

## 4. Numerical data contracts

Before porting algorithms, define typed structures that preserve the fields
used across the MATLAB module:

- fracture planes: polygon vertices, normal, aperture, porosity,
  permeability, fracture set and processed-state metadata;
- matrix and fracture subgrids plus global cell/face/node maps;
- NNC records: neighbor cells, transmissibility, contact/projection area,
  type, normal, projection direction, and matrix-matrix projection pairs;
- shale rock: permeability tensor, porosity, net-to-gross, `is_matrix`, and
  enabled shale mechanisms;
- wells: matrix and fracture completions, well indices, segment geometry,
  reference depth, control and component composition;
- shale mechanism options: sorption/isotherm data, diffusion coefficients,
  Gangi parameters and non-Darcy-flow parameters.

All public APIs use zero-based indices. MATLAB fixtures must convert indices at
the fixture boundary only. Array ordering and global/local maps must be tested
explicitly because an apparently correct pressure solution can hide a topology
permutation.

## 5. Work packages

### WP0 — Reproducible audit and fixture harness

- Add `scripts/audit_shale_parity.py` to compare MRST owners, Python owners,
  exports, tests, data, and fixture coverage.
- Generate `docs/shale/parity.csv`; do not infer parity from filename alone.
- Add MATLAB fixture generators under `scripts/matlab/shale/`.
- Store compact, deterministic fixtures under `tests/fixtures/shale/` with the
  MRST version, generating script, units, index conversion and tolerances.
- Add separate pytest markers for unit, MATLAB-parity, example, slow, and
  renderer tests.

Exit gate: every one of the 47 executable MRST owners is represented in the
ledger and no owner can be marked validated without a named test or fixture.

### WP1 — Package layout, data loading and elementary utilities

Port first:

- `Pts2CellIdx` with boundary and tolerance behavior;
- `gangiFn`, including AD-compatible derivatives;
- `makeShaleRock`, including scalar/diagonal/full tensor expansion,
  `ntg`, warnings, and `isMatrix` semantics;
- `processStochFracs`;
- `createMultiStageHFs` and the SPE multistage/non-planar variant;
- `meshHFsystem`, `StructuredGridFractureonFace`, `explicitStencil`, and
  `explicitStencil3Fracs`.

Copy the five MAT datasets and the CMG CSV into package data without rewriting
the reference files. Add loader functions using `importlib.resources` and
`scipy.io.loadmat`, with explicit conversion of MATLAB structs and indices.

Exit gate: deterministic geometry, property and dataset-loader fixtures pass
against MATLAB for planar, non-planar, structured and stochastic cases.

### WP2 — EDFM shale grid and global topology

Port the actual behavior of:

- `EDFMshalegrid`;
- `assembleShaleGlobalGrid`;
- `fracturematrixShaleNNC3D`;
- `fracturefractureShaleNNCs3D`.

The implementation must support multiple fracture planes, fracture-plane
intersections, incremental preprocessing, local-to-global mapping, duplicate
intersection suppression, matrix/fracture rock fields and NNC labels. Existing
Consolidated `hfm` geometry functions may supply polygon intersection
primitives, but the
shale global-grid and metadata contracts remain owned here.

Required tests include fractures inside cells, on cell faces, through edges and
vertices, parallel/coplanar planes, intersecting fracture networks, small
sliver intersections, non-planar hydraulic fracture sets, anisotropic rock and
repeat preprocessing.

Exit gate: cell/face/node counts, centroids, volumes/areas, global maps, NNC
pairs, NNC areas and NNC transmissibilities match MATLAB fixtures.

### WP3 — Projection-based EDFM

Port:

- `pMatFracNNCs3D` and all of its projection geometry helpers;
- `transmultpEDFM`;
- `setupPEDFMOpsTPFA`.

This is the defining shale-module capability and the largest kernel. It must
construct projected matrix-fracture and associated matrix-matrix connections,
compute union projection areas, projection direction/type, harmonic
permeability and transmissibility, and modify the normal TPFA
transmissibilities without losing connection ordering.

Do not equate the present generic `build_edfm_3d` face-coincidence behavior with
pEDFM. pEDFM parity requires the reference projection areas, projection
neighbors, transmissibility multipliers and pressure/flux response.

Exit gate: direct MATLAB comparisons pass for each projection direction and
for crossed, overlapping, face-aligned and non-planar fracture configurations;
the SPE validation cases reproduce the EDFM, pEDFM and explicit-grid profiles.

### WP4 — Shale EDFM operators and wells

Port:

- `setupShaleEDFMOpsTPFA`;
- `computeWIShale`;
- `addWellShale`;
- `wellfractureNNCs3D`.

Preserve matrix, hydraulic-fracture and natural-fracture completion behavior;
Peaceman and EDFM equivalent radii; anisotropic permeability; well direction;
skin, radius, depth and control options; and direct well-fracture intersections.
The operator assembly must retain NNC ordering, transmissibility, pore volume,
gradient/divergence and adsorption pore-volume behavior.

Exit gate: direct fixtures match WI, equivalent radius, completion depths,
well-fracture intersections, operator sparse matrices, pore volumes and one-step
well residuals.

### WP5 — Shale compositional thermodynamics and equations

Port:

- `SorbedCompositionalFluid` as a real subclass of PyMRST's table
  compositional mixture;
- both MRST `getShaleCompFluidCase` variants, while documenting their
  duplicate/different case tables and exposing unambiguous owners;
- `NatVarsShaleModel`;
- `eqnsShaleNaturalVars`.

Start from the validated PyMRST natural-variable compositional formulation,
then add and directly test every shale modification:

- Langmuir sorbed-component accumulation;
- gas molecular diffusion;
- Gangi pressure-dependent permeability applied consistently to phase fluxes;
- non-Darcy/turbulence corrections;
- pEDFM/EDFM NNC fluxes;
- water and component accumulation, wells, boundary conditions, phase-state
  switching, minimum fractions and state flux outputs.

All new physics must operate on NumPy values and PyMRST AD variables. Finite
difference derivative checks are mandatory for sorption, Gangi, diffusion and
the combined component residual.

Exit gate: values and Jacobians for accumulation, each phase flux, diffusion,
sorption, Gangi scaling, wells and full residual blocks match direct MATLAB
fixtures; closed-system component mass balance meets the documented tolerance.

### WP6 — Utilities, plots and examples

Port the example catalog and all 13 executable examples:

- elementary: `gangiPlot`, `langmuirPlot`, `stochasticFracs`;
- physics isolation: `sorption`, `diffusion`, `gangi`;
- integrated: `compositional3D`;
- Eagle Ford: pEDFM/EDFM, short/full fracture systems, and accelerated variants.

Port `plotFracSystem` and `plotfracSystem_NF` through PyMRST plotting APIs. Every
example must expose a callable `run(...)` that returns structured results and a
CLI entry point. Plotting must be optional for headless tests.

The MATLAB `_MEX` examples are workflow variants, not instructions to create
fake MEX wrappers. Provide an optional accelerated Python path using vectorized
NumPy/SciPy and, only if profiling justifies it, Numba/Cython. The accelerated
and reference paths must produce equivalent results.

Exit gate: all examples run headlessly from a clean checkout; Eagle Ford cases
produce matching pressure, saturation, composition, well-rate and cumulative
production histories at common report times.

### WP7 — SPE-201243-PA and external-result certification

Port all ten scripts in `SPE201243PA`, preserving their role as reproducible
figure/data workflows rather than interactive notebooks. Validate:

- low- and high-fracture-permeability Tene cases;
- validation case 2 on pEDFM, EDFM and explicit grids;
- structured fracture-on-face behavior;
- Bakken EDFM and pEDFM production cases;
- the bundled CMG time-series comparison.

Each workflow writes machine-readable metrics in addition to optional figures:
mesh/NNC statistics, nonlinear iterations, timestep cuts, material-balance
error, pressure-profile norms, well-rate errors and cumulative-production
errors.

Exit gate: regenerated metrics and figures meet documented tolerances against
MATLAB and, where supplied, CMG data.

### WP8 — Robustness, performance and release gate

- Run parameterized geometry/property tests across structured, corner-point and
  general polyhedral grids supported by the reference algorithms.
- Test units, empty sets, repeated fractures, inactive cells, degenerate
  polygons, zero/near-zero porosity and invalid options.
- Profile preprocessing, sparse assembly, residual/Jacobian construction and
  nonlinear solves on the full Eagle Ford case.
- Add checkpoint/restart and deterministic result serialization for long runs.
- Document unsupported external dependencies such as ADFNE rather than silently
  substituting a different stochastic generator. Bundled `StochasticFracs.mat`
  must keep the standard examples runnable without ADFNE.
- Run the full Python suite and the MATLAB cross-validation suite.

Exit gate: every owner in `shale_parity.md` is `Validated`, every required data
asset is packaged, all examples run, and no completion claim relies solely on
an import test or shared-module alias.

## 6. Validation matrix

| Layer | Required evidence |
|---|---|
| Geometry | Counts, topology, maps, intersection polygons, areas, normals |
| EDFM NNC | Pairs, types, CI, areas and transmissibilities |
| pEDFM | Projected pairs/areas/directions, multipliers and modified face fluxes |
| Rock | Shape expansion, tensor ordering, porosity, NTG and matrix flags |
| Wells | Completion mapping, WI, equivalent radius, depth and controls |
| Fluid | Component tables, isotherms and case metadata |
| Equations | Residual values, sparse Jacobian blocks, sources and state updates |
| Conservation | Water and per-component material-balance errors |
| Simulation | Pressure, saturation, composition, rates and cumulative production |
| Presentation | Plot data/mesh statistics first; renderer-safe image checks second |

Tolerances must be quantity-specific. Topology and integer maps require exact
agreement. Geometry and operators use scale-aware absolute/relative tolerances.
Simulation comparisons use common report times and include conservation errors;
matching a plot by eye is not sufficient.

## 7. Dependency policy

The implementation may depend on completed PyMRST owners for grids, TPFA, AD,
compositional EOS/flash, schedules, wells, plotting and consolidated `hfm`
primitives. The shale package uses `pymrst.modules.hfm` as its HFM-generation
dependency; the obsolete forwarding-only `hfm_v2` package must not be
reintroduced. For each dependency, tests must
show that the precise behavior needed by shale exists.
If not, fix the lower-level owner rather than embedding a second generic solver
inside shale.

This policy preserves PyMRST as a Python replica of MRST while adding the shale
module on top of it. It also keeps shale-specific pEDFM and physics visible in
the same conceptual owners as MRST.

## 8. Recommended implementation order

1. WP0 audit and fixture harness.
2. WP1 data contracts, datasets and elementary utilities.
3. WP2 full 3D EDFM shale topology.
4. WP3 pEDFM projection kernel.
5. WP4 operators and wells.
6. WP5 compositional shale physics and AD derivatives.
7. WP6 examples.
8. WP7 paper/CMG certification.
9. WP8 performance and release gate.

The critical path is WP2 → WP3 → WP4 → WP5. Example ports should not precede
their numerical owners, because that produces runnable scripts backed by the
wrong physics.

## 9. Definition of end-to-end complete

The module is complete only when all of the following are true:

- the MRST directory and owner layout has a native Python counterpart;
- all 47 executable owners are implemented and directly tested;
- all six bundled data assets are loadable through package resources;
- EDFM and pEDFM geometry, NNCs and operators match MATLAB;
- shale sorption, diffusion, Gangi and non-Darcy terms match in residuals and
  Jacobians;
- well/fracture coupling matches MATLAB;
- all 13 normal examples and all 10 SPE workflows run headlessly;
- Eagle Ford and SPE cases have recorded MATLAB/CMG regression evidence;
- reference and accelerated Python paths agree; and
- `shale_parity.md` contains no `Missing`, `Surface only`, `Partial`, or
  `Unvalidated` numerical owner.

The executable audit and qualification entrypoints are:

```bash
python3 scripts/audit_shale_parity.py
python3 scripts/qualify_shale.py
python3 scripts/qualify_shale.py --matlab
```

The audit writes `docs/shale/parity.csv` and `SHALE_PARITY_AUDIT.md`. The
qualification gate writes a timestamped, machine-readable command/result record
to `results/shale/qualification.json` by default.
