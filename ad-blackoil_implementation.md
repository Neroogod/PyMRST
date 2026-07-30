# PyMRST `ad-blackoil` End-to-End Implementation Plan

## 1. Purpose

This document defines the work required for
`pymrst/modules/ad_blackoil` to match MRST 2024a's
`modules/ad-blackoil` module in layout, public behavior, numerical results,
examples, and validation coverage.

The target is not filename parity alone. Completion means that a supported
MRST black-oil case can be expressed through the equivalent PyMRST API, run
through a complete timestep schedule, and produce matching reservoir states,
well solutions, nonlinear reports, and conservation behavior within declared
tolerances.

The reference source tree is:

```text
mrst-2024a-Geomech/modules/ad-blackoil/
```

The implementation tree is:

```text
PyMRST/pymrst/modules/ad_blackoil/
```

## 2. Definition of done

The module is end-to-end complete when all of the following are true:

1. Every MRST `.m` file is represented by a Python owner, a documented
   consolidated implementation, or an explicit non-code exclusion.
2. The `components`, `models`, `utils`, and `examples` directory layout mirrors
   MRST with importable snake-case Python modules.
3. Single-phase water, oil-water, and three-phase black-oil models support the
   same relevant phase and dissolution combinations as MRST.
4. `DISGAS` and `VAPOIL` work independently and together, including phase
   appearance and disappearance.
5. Wells, changing controls, limits, sources, boundary conditions, gravity,
   aquifers, thermal forces, and multisegment wells work through schedules.
6. Public utility functions reproduce MRST's input/output contracts rather
   than existing only as private logic inside a model.
7. Supported decks and schedules can be imported without case-specific parser
   code.
8. Every shipped Python example either runs or is classified as a deliberate
   data-only exclusion. Placeholder files that raise `NotImplementedError` do
   not count as implemented.
9. MATLAB reference fixtures validate utility outputs, equation residuals,
   Jacobians, nonlinear updates, complete timesteps, and benchmark histories.
10. The Python test suite and parity audit run cleanly from a fresh checkout.

## 3. Current baseline

The existing implementation is substantial. The following model owners are
already present:

| MRST model | PyMRST owner | Current status |
|---|---|---|
| `GenericBlackOilModel.m` | `models/generic_black_oil_model.py` | Broad graph-based implementation |
| `ThreePhaseBlackOilModel.m` | `models/three_phase_black_oil_model.py` | Working reference model, primarily dissolved-gas workflow |
| `TwoPhaseOilWaterModel.m` | `models/two_phase_oil_water_model.py` | Working oil-water model |
| `WaterModel.m` | `models/water_model.py` | Working single-phase water model |
| `WaterThermalModel.m` | `models/water_thermal_model.py` | Delegates to the shared geothermal mass-energy kernel |

The oil and gas component classes are present. Core support also exists for
black-oil PVT, capillary pressure, J-function scaling, wells, boundary
conditions, sources, gravity, aquifers, control switching, equilibration,
multiperforation well columns, and combined dissolved-gas/vaporized-oil
calculations.

SPE1 and several synthetic examples run. Existing MATLAB reference validators
cover important fluid, residual, state-function, well-control, capillary,
aquifer, and simulator paths. These capabilities form the baseline and must
remain green throughout the remaining work.

### Implementation progress

The first implementation batch is complete:

- `get_b_o_bo.py` ports `getbO_BO.m`;
- `get_b_g_bo.py` ports `getbG_BO.m`;
- `calculate_hydrocarbons_from_status_bo.py` ports
  `calculateHydrocarbonsFromStatusBO.m`;
- the oil-water, three-phase, and generic model paths use the shared helpers;
- numeric and ADI derivative tests cover the new utilities;
- paired MATLAB fixtures validate dead/live oil, dry/wet gas, all three
  hydrocarbon status modes, and disabled dissolution;
- the ad-blackoil parity auditor and its 57-row baseline ledger are present.

The generated baseline currently reports 30 ported, 3 partial, 15 pending,
and 9 catalog entries. This is a progress measurement, not a completion claim.

## 4. Required final layout

The target utility layout is:

```text
pymrst/modules/ad_blackoil/
├── components/
│   ├── gas_component.py
│   └── oil_component.py
├── models/
│   ├── generic_black_oil_model.py
│   ├── three_phase_black_oil_model.py
│   ├── two_phase_oil_water_model.py
│   ├── water_model.py
│   └── water_thermal_model.py
├── utils/
│   ├── calculate_hydrocarbons_from_status_bo.py
│   ├── compute_flash_black_oil.py
│   ├── compute_init_aquifer.py
│   ├── equations_black_oil.py
│   ├── equations_oil_water.py
│   ├── equations_water.py
│   ├── equations_water_thermal.py
│   ├── get_b_g_bo.py
│   ├── get_b_o_bo.py
│   ├── get_capillary_pressure_bo.py
│   ├── get_cell_status_vo.py
│   ├── get_flux_and_props_gas_bo.py
│   ├── get_flux_and_props_oil_bo.py
│   ├── get_flux_and_props_water_bo.py
│   ├── get_polymer_shear_multiplier.py
│   ├── get_well_polymer.py
│   └── update_state_black_oil_generic.py
└── examples/
    ├── egg/
    ├── spe1/
    ├── spe3/
    ├── spe9/
    ├── spe10/
    └── utils/
```

Python-only backends and compatibility aliases may remain, but they must be
identified as extensions rather than substitutes for missing MRST entry
points. Generated `__pycache__` files are not part of parity.

## 5. Missing utility implementations

### 5.1 `calculateHydrocarbonsFromStatusBO`

Add `utils/calculate_hydrocarbons_from_status_bo.py`.

It must compute `sG`, `rs`, `rv`, `rsSat`, and `rvSat` from pressure,
saturations, hydrocarbon status, and the `disgas`/`vapoil` flags. It must:

- preserve NumPy and ADI inputs;
- use saturated PVT values only in the appropriate status regions;
- reproduce water-only-cell handling;
- support dead oil, live oil, wet gas, and combined live-oil/wet-gas cases;
- avoid mutating caller-owned arrays unless MRST explicitly mutates state;
- become the shared owner used by both model assembly and public utilities.

### 5.2 `computeInitAquifer`

Add `utils/compute_init_aquifer.py`.

The function must initialize analytic aquifer pressure/volume variables using
the model's aquifer definitions and initial volumes. Existing aquifer logic in
AD-core should be reused, while preserving MRST-style array ordering and state
field names.

### 5.3 Phase flux and property helpers

Add:

- `get_flux_and_props_water_bo.py`
- `get_flux_and_props_oil_bo.py`
- `get_flux_and_props_gas_bo.py`

Each function must expose the corresponding MRST return tuple: phase flux,
shrinkage factor, mobility, density, phase pressure, upstream flag,
phase-potential difference, and viscosity. The implementation should compose
the existing AD-core state functions, not duplicate discretization logic.

Required cases include:

- gravity on and off;
- capillary pressure on and off;
- numeric and ADI values;
- internal faces and NNC connections;
- saturated and undersaturated hydrocarbon PVT;
- `rs` and `rv` transport contributions;
- multiple PVT and saturation regions.

### 5.4 `getbO_BO` and `getbG_BO`

Add `get_b_o_bo.py` and `get_b_g_bo.py` as public PVT dispatch helpers. They
must select saturated or undersaturated table evaluation by cell and preserve
derivatives for ADI input. Region selection must agree with MRST.

### 5.5 `updateStateBlackOilGeneric`

Add `update_state_black_oil_generic.py` and make it the common Newton-update
owner. It must implement:

- relative and absolute pressure-step limits;
- saturation-step limits and saturation normalization;
- status-dependent hydrocarbon primary variables;
- phase appearance/disappearance;
- `rs` and `rv` bounds;
- well and facility variable updates;
- minimum-pressure policy;
- model state-function cache invalidation;
- an MRST-shaped update report.

Existing update logic should be extracted carefully so current simulator
behavior remains unchanged before new cases are enabled.

### 5.6 Thermal equations

Add `equations_water_thermal.py`. It should adapt the shared geothermal
mass-energy implementation to MRST's `equationsWaterThermal` interface,
including `drivingForces`, pressure/temperature boundary conditions, sources,
wells, and returned equation names/types.

### 5.7 Polymer helpers

Add `get_well_polymer.py` and `get_polymer_shear_multiplier.py`.

The shear multiplier requires the same nonlinear local solve, convergence
report, velocity handling, and injection/production treatment as MRST. These
utilities should be usable by polymer models outside `ad_blackoil` without a
circular dependency.

## 6. Equation-entry-point parity

The existing equation modules are useful adapters but not complete MRST
interfaces. Refactor them to accept equivalent driving forces and options:

```python
problem, state = equations_black_oil(
    state0, state, model, dt, driving_forces, **options
)
```

Equivalent contracts are required for oil-water, water, and thermal water.
The returned problem should carry:

- ordered equations;
- equation names and types;
- primary-variable names;
- state and timestep metadata;
- facility equations;
- convergence tolerances or scaling metadata used by the nonlinear solver.

Compatibility with the current compact calling convention may be retained,
but the MRST-shaped convention must be tested directly.

## 7. Model parity work

### 7.1 Phase configurations

The model matrix must support and validate:

| Water | Oil | Gas | `DISGAS` | `VAPOIL` | Required |
|---:|---:|---:|---:|---:|---|
| yes | no | no | no | no | Single-phase water |
| yes | yes | no | no | no | Oil-water |
| yes | yes | yes | no | no | Dead three-phase |
| yes | yes | yes | yes | no | Live oil |
| yes | yes | yes | no | yes | Vaporized oil without dissolved gas |
| yes | yes | yes | yes | yes | Combined live oil and wet gas |

The current explicit rejection of `vapoil=True, disgas=False` must be removed
only after its primary-variable, flash, accumulation, flux, facility, and
state-update paths are implemented and compared with MRST.

### 7.2 Regions and fluid properties

Complete and test per-cell dispatch for `PVTNUM`, `SATNUM`, `ROCKNUM`, and any
region arrays consumed by the module. All table evaluations must work for
numeric and ADI inputs and at table endpoints.

### 7.3 Wells and schedules

Validate:

- BHP and total/phase-rate controls;
- `orat`, `wrat`, `grat`, `lrat`, and `vrat` limits;
- automatic control switching;
- well shut-in and reopening;
- changing well status across schedule steps;
- injector composition, `rs`, `rv`, and polymer concentration;
- sign changes according to MRST policy;
- multisegment well variables and topology;
- well-solution field names and ordering.

### 7.4 Hysteresis

Connect the existing hysteresis helpers to deck keywords and the complete
relative-permeability/capillary state-function path. Validate drainage,
imbibition, reversal points, scanning curves, and state-history updates.

### 7.5 Thermal model

Confirm that `WaterThermalModel` matches MRST's pressure-temperature unknowns,
rock/fluid energy, conduction, advective energy, well energy, boundary energy,
and update behavior. Delegation to the geothermal kernel is acceptable when
the public behavior and numerical results agree.

## 8. Deck, schedule, and result infrastructure

The guarded field examples cannot be completed case by case. Implement or
finish reusable infrastructure for:

- general ECLIPSE deck parsing;
- INCLUDE resolution and unit conversion;
- grid and ACTNUM processing;
- PVT, saturation, rock, aquifer, and thermal keywords;
- schedule controls and well completion changes;
- multisegment well tables;
- VFP production/injection tables;
- restart and summary-file reading where examples compare external results;
- deterministic dataset discovery and clear download instructions.

The importer must preserve original deck ordering and region numbering while
converting MATLAB one-based indices to Python zero-based indices exactly once
at the boundary.

## 9. Example completion matrix

| Example | Current state | Completion requirement |
|---|---|---|
| `blackoilTutorialOnePhase` | Runnable | MATLAB state/history comparison |
| `blackoilTutorialGravSeg` | Runnable | Full state and conservation comparison |
| `immiscibleTimeIntegrationExample` | Runnable | Match integration variants and reports |
| `blackoilTutorialSPE1` / `simulateSPE1` | Runnable | Full schedule and summary regression |
| `aquifertest` | Runnable slice | Match all aquifer cases and external result path |
| `simulatorWorkflowExample` | Runnable | Verify hooks, reports, and restart behavior |
| `blackoilTutorialPlotHook` | Runnable | Verify callback timing and state payload |
| `blackoilTutorialMexAcceleration` | Python timing analogue | Document backend equivalence and compare residuals |
| `multisegmentWellExample` | Reduced runnable case | Complete SPE1 multisegment topology and controls |
| `blackoilTimeIntegrationExample` | Fully implicit SPE1 only | Add explicit and adaptive-implicit paths and SPE9 |
| `eggExample` | Guarded | Dataset loader plus complete simulation |
| `simulateSPE3` | Guarded | Deck/schedule import plus reference comparison |
| `simulateSPE9` / tutorial | Guarded | Deck, VFP, schedule, and summary comparison |
| `tenthCSP_Model_I` | Guarded | SPE10 properties and complete setup |
| `blackoilSectorModelExample` | Guarded | Sector deck and boundary-pressure workflow |
| `fieldModelNorneExample` | Guarded | Full Norne deck/schedule and field run |

Every example must provide a noninteractive CLI path suitable for CI. Plotting
must be optional and must not change numerical execution.

## 10. Parity ledger and audit tooling

Create:

```text
scripts/audit_ad_blackoil_parity.py
docs/ad_blackoil/parity.csv
AD_BLACKOIL_PARITY.md
```

The CSV should contain at least:

```text
mrst_path,symbol,section,status,python_path,mapping,validation,evidence,notes
```

Allowed statuses should distinguish:

- `ported`
- `ported-consolidated`
- `excluded-catalog`
- `excluded-documentation`
- `excluded-reference-data`
- `pending`

The audit must discover MRST files from the local 2024a tree, discover Python
owners and symbol references, identify placeholder examples, attach tests and
validators, and fail when an executable MATLAB file has no disposition. A
zero-pending ledger is necessary but does not replace runtime validation.

## 11. MATLAB reference-validation design

For each implementation batch, add a paired fixture:

```text
examples/validation/dump_ad_blackoil_<feature>_reference.m
examples/validation/ref_ad_blackoil_<feature>.mat
examples/validation/validate_ad_blackoil_<feature>.py
```

MATLAB dumps must use the local MRST 2024a tree and save primitive arrays and
metadata with `-v7`. Python validators must recreate the case independently;
they must not simply replay saved Python inputs and outputs.

### Required comparison levels

1. **Utility level:** every returned array, flag, and report field.
2. **Property level:** pressure, saturation, PVT, mobility, density, upstream
   flags, potential differences, and fluxes.
3. **Equation level:** residual values and sparse Jacobian blocks.
4. **Newton-update level:** updated state, status changes, step limiting, and
   well variables.
5. **Timestep level:** converged state, iteration count, convergence flags,
   and well solution.
6. **Schedule level:** state and well histories at every report step.
7. **Conservation level:** component inventory changes against wells, sources,
   boundaries, and aquifers.

### Numerical tolerances

- Topology, status, region, and control arrays: exact equality.
- Small utility and residual fixtures: `rtol <= 1e-11`, `atol <= 1e-12` when
  conditioning permits.
- Sparse Jacobians: compare shapes, sparsity patterns, and values separately.
- Converged field cases: use scale-aware pressure, saturation, rate, and
  component-balance tolerances documented per validator.
- Different nonlinear iteration counts are acceptable only when both runs
  converge to the same solution and the reason is documented.

## 12. Python test plan

Add a dedicated tree:

```text
tests/ad_blackoil/
├── components/
├── models/
├── utils/
├── examples/
├── integration/
└── test_parity_ledger.py
```

Tests must include:

- unit tests for every public utility;
- finite-difference checks of AD Jacobians;
- parameterized phase/status combinations;
- phase appearance/disappearance at saturation and PVT limits;
- multi-region property dispatch;
- wells, BCs, sources, gravity, NNCs, and aquifers;
- thermal and polymer paths;
- timestep cutback and control switching;
- mass conservation after each converged step;
- smoke tests for every example;
- MATLAB-reference validators callable from CI when fixtures exist.

Avoid relying only on `if __name__ == "__main__"` self-tests. Important model
behavior must be collected by pytest.

## 13. Implementation phases

### Phase A — Establish measurable parity

1. Add the parity auditor, CSV, and generated Markdown report.
2. Classify all MRST source, catalog, data, and example files.
3. Add import and public-export tests for the final package layout.

Acceptance: the audit accounts for every MRST file and honestly retains
`pending` status for incomplete implementations.

### Phase B — Extract the black-oil numerical kernel

1. Port `getbO_BO` and `getbG_BO`.
2. Port the three phase flux/property helpers.
3. Port `calculateHydrocarbonsFromStatusBO`.
4. Refactor model code to call these owners.
5. Compare utility values and AD Jacobians with MATLAB.

Acceptance: identical property and flux results for all phase/status cases.

### Phase C — Complete state transitions

1. Port `updateStateBlackOilGeneric`.
2. Align `computeFlashBlackOil` completely.
3. Enable `VAPOIL` without `DISGAS`.
4. Validate phase transitions, limiting, and minimum pressure.

Acceptance: full `DISGAS`/`VAPOIL` matrix matches MATLAB across Newton updates
and complete timesteps.

### Phase D — Complete force and equation APIs

1. Expand water, oil-water, and black-oil equation entry points.
2. Port `equationsWaterThermal`.
3. Port `computeInitAquifer`.
4. Validate wells, BCs, sources, gravity, aquifers, and thermal forces.

Acceptance: MRST-shaped calls return equivalent problem metadata, residuals,
Jacobians, and converged states.

### Phase E — Polymer, wells, and advanced schedules

1. Port polymer well and shear helpers.
2. Complete well status/sign behavior and multisegment support.
3. Complete schedule control conversion and VFP integration.

Acceptance: polymer and advanced-well fixtures match MATLAB and run through
control changes without case-specific code.

### Phase F — Decks and benchmark examples

1. Generalize deck/schedule/result import.
2. Replace guarded EGG, SPE3, SPE9, SPE10, sector, and Norne examples.
3. Add explicit and adaptive-implicit time-integration variants.
4. Store compact deterministic references rather than large duplicate data
   files where licensing or repository size is a concern.

Acceptance: all shipped examples run noninteractively, and selected benchmark
histories match MRST or supplied commercial-simulator references.

### Phase G — Release gate

1. Run the complete PyMRST suite.
2. Run every ad-blackoil MATLAB validator.
3. Run long benchmark and conservation tests.
4. Regenerate the parity ledger with zero pending executable entries.
5. Document supported deck keywords, known numerical differences, and example
   commands.

Acceptance: clean checkout, reproducible environment, zero unexplained parity
gaps, and no guarded example advertised as implemented.

## 14. Regression commands

The final implementation should provide stable commands equivalent to:

```bash
python3 scripts/audit_ad_blackoil_parity.py
python3 -m pytest -q tests/ad_blackoil
python3 -m pytest -q tests/ad_core
python3 -m pytest -q tests/test_all.py -k "ad_blackoil or blackoil"
```

Each MATLAB reference dump should be reproducible with:

```bash
/Applications/MATLAB_R2021b.app/bin/matlab -batch \
  "run('../mrst-2024a-Geomech/startup.m'); run('<dump-script>');"
```

Then run the corresponding Python validator using `python3`, not `python`, on
systems where only the Python 3 executable is installed.

## 15. Engineering rules

- Preserve zero-based indexing inside Python; convert MATLAB indices only at
  import/export boundaries.
- Preserve Fortran ordering where deck or MATLAB array layout requires it.
- Keep value and Jacobian paths together; never validate only numeric values
  for a function intended to support ADI.
- Reuse AD-core components and state functions where they implement the same
  mathematics, but expose the MRST-shaped ad-blackoil entry point.
- Avoid silent fallback from unsupported physics to a simpler model.
- Raise targeted errors naming the missing deck keyword or feature.
- Keep examples deterministic and noninteractive by default.
- Do not commit user-generated simulation output as reference data unless it
  is intentionally reviewed and documented.
- Preserve backward-compatible aliases until downstream PyMRST modules have
  migrated to the final package paths.

## 16. Immediate next work

The recommended next implementation batch is Phase A followed by the smallest
independent part of Phase B:

1. create the parity auditor and baseline ledger;
2. implement `get_b_o_bo.py` and `get_b_g_bo.py`;
3. generate saturated/undersaturated, region-aware MATLAB references;
4. add numeric and AD Jacobian tests;
5. refactor the generic model to use the validated helpers without changing
   existing SPE1 results.

This sequence establishes measurable progress and supplies the low-level PVT
dispatch required by the later flux, hydrocarbon-status, and state-update
ports.
