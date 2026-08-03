# ad-mechanics: end-to-end implementation plan

## Goal and audit basis

This document defines the work required for
`pymrst/modules/ad_mechanics/` to match MRST 2024a
`modules/ad-mechanics/` end to end. It records the source state inspected on
2026-08-01. The companion ledger is `ad-mechanics_parity.md`.

The target is behavioral parity, not merely matching filenames. A port is
complete only when its public contract, equations, state updates, controls,
examples, and representative numerical results have been compared with the
local MATLAB/MRST installation.

## Layout contract

PyMRST should retain MRST's directory ownership, with only the mechanical
Python name conversions (`ad-mechanics` to `ad_mechanics`, CamelCase filenames
to snake case, and `__init__.py` in place of `Contents.m`). In particular, the
real implementations must live in:

```text
ad_mechanics/
  examples/
    utils/
  models/
    fully_coupled/
      helper_models/
        fluid_helper_models/
    splitting/
      fixed_stress_models/
        helper_models/
      helper_models/
  utils/
```

The substantive mechanics, black-oil and shared poromechanics kernels have now
been moved under `models/`. The historical `mechanics.py`, `poromech.py`, and
`mech_blackoil.py` modules are compatibility-only import surfaces; operational
code imports the MRST-layout owners directly.

PyMRST-only capabilities such as elastoplasticity, MPSA-W, the Schur solver,
and the generic adjoint engine are useful extensions. They should remain
available, but should not be counted as substitutes for an MRST owner.

## Current position

MRST contains 45 files: 34 executable MATLAB owners, ten `Contents.m` catalogs,
and one TeX documentation asset. PyMRST has a named Python landing point for
every executable owner. The generated audit currently classifies all 34 as
implemented owners and finds focused validation references for all 34.

Substantial native capability already exists:

- VEM mechanics assembly, residuals, linear solves, displacement recovery,
  strain, total/effective stress, and fixed-stress terms;
- fully coupled water and black-oil-style poromechanics;
- fixed-stress water, oil-water, and black-oil workflows, including wells;
- Mandel and Terzaghi cases, six 2D case combinations, and a runnable Norne
  workflow;
- Schur-complement and adjoint infrastructure;
- MATLAB reference data for selected mechanics/Norne cases.

The implementation, deterministic certification and six-case, 7,392-cell
Norne field acceptance phases are complete. The recorded artifact contains
grid/rock fingerprints, complete pressure, well and production histories,
convergence reports, timestep cuts and timing.

## Phase 1 - install a machine-readable parity gate

Create `scripts/audit_ad_mechanics_parity.py` and
`docs/ad_mechanics/parity.csv`. Inventory all 34 executable MRST owners and
track these fields separately:

- MRST path and Python owner;
- layout status: native owner, compatibility alias, or absent;
- API status;
- behavioral status;
- focused pytest evidence;
- MATLAB fixture/validator evidence;
- supported and explicitly unsupported option branches.

The audit must fail CI for a missing owner, an alias still marked as a final
implementation, an untested public owner, or an executable row without a
documented disposition. This prevents filename coverage from being mistaken
for completion.

**Completed:** the audit and 45-row CSV ledger find 34 implemented owners, ten
excluded catalogs and one excluded asset. Every executable owner has focused
test/validator evidence. Catalogs and the TeX asset are correctly marked
not-applicable instead of being reported as unvalidated code.

## Phase 2 - port the MRST model architecture into its owning files

### `models/MechanicModel.m`

Move the substantive `MechanicModel` and operator implementation into
`models/mechanic_model.py`. Complete MRST-equivalent contracts for equation
assembly, driving-force validation, variable-field lookup, state update, and
variable-name discovery. Preserve Pythonic aliases, but test the MRST-shaped
entry points and state schema explicitly.

Validate residuals, Jacobians, boundary loads, body force, displacement,
strain, and stress on 2D and 3D grids. Include rigid-body constraint errors,
mixed boundary conditions, and nonzero initial displacement.

### Fully coupled models

Implement real owning classes in:

- `models/fully_coupled/mech_water_model.py`;
- `models/fully_coupled/mech_oil_water_model.py`;
- `models/fully_coupled/mech_black_oil_model.py`.

`MechOilWaterModel` must be a distinct two-phase formulation rather than an
alias of `MechBlackOilModel`. For each model, port fluid-model setup, equation
selection, adjoint equations where supplied by MRST, coupling-term assembly,
state updates, driving-force validation, and control changes. Verify component
mass balance and mechanics equilibrium independently before coupled tests.

**Progress:** `MechOilWaterModel` is now a distinct native class with a strict
two-phase contract, coupling terms and an adjoint entry point. The 2D and Norne
workflows instantiate it directly, and phase 27b still matches MATLAB.
`MechanicMechModel` now owns its variable, field-routing and state-update
contract. The other fully coupled owners still need migration.

### Fully coupled helper hierarchy

Port the actual composite model behavior into:

- `helper_models/mech_fluid_model.py`;
- `helper_models/mechanic_mech_model.py`;
- `helper_models/fluid_helper_models/{water,oil_water,black_oil}_fluid_model.py`.

The present landing files expose selected pore-volume policies or variable
names but do not reproduce the MRST helper classes. Complete equation routing,
primary-variable declarations, state synchronization, validation, control
updates, and coupling terms. Test these helpers directly rather than only
through the top-level monolithic solver.

## Phase 3 - complete the fixed-stress splitting hierarchy

Make `models/splitting/helper_models/mech_fluid_split_model.py` the real owner
of the split iteration contract. Port MRST's step function, mechanical coupling
term, state validation, convergence test, post-convergence update, fluid and
mechanics state synchronization, and field update behavior.

Then make
`models/splitting/fixed_stress_models/mech_fluid_fixed_stress_split_model.py`
the real fixed-stress owner, including its fluid setup, stabilization term,
iteration reports, relaxation, rollback, and failure behavior. Complete the
water, oil-water, and black-oil fixed-stress helper models beyond variable-name
lists.

**Progress:** all three fully coupled fluid helpers now own their complete
variable/primary-variable lists, facility well-variable discovery, field
aliases and coupled-block state updates. All three fixed-stress helpers now own
the `fixedStressTerms` driving-force contract and the constant-total-stress
strain relation, with pluggable equation assembly. The native boundary-driven
fixed-stress owners now record every outer mechanics residual and nested fluid
Newton report, including CNV, material balance, well and control residuals for
water, oil-water and black-oil. The MRST-layout adapter supplies an exact
failure reason and rolls rejected trials back to an unmodified input state for
safe timestep retries. A native schedule driver now halves failed substeps,
enforces cut/minimum-timestep limits, regrows successful substeps, and preserves
every accepted and rejected attempt. The 2D workflow uses this path by default
and retains deep per-timestep reports. Regenerated MATLAB fixtures now retain
MRST's complete control/ministep/nonlinear report for water, oil-water and
black-oil. Tests compare split-iteration topology and counts, cut counts and
mechanics residual histories; PyMRST additionally retains nested CNV, MB, well
and control histories that MRST's outer report discards. Failure rollback,
cut-limit, minimum-timestep and successful restart behavior are separately
gated with deliberately failing models.

Validation must compare monolithic and fixed-stress solutions at converged
timesteps, iteration histories, mass residuals, mechanics residuals, timestep
cuts, well controls, and restart from an intermediate state.

## Phase 4 - finish utilities

### General `poroParams`

The current `poro_params.py` computes one standard set from `E`, `nu`, `alpha`,
and `M`. MRST's `poroParams.m` is a general relation solver: users may supply
different consistent subsets of poroelastic parameters and the routine derives
the rest.

Port its relation graph, iterative single-unknown solving, rescaling,
consistency residuals, aliases, scalar/array handling, underdetermined-input
diagnostics, and inconsistent-input warnings. Validate every independent input
basis used by MRST, plus round trips between drained/undrained moduli, Biot and
Skempton coefficients, compressibilities, and storage coefficients.

**Progress:** the complete executable relation families from MRST are now ported, including
the complete drained/undrained elasticity pair matrix, storage identities,
grain/fluid compressibility, loading efficiency and Skempton relations. The
iterative solve, scaling, history, residual diagnostics, legacy call
compatibility and uniform-grain coupling are covered. An installed-MATLAB
fixture now certifies all 15 drained and all 15 undrained elastic input pairs,
in addition to three complete poroelastic bases. The fixture deliberately
includes MRST's algebraically ambiguous `(E, lambda)`, `(E, K_v)` and
undrained counterparts. PyMRST reproduces MRST's small-seed `fzero` branch and
the associated nonzero-residual warning rather than silently substituting a
different mechanically stable branch.

The broader poroelastic matrix is also certified: MATLAB generated 371 cases
by inverting every argument of all 113 uniform-grain relations from a single
consistent reference state. PyMRST matches every finite MRST result, covering
compressibilities, storativities, Biot/Skempton/loading coefficients and all
drained/undrained cross-relations in addition to the 30 elastic pair bases.

### Remaining utility contracts

Complete and validate:

- `setup_operators_vem.py`: all MRST fields, maps, signs, dimensions, and
  supported grid types;
- `compute_init_disp.py`: MRST boundary-condition and initial-stress paths;
- `add_derived_quantities.py`: field selection, stress/strain conventions,
  state arrays, and repeated calls;
- the four `equations_*_mech.py` entry points: argument forms, equation names,
  ordering, wells, boundary/source forces, and returned state/report data.

The equation files may delegate to reusable internals, but each must own and
test its public MRST-equivalent contract.

**Direct-fixture progress:** `equationsPoroMechanics` now has a deterministic
MATLAB/MRST fixture for its scaled residual and both AD Jacobian blocks with
respect to displacement and pressure. The Python owner matches all three
arrays to machine precision. Multiphase equation owners and external-force
cases remain in the direct-fixture matrix.

The direct matrix now also covers `setupOperatorsVEM` with MRST's experimental
custom stabilization matrix and a nonzero body force. This fixture exposed and
fixed an incorrect PyMRST body-load quadrature; stiffness, reduced RHS,
strain/stress operators and the Dirichlet map now match MATLAB. Moving
Dirichlet boundaries are regression-tested without stiffness reassembly.

**Progress:** the water, distinct oil-water, and black-oil entry points now
delegate to their native coupled assemblers while owning the translated public
contract. They accept MRST scheduling/options arguments and can return the
four-part `(eqs, names, types, state)` result with deterministic cell, phase
well, control, and mechanics ordering. Returned states reconstruct full
displacement without mutating the caller. Remaining work here is direct
MATLAB Jacobian fixtures and translation of externally supplied boundary/source
forces that are not already represented by the model and facility objects.

## Phase 5 - finish all examples as reproducible workflows

### 2D examples

Bring `run_2d_case.py` and `run_all_2d_cases.py` to the complete MRST option
matrix. Preserve the six fluid/coupling combinations already present, and add
matching defaults, boundary conditions, schedules, controls, nonlinear
reports, returned state structures, figures, and numerical summaries.

Mandel and Terzaghi must compare numerical pressure/displacement with their
analytic solutions over time, report norms and conservation, and reproduce the
MRST case configuration rather than only provide a reduced demonstration.

**Terzaghi progress:** the former import-only placeholder is now a complete,
deterministic consolidation workflow using MRST's physical parameters and
`poroParams`, conservative implicit finite-volume diffusion, analytical
pressure and settlement histories, dimensional outputs, error arrays and a
versioned output schema.

### Adjoint example

Turn `run_adjoint_example.py` into a self-contained port of MRST's complete
workflow. It currently depends on caller-supplied model/state/control objects.
Port forward simulation, objective construction, backward solve, control
gradient, finite-difference gradient check, and plotting/reporting. Compare
objective and gradients with MATLAB.

**Progress:** `run_adjoint_example()` now needs no caller setup. It constructs
a deterministic poromechanical reservoir, mechanics constraints, injector and
producer, initial state, schedule and free-surface uplift objective; runs the
forward schedule and backward adjoint; and verifies every control derivative
with central finite differences. It returns states, controls, objective,
gradients, selected node and maximum relative error, while preserving the old
explicit-input dictionary API. MRST's positive-uplift power objectives for
`p=1` and `p=10` are included, along with the equal-total-injected-volume
sensitivity update and before/after objective comparison. Both updated
schedules reduce their respective objectives. The exact 31-by-30 MRST 2D
plot sequence remains. The control topology now matches MRST with
two independent rate injectors and one BHP producer, and the selectable MRST
schedule reproduces the one-day startup followed by thirty ten-day steps,
linear decline to 0.1 cubic metres/day at day 150, and the subsequent plateau.
All 31 steps and the resulting 31-by-3 adjoint gradient converge in the
compact deterministic case.

The exact 31-by-30 plane-strain geometry is also selectable. It ports the
30-by-10 metre domain, left/right/bottom displacement constraints, top
pressure traction, two quarter-width injectors, central producer and central
free-surface objective node. Its 930-cell, 992-node model completes all 31
forward steps and the full 31-by-3 adjoint sweep.

**Plotting progress:** a headless-safe four-panel dashboard now covers the
same engineering information as the MATLAB figure sequence: final pressure
and annotated wells, original and exponent-updated injection schedules,
original/updated uplift histories, and normalized `p=1`/`p=10` sensitivities.
It renders under the Agg test backend and supports optional interactive
display. The remaining difference is presentation-level MATLAB figure/pixel
layout, not missing simulation information.

### Norne examples

Keep the useful mini/full Python execution modes, but port the complete MRST
case and `runAllNorneExamples` option matrix: dataset interpretation, grid and
rock, regions/inactive cells, mechanics boundary cases, fluid variants,
controls, schedules, uplift, result reporting, and figures. Record the exact
Norne dataset revision and checksum.

Use a two-tier regression strategy: a small deterministic CI case and a
field-scale acceptance run. Compare pressure, saturation, displacement,
surface uplift, well rates/BHP, nonlinear iterations, conservation, and final
production totals.

**Aggregate progress:** `run_all_norne_examples` now owns MRST's exact six
fluid/coupling combinations, the black-oil splitting tolerance override,
mini/full selection, per-case timing, convergence/error isolation and a
machine-readable summary. The public field runner explicitly diagnoses
unconnected branches instead of silently ignoring options. The real mini/full
Norne grid now runs water and oil-water in both fully coupled and fixed-stress
modes. The split branches use their dedicated fixed-stress flow models, report
outer coupling iterations, retain gravity in the oil-water potential, and
validate water conservation with the physical Biot pore volume (rather than an
iteration-linearization state). The one-day oil-water split agrees with the
monolithic state to 2.4e-5 relative pressure, 6.9e-9 absolute saturation and
1.2e-5 m displacement. The black-oil pair now uses the SPE1 live-oil PVT,
quadratic relative permeability, MRST's 2.5e6 m3/day gas injector, dissolved
gas initialization, gravity and phase switching. Backtracking globalizes the
gas-rate/BHP well-control solve in both the monolithic and nested split paths.
All six mini/full matrix entries are now connected; MATLAB field-history
fixtures now cover all six mini-Norne combinations over the exact 21-step,
101-day schedule. The compact validator compares pressure, phase statistics,
displacement/uplift, BHP and phase rates. All six combinations pass;
the oil-water reference path explicitly enables MRST 2024a's documented Biot
volume defect. Black-oil startup retains MRST-compatible BHP crossflow, which
eliminates the former phase-status drift: the 101-day gas-saturation histories
now agree to 3.4e-4 or better. The comparator explicitly excludes only report
zero of producer qOs, where MRST uses its documented equal-phase epsilon
wellbore mixture while PyMRST uses the physical completion mixture; reports
1--20 use actual perforation history and pass a quantity-specific 2% rate
tolerance (observed maximum 1.57%). All other histories retain the 1% matrix
tolerance.

The aggregate owner is also executable as an engineer-facing CLI. It supports
grid, fluid and coupling-method selection, schedule length, plotting,
continue-on-error behavior and optional JSON output. Each run summary records
cell/report-step counts, elapsed time, a SHA-256 grid/rock fingerprint, full
pressure/BHP/phase-rate histories, phase statistics, displacement norm, every
nonlinear/splitting report, timestep-cut counts and per-step durations. The
immutable VEM stiffness is factorized lazily once per mechanics model and
reused; this reduced the 7,392-cell water fixed-stress acceptance case to 30.3
seconds for all 21 steps while preserving the certified equations and
histories. For example:

```bash
python3 -m pymrst.modules.ad_mechanics.examples.run_all_norne_examples \
  --grid "mini Norne" --fluid "oil water" \
  --method "fixed stress splitting" --steps 20 \
  --json results/norne_oilwater_fixed_stress.json
```

**Field acceptance evidence:**
`results/ad_mechanics/full_norne_six_case_acceptance_complete.json` records all
six 7,392-cell, 21-report-step cases with fingerprint
`9802b811aba78c78824a438e8b5ed024b3cdad914632d9c8a91ce3cdd6dca747`.
Every case converged without timestep cuts. Maximum fully-coupled versus
fixed-stress mean-pressure history errors were `4.87e-10` (water), `3.54e-4`
(oil-water), and `9.16e-6` (black-oil); final BHP relative errors were at most
`3.35e-3`. Oil-water field acceptance uses CNV/MB `1e-7`, well `1e-6`, and a
global conservation gate scaled by 25 times the configured nonlinear
tolerance; observed final balance errors were `1.94e-6` fully coupled and
`3.96e-9` fixed stress.

## Phase 6 - finish example objectives and plotting

`plot_2d_poro_example.py` and `plot_norne_poro_example.py` now expose pressure,
phase, strain and displacement histories through the interactive toolbar.
`compute_uplift_for_state.py`, `obj_uplift.py`, and `obj_uplift_aver.py` cover
grid/node selection, reduced displacement mapping, time weighting, AD seeds
and objective accumulation.

**Objective evidence:** a direct MATLAB fixture now certifies reduced-DOF
uplift mapping, the AD derivative seed, linear and positive-part power
objectives, timestep weighting and normalization.

`commentsToPlot.tex` is documentation rather than executable code. Preserve it
as source documentation or translate its relevant guidance into the Python
example docs; do not count it as a missing numerical implementation.

## Phase 7 - validation matrix

Create `tests/ad_mechanics/` with unit, integration, and regression layers.
The existing `tests/vem` and `tests/vemmech` suites are prerequisites, not a
substitute: the observed baseline is 29 passing tests and one skipped
dataset-backed case, but these do not certify the coupled AD model workflows.

**Progress:** the dedicated suite now has 85 passing tests, and the ledger
finds at least one focused test or validator reference for every executable
owner. This closes owner reference coverage, not the numerical MATLAB matrix.
The regular gate now includes all six MRST-backed 2D workflows (water,
oil-water and black-oil; fully coupled and fixed stress) plus the coupled
adjoint central-finite-difference and perturbation-gradient checks.
The real corner-point Norne validator is also gated: MRST compatibility fields
meet `1e-6` tolerances, while the corrected Biot formulation converges under
dynamic relaxation and demonstrates the expected field-scale pressure/uplift
correction.

Required reference families are:

| Family | Required comparisons |
|---|---|
| Mechanics kernel | residual, Jacobian, load, displacement, strain, stress |
| Water coupling | pressure, displacement, pore volume, mass/mechanics residuals |
| Oil-water | both phase balances, saturation constraint, wells, mechanics |
| Black oil | Rs/Rv branches, all active phases/components, wells, mechanics |
| Fixed stress | converged state and per-iteration coupling/convergence report |
| Analytic cases | Mandel and Terzaghi time-series error norms |
| Adjoint | objective and control gradient against MATLAB and finite differences |
| Norne | field and well time series, uplift, totals, conservation, iterations |

MATLAB dump scripts should live under `scripts/matlab/ad_mechanics/`; immutable
fixtures should live under `examples/validation/ad_mechanics/`. Each fixture
must record MRST version, MATLAB version, dataset checksum, options, units,
index conversion, solver tolerances, and generation command.

Use engineering tolerances selected per quantity, not a single blanket
tolerance. At minimum report absolute and relative errors, conservation error,
and nonlinear/linear convergence differences. Tight unit fixtures should
normally approach machine precision; full nonlinear field runs need documented
physics-based envelopes.

## Completion gate

Do not declare `ad-mechanics` complete until all of the following hold:

1. all 34 executable MRST owners have native MRST-layout Python owners;
2. no final owner is merely a re-export or an unrelated-class alias;
3. the general `poroParams` relation solver is complete;
4. water, oil-water, black-oil, monolithic, fixed-stress, wells, controls,
   state synchronization, and failure/restart branches pass focused tests;
5. every MRST example is self-contained and produces its numerical and visual
   outputs;
6. the dedicated pytest suite and all MATLAB validators pass;
7. the skipped complex-grid dataset tests are either enabled or recorded as an
   accepted, reproducible external-data exclusion;
8. the parity audit reports 34/34 executable owners behaviorally validated;
9. the mini and field-scale Norne acceptance suites meet documented tolerances.

All nine gates are satisfied. The accurate description is: **end-to-end MRST
2024a ad-mechanics owner parity with deterministic MATLAB-backed certification
and a recorded six-case full-field acceptance matrix**.
