# Compositional module: end-to-end completion plan

## Objective

Complete `pymrst/modules/compositional/` as the Python counterpart of MRST
2024a's `modules/compositional/`, preserving the MRST directory layout and
public behavior while using Python/NumPy/SciPy and PyMRST's AD infrastructure.

The comparison source is:

```text
mrst-2024a-Geomech/modules/compositional/
PyMRST/pymrst/modules/compositional/
```

MRST contains 96 MATLAB files: 82 executable files and 14 `Contents.m`
catalogs. PyMRST currently has a named Python owner for every executable file
and Python catalog modules for the MRST package indexes. This is excellent
structural coverage, but filename coverage is not the completion criterion.

## Current state

The numerical core is already substantial. The following representative
MATLAB comparisons pass:

- Peng-Robinson flash, phase fractions, compositions, fugacity, and Z factors;
- fully implicit overall-composition flow with component mass conservation;
- natural-variable flow, including water and compositional wells;
- compositional pressure and flux boundary conditions with schedule changes;
- SPE1 black-oil-to-compositional conversion and a two-step native solve;
- EOS separator and multistage separator-tree calculations;
- sequential pressure/transport, including water and wells;
- facility and `SimpleWell` helper behavior.

The audit performed while writing this plan ran 17 critical validators. All 17
passed. This does not replace running the complete compositional regression
suite.

## What remains

### 1. Establish an executable parity ledger

Create `scripts/audit_compositional_parity.py` and
`docs/compositional/parity.csv`. Each of the 96 MRST files must have:

- a Python owner or an explicit catalog/exclusion classification;
- a status of `complete`, `broad`, `partial`, or `excluded`;
- focused Python-test evidence;
- MATLAB reference evidence for material numerical behavior;
- notes for deliberately centralized implementations and compatibility files.

The audit must detect missing owners, stale paths, untested executable rows,
unclassified files, and MRST files added in future source upgrades.

Initial implementation completed on 2026-08-01:

```text
scripts/audit_compositional_parity.py
docs/compositional/parity.csv
96/96 MRST files inventoried
82/82 executable owners present
14/14 catalog owners present
82 broad, 0 partial, 0 external-data, 0 complete executable rows
```

The normal audit passes structural integrity. Its `--require-complete` option
is the deliberately failing final gate until every executable row is reviewed.

### 2. Convert embedded self-tests into a dedicated pytest suite

Only import-health and an AD-core initialization test currently exercise the
module through `tests/`. Thirty-seven compositional files contain executable
`__main__` self-tests, while most cross-language evidence lives in standalone
validation scripts. Add a mirrored test hierarchy:

```text
tests/compositional/
  fluids/
  models/eos/
  models/natvars/
  models/overall/
  properties/
  utils/
  examples/
  integration/
```

Move behavioral assertions into pytest without removing runnable examples.
Test value, shape, indexing, phase flags, AD Jacobians, conservation, invalid
input, convergence, timestep cuts, and failure recovery. The normal project
test command must discover compositional regressions automatically.

The first pytest baseline is now present under `tests/compositional/`. It pins
the ledger, owner/evidence paths, PR cubic roots, flash closure, Wilson K-value
vectorization, MRST's explicit ZJ refusal, the full stored-fluid table, deck EOS
fallback behavior, mixed-phase partial-volume dispatch, EOS-derived analytical
pressure derivatives, and water-first mass/molar partial volumes. The current
result is 26 passed; the
subsystem hierarchy and option matrix above remain to be
populated.

### 3. Finish EOS option parity

`EquationOfStateModel` supports the principal PR/SRK paths. PyMRST explicitly
refuses the Zudkevitch-Joffe EOS, which is correct MRST 2024a parity: MRST
throws `Not implemented yet` both when selecting ZJ and in its mixing-parameter
branch. Preserve this as a tested compatibility exclusion. Complete and validate:

- every `setType` alias and custom coefficient path;
- single-phase labeling and stability transitions at phase boundaries;
- volume shifts, binary interaction coefficients, and critical-property
  corrections for scalar, cell-wise, and AD inputs;
- analytic or independently verified AD derivatives of phase fractions,
  compositions, fugacity, density, and viscosity;
- difficult near-critical, near-pure, and disappearing-phase cases.

`initDeckEOSModel` warns and uses region 1 when multiple EOS regions are
supplied. This exactly matches MRST 2024a and is now a tested compatibility
exclusion (`validate_phase269.py`), not a PyMRST parity gap. Multi-region EOS
would be a future extension beyond the target version.

### 4. Finish partial-volume and pressure-equation behavior

`getPartialVolumes` now covers numerical and analytical strategies across the
supported hydrocarbon and water phase configurations, subsets, unit modes, and
accumulation terms. The two-phase EOS strategy evaluates pressure derivatives
with the same pressure perturbation used by MRST.

Mixed single-/two-phase dispatch is now implemented: each phase subset can use
its own strategy and differentiation mode while preserving requested subset
order. AD accumulation reordering, well-variable removal through the MRST
Schur complement, and iteration-sensitive numerical differentiation are also
implemented. The single-phase analytical strategy now seeds pressure and every
composition as AD variables, attaches the cubic-root derivative implicitly,
and forms MRST's density-composition partial molar volumes and pressure
derivatives. Explicit density input remains zero because it carries no pressure
law. Water is emitted as the first component and uses density-based mass or
molar partial volume consistently in both analytical and EOS branches. Phases
59, 275, and 284 pass. This owner is now classified broad rather than partial;
near-critical and large-model performance certification remains module-level
acceptance work.

Pressure-option behavior must cover total-volume, partial-volume, and CPR/IMPES
weighting branches consistently for fully implicit and sequential models.

### 5. Close model API and equation-assembly parity

Several MRST-layout files are intentionally thin compatibility dispatchers;
the real implementation is centralized in `model.py` or
`natural_variables.py`. This is acceptable only after method-level parity is
audited. Complete and test every material branch of:

- `ThreePhaseCompositionalModel`;
- `GenericOverallCompositionModel`;
- `GenericNaturalVariablesModel`;
- overall and natural-variable concrete models;
- fully implicit equation assemblers;
- phase switching, minimum saturations, increment limits, convergence values,
  state validation, post-step updates, and failure reports;
- component source terms, well contributions, boundary conditions, capillary
  pressure, gravity, regions, and non-EOS phases.

Compatibility files such as `equations_compositional.py` and the sequential
equation modules should expose the MRST-equivalent callable contract—not only
class imports or an unchecked method dispatch.

### 6. Complete sequential formulation coverage

The principal sequential workflows pass small MATLAB fixtures. Finish the full
surface for both overall and natural formulations:

- pressure and transport equation options;
- reduced linearized systems and reconstruction of updates;
- BHP correction and frozen-variable rules;
- water, wells, boundary conditions, source terms, capillary pressure,
  gravity, and region-dependent properties;
- nonlinear tolerances, timestep cuts, rollback, ministeps, and reporting;
- cross-formulation convergence and mass-balance comparisons over long runs.

The natural-variable `PressureReducedLinearSystem` now performs the MRST
weighted pressure collapse after its base Schur reduction: component rows,
partial-volume derivatives, and retained well rows contribute to the reduced
matrix and residual. The plain reduced-system fallback and phases 59/299 remain
green. `processResultAfterSolve` now reconstructs pressure, liquid/vapor
composition, phase-saturation, and water-saturation blocks in MRST primary
variable order, then re-solves retained well equations against those reservoir
updates. Full field-scale validation remains part of the sequential acceptance
matrix rather than an explicit missing implementation.

### 7. Finish fluid and property model coverage

Complete method-level parity for `CompositionalMixture`, `PropertyModel`,
`CompositionalPropertyModel`, and `BlackOilPropertyModel`, including all
vectorization, region, subset, and AD input forms.

`TableCompositionalMixture` and all 122 stored MRST CoolProp records are already
present and validated by `validate_phase268.py`. Preserve table order, values,
case-insensitive lookup, aliases, and invalid-name behavior. Treat live
CoolProp as an optional dependency:
test both installed and unavailable behavior and compare supported fluids with
MRST/CoolProp at fixed states.

### 8. Complete deck, schedule, facility, and restart integration

Validate compositional decks beyond the current SPE1 path:

- all EOS/property keywords consumed by `initDeckEOSModel`;
- multiple EOS/PVT/SAT regions and inactive-cell mappings;
- compositional well injection/production compositions and surface rates;
- control switching, limits, VFP integration, groups, guide rates, completion
  changes, and separators;
- restart-state serialization/deserialization and summary-output extraction;
- conversion between compatible black-oil and compositional states/models.

Add at least SPE1, SPE5, and a multi-region deck as maintained end-to-end
fixtures. Compare report-step pressure, phase saturation, component moles or
mass, well BHP/rates, iteration counts, and conservation.

### 9. Finish examples at MRST scale

All 13 executable MRST examples have corresponding Python files, but several
default to reduced grids for test speed. End-to-end completion requires:

- a documented quick mode and an independently validated MRST-size mode;
- full 3D six-component, larger-problem, miscibility, performance, accuracy,
  natural-vs-overall, separator, validation, and SPE1 workflows;
- reproducible plots/tables and saved result artifacts where MRST produces
  them;
- a checksum-verified loader now imports and aligns the official 499-step E300
  and AD-GPRS histories for `compositionalValidationSimple`; complete the
  full-schedule PyMRST comparison and quantify its acceptance envelopes;

The validation-simple full mode now uses the actual 1000-cell deck dimensions,
10 mD rock, Decane/CO2/Methane ordering, 100/50 bar BHP wells, and all 500
nonuniform deck timesteps. `max_report_steps` permits bounded acceptance runs,
and the comparison API reports pressure RMSE/max error in bar plus composition
and gas-saturation errors against E300 and AD-GPRS. The first 1000-cell report
step converged in 30 nonlinear iterations and approximately 15 seconds on the
development machine; the complete serial history is therefore retained as an
explicit long-running acceptance gate.

The public one-report-step acceptance run is green. Against E300/AD-GPRS it
currently gives approximately 0.39 bar pressure RMSE, 4.23--4.28 bar maximum
cell pressure error, and 1.30e-4--1.54e-4 composition MAE. These are diagnostic
baseline values, not yet the final 499-step acceptance envelope.
- noninteractive CLI entry points with useful convergence and mass-balance
  reporting.

### 10. Tighten numerical parity and performance evidence

The sampled native SPE1 validator passes a deliberately broad envelope:
approximately 4.6 bar pressure difference and several (10^{-3}) saturation
or rate differences. Diagnose pseudo-component interpolation, surface-volume
conversion, well coupling, and timestep effects, then tighten tolerances to a
documented engineering target.

The performance validator currently proves benchmark structure and available
kernel rows, not full MRST-scale performance parity. Benchmark sparse,
diagonal, row-major, deferred assembly, linear solve, and flash cost on fixed
cases. Native MATLAB MEX behavior may be an accepted platform exclusion, but
PyMRST must provide and benchmark its Python-native/Numba equivalent.

### 11. Repository and packaging cleanup

- The seven `model_runtest_*.pt` PyTorch checkpoints were verified as
  unreferenced by the MRST-compatible runtime and relocated from the package
  root to the explicitly optional `compositional/ML_models/` asset directory.
- Add package READMEs corresponding to MRST's catalogs.
- Remove duplicate spelling modules only where imports prove they are safe;
  otherwise document them as compatibility aliases.
- Define optional dependencies and avoid import-time failures when CoolProp,
  Numba, plotting, or external datasets are absent.

## Implementation order

1. Build the executable audit and pytest skeleton.
2. Pin current EOS/flash/property behavior with unit and MATLAB fixtures.
3. Complete partial-volume option parity; retain the validated MRST ZJ and EOS-region exclusions.
4. Audit fully implicit model methods and equation branches.
5. Audit sequential models and reduced systems.
6. Complete fluid tables and optional CoolProp behavior.
7. Complete deck/facility/restart/summary integration.
8. Validate every example in quick and MRST-size modes.
9. Tighten SPE1 tolerances and add long-run conservation tests.
10. Run the full acceptance gate and mark parity rows complete individually.

## Final acceptance gate

Full completion requires all of the following:

- 82/82 executable MRST files have reviewed owners and behavioral evidence;
- no undocumented `NotImplementedError`, placeholder, or silent fallback;
- the MRST-compatible Zudkevitch-Joffe refusal and region-one EOS fallback
  remain tested and documented;
- the dedicated `tests/compositional` suite passes in the standard test run;
- every material EOS, property, fully implicit, natural-variable, sequential,
  deck, well, facility, and example branch has MATLAB regression evidence;
- long-run component mass balance closes to the stated tolerance;
- SPE1/SPE5/multi-region report steps and well results match the agreed parity
  envelope;
- full-size examples complete and produce engineering-useful reports;
- the audit reports zero missing, zero unvalidated executable owners, and zero
  partial rows.

Suggested final commands:

```bash
python3 scripts/audit_compositional_parity.py
python3 -m pytest -q tests/compositional
python3 examples/validation/validate_compositional_suite.py
python3 -m pytest -q tests/ad_core tests/ad_blackoil tests/deckformat
```
