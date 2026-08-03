# Compositional parity ledger

## Scope and interpretation

This ledger compares MRST 2024a `modules/compositional/` with
`pymrst/modules/compositional/`. It records the state observed on 2026-08-01.

“Owner present” means a Python implementation or compatibility module exists.
It does not mean every MATLAB option and edge case has been validated.

Status definitions:

- **Validated core**: representative numerical branches match MATLAB.
- **Broad**: substantial behavior exists, but method/option completeness is not
  yet proven.
- **Partial**: an explicit unsupported branch or material missing workflow is
  known.
- **Catalog**: a non-executable `Contents.m` counterpart.

## Inventory summary

| Classification | Count |
|---|---:|
| MRST files | 96 |
| Executable MRST files | 82 |
| MRST `Contents.m` catalogs | 14 |
| Executable files with named Python owners | 82 |
| Missing named executable owners | 0 |
| Dedicated compositional pytest tree | 1 baseline hierarchy (`26 passed`) |
| Python files with embedded `__main__` tests | 37 |
| Critical MATLAB validators sampled for this audit | 17 passed |

The module therefore has complete filename coverage but not yet complete
behavioral certification.

## Folder-level parity

| MRST area | Files | Python layout | Current status | Principal remaining work |
|---|---:|---|---|---|
| `examples/` | 14 | Matching `examples/`, `book_ii/`, `utils/` | Broad | Validate all MRST-size modes, plotting/report artifacts, and external simulator references |
| `fluids/` | 14 | Matching `fluids/` and `blackoil/` | Broad | Complete method/input forms and optional live-CoolProp matrix |
| `models/` | 29 | Matching `models/eos`, `natvars`, `overall`, `sequential` | Broad | Method-level audit, thin dispatcher contracts, all equation and convergence branches |
| `properties/` | 10 | Matching `properties/` and `components/` | Validated core | Complete AD/subset/region edge matrix and cross-property consistency |
| `utils/` | 29 | Matching `utils/` and `units/` | Broad | Solver/failure paths, utility edge cases, and broader integration certification |

## Models and equation assembly

| MRST owner group | Python owner group | Status | Evidence | Remaining gap |
|---|---|---|---|---|
| `ThreePhaseCompositionalModel.m` | `models/three_phase_compositional_model.py`, centralized `model.py` | Broad | Phases 53, 54–59, 81–83 and later fixtures | Audit complete MRST method surface, convergence/CPR, region and failure branches |
| `GenericOverallCompositionModel.m` | `models/generic_overall_composition_model.py`, `model.py` | Validated core | Phases 53, 82, 114–115 | Complete generic component/non-EOS/region and state-function contracts |
| `GenericNaturalVariablesModel.m` | `models/generic_natural_variables_model.py`, `natural_variables.py` | Validated core | Phases 80–81, 83, 114–115 | Phase-transition edge cases, update limits, origin/primary-variable API parity |
| Overall model/equations | `models/overall/`, `model.py` | Broad | Fully implicit flow and BC schedules pass | `equations_compositional.py` is primarily a compatibility landing file; expose and test callable MRST contract |
| Natural model/equations | `models/natvars/`, `natural_variables.py` | Broad | Natural flow, water, wells, BC pass | Complete all 420-line equation helper branches and reduced-system integration |
| Overall sequential | `models/overall/sequential/` | Broad | Phases 121–124 pass | Full pressure/transport options, rollback/timestep/reporting and dispatcher contracts |
| Natural sequential | `models/natvars/sequential/` | Broad | Sequential fixtures pass | Full transport equations, pressure reduced system, BHP correction, all forces |
| Natural pressure reduced system | `models/natvars/sequential/pressure_reduced_linear_system.py` | Broad | Phases 59/299/300; focused weighted-assembly and reconstruction tests | Add long-run field/well reconstruction regression |

## EOS, fluids, and properties

| Capability | Main Python owners | Status | Evidence | Remaining gap |
|---|---|---|---|---|
| PR/SRK cubic EOS and flash | `eos.py`, `eos_ad.py`, `models/eos/equation_of_state_model.py` | Validated core | Phase 13 and flash/property fixtures | Near-critical/pure/disappearing-phase and complete derivative matrix |
| Zudkevitch-Joffe EOS | `models/eos/equation_of_state_model.py` | Accepted MRST exclusion | PyMRST and MRST 2024a both explicitly refuse it | Preserve refusal and test source-version compatibility |
| Deck EOS initialization | `utils/init_deck_eos_model.py` | Accepted MRST behavior | Phase 269 | Multiple regions warn and select region 1 in both MRST and PyMRST |
| Mixture/component data | `fluids/compositional_mixture.py`, `properties/components/` | Broad | Benchmark and example fixtures | Method/input/region completeness audit |
| Density, viscosity, phase fractions, fugacity | `properties/` | Validated core | EOS and flow validators | All AD, subset, volume-shift and region combinations |
| Table fluids | `fluids/table_compositional_mixture.py` | Validated core | Phase 268; all 122 MRST records present | Preserve values, order, lookup, aliases and validation |
| CoolProp fluids | `fluids/coolprops_compositional_mixture.py` | Broad | Optional implementation exists | CI matrix with dependency present/absent and fixed-state MRST comparison |
| Black-oil conversion | `fluids/blackoil/` | Validated core | Phases 110 and 114 | Broader decks, regions, Rs/Rv edge cases and long schedules |
| Partial volumes/IMPES weights | `utils/get_partial_volumes.py`, `get_impes_weights_overall_composition.py` | Broad | Phases 59, 275, 284; mixed-phase, reorder, well-column, iteration, water-first mass/molar volumes, native single-phase cubic-EOS AD, and MRST-style two-phase pressure perturbation | Certify near-critical and large-model performance behavior |

## Wells, boundaries, facilities, and schedules

| Capability | Status | Evidence | Remaining gap |
|---|---|---|---|
| Component-aware wells and surface rates | Validated core | Natural/overall and sequential well fixtures | Long schedules, all controls/limits, failure recovery |
| Pressure/flux BCs with composition | Validated core | Phases 82, 83, 115, 118 | Region combinations and malformed-force tests |
| Separators and separator trees | Validated core | Phase 116 | Couple to complete schedule/restart workflows |
| Groups, guide rates, VFP and completion controls | Broad | Extensive facility implementation and later phase validators | Consolidated end-to-end deck schedule regression |
| Restart and summary output | Partial | State structures exist | Explicit compositional restart reader/writer and summary comparison gate |

## Examples

Every executable MRST example has a corresponding Python owner:

| MRST examples | Python location | Status |
|---|---|---|
| Boundary conditions, 3D six-component, larger tutorial, validation, SPE1 | `examples/` | Runnable; full-scale and output parity incomplete |
| Accuracy, miscibility, performance, separator, VLE, natural/overall comparisons | `examples/book_ii/` | Runnable; comprehensive MRST-size regression incomplete |
| Simple setup utility | `examples/utils/` | Implemented |

Known example gaps:

- several examples default to reduced grids and expose `full_size=True`;
- `compositionalValidationSimple` now has a checksum-verified optional dataset
  installer and loader. It aligns `statesECL(2:end)` with all 499 AD-GPRS
  states and exposes the imported pressure, composition, saturation, density,
  viscosity, and phase data. Full mode now matches the 1000-cell grid, deck
  component ordering, rock, BHP controls, and 500-step schedule and computes
  pressure/composition/saturation error summaries; the complete hours-scale
  serial run and acceptance thresholds remain;
- the performance fixture validates structure and available kernels rather
  than a complete full-scale backend benchmark;
- plot/table/result artifacts are not yet accepted as parity evidence for every
  example.

## Validation evidence sampled during this audit

The following validators were executed successfully:

```text
validate_phase13.py   PR flash
validate_phase53.py   fully implicit compositional flow
validate_phase80.py   natural variables
validate_phase81.py   water natural variables
validate_phase82.py   overall-composition BC schedule
validate_phase83.py   natural-variable BC schedule
validate_phase110.py  SPE1 black-oil conversion
validate_phase114.py  native SPE1 overall/natural solve
validate_phase115.py  BC schedule parity
validate_phase116.py  separators
validate_phase117.py  performance structure
validate_phase118.py  BC example setup
validate_phase121.py  sequential transport
validate_phase122.py  sequential transport with water
validate_phase123.py  sequential wells
validate_phase124.py  sequential wells with water
validate_phase336.py  SimpleWell helper behavior
```

Blocker 4 (phase 114's broad SPE1 envelope) is **resolved**. It previously
accepted 5 bar and observed 4.6; the overall-composition model now agrees with
MATLAB to 4.6e-04 bar and the envelope has been tightened accordingly. The full
account is in the `validate_phase114.py` docstring; in summary:

* The previous attribution to "pseudo-component composition interpolation" was
  **wrong**. Phase 111 matches every K-value and property table to 1e-13 or
  better, and rebuilding the tables on MRST's literal raw coordinates makes the
  gap worse (32 bar) - the normalization is load-bearing.
* Also ruled out by measurement: timestep control, convergence (the gap was
  fixed at 4.5726 bar from tol 1e-5 to 1e-11), and transmissibility, well
  index, relperm and initial state (phase 24 matches WI to 4e-28 and the
  black-oil history to ~1e-14; phase 110 matches the initial state).
* The actual cause: **the compositional model carried no rock or water
  compressibility**. `self.pv` is fixed at construction, and the accumulation
  terms used that same constant pore volume at both time levels, so `pvMultR`
  cancelled out; the water equation was `pv (sW - sW0)/dt` with no `b_w`, which
  pins `sW` exactly wherever water is immobile. MRST keeps the deck's
  `bW`/`muW` through the conversion and carries `pvMultR` with the rock.
* Restoring both (`water_b`, `rock_pv_mult`, wired from the deck's PVTW/ROCK)
  moved the overall model from 4.5726 bar to 4.6e-04, and `sW` spread from
  2.8e-17 to 7.32e-04 against MATLAB's 7.32e-04. Rock was the larger term
  (4.57 -> 0.40 alone); water closed the remainder.

Both are opt-in - a model constructed without them behaves exactly as before -
so no other compositional result moved. All 20 compositional validators pass.

A second discrepancy surfaced and turned out to be **in the reference, not in
PyMRST**. The natural-variable model sat at 7.0e-02 bar against the old .mat,
150x looser than the overall model. But PyMRST's two formulations agreed with
each other to 1.0e-10 bar while MATLAB's two disagreed by 7.08e-02 bar: the
reference had been dumped at MRST's default 1e-3 tolerances and was not
converged. `dump_phase114_reference.m` now tightens to 1e-7 (CNV, well rate)
before running either model; MATLAB's internal spread drops to 6.3e-06 bar and
both PyMRST models match it to ~6e-05 bar. No PyMRST code changed for this.

Tightened phase 114 tolerances - one budget, both formulations, since they now
agree:

| check | old | now | achieved |
|---|---|---|---|
| pressure | 5 bar | 5e-3 bar | 6.1e-05 bar |
| components | 1e-3 | 1e-7 | 4.1e-09 |
| saturations | 5e-3 | 1e-6 | 8.9e-08 |
| well BHP | 3 bar | 5e-3 bar | 6.3e-05 bar |
| oil rate | 1.2e-3 | 5e-8 | 2.5e-09 |

Newton iterations are now checked as a direction (PyMRST must not exceed
MATLAB) rather than an equality: converged MRST needs [4 7] and [9 10] where
PyMRST needs [3 4], so equality would fail on PyMRST being faster.

A separate defect was found and fixed during this diagnosis: the K-value
Rachford-Rice solver decided phase state from whether every `K_i` sat on one
side of 1, which is necessary but not sufficient, and then bisected `[0, 1]`
without checking that a root existed there. Single-phase liquid with
`K_oil < 1 < K_gas` - ordinary undersaturated oil - collapsed onto `V = 1` and
was reported as **all vapour**. On a sweep of that regime it hit 176 of 4000
states, each with the maximum possible error. SPE1 never reaches the branch
(0 of 22200 evaluations), so no existing validator moved, but any case that
does reach it was silently wrong. Fixed in `kvalue_model.py` with the endpoint
sign test `f(0) > 0 > f(1)`, plus self-tests.

**The same defect had a second copy**, found later by writing the missing
self-tests for `models/eos/equilibrium_constant_model.py`. Its `_rachford_rice`
bisected `[0, 1]` with no bracket check, so `z = [0.5, 0.5]`, `K = [1.6, 0.2]` -
where `f(0) = -0.10` and `f(1) = -1.81`, so no root exists and the feed is
entirely liquid - returned `V = 1.0`. Fixed with the same guard.

All four Rachford-Rice implementations were then checked against that case
rather than assumed correct, because a defect present in two of four copies is
exactly the kind that hides in the others:

| implementation | before |
|---|---|
| `fluids/blackoil/kvalue_model._rachford_rice_k` | wrong, fixed |
| `models/eos/equilibrium_constant_model._rachford_rice` | wrong, fixed |
| `utils/solve_rachford_rice_vle` | already correct |
| `eos.rachford_rice` | already correct |

Four separate implementations of one equation is itself the finding. Each is
small, and each is a place the next bracket bug can appear independently.

## Module self-tests

120 of 121 compositional modules pass their own `__main__` tests (counting the
one argparse CLI the standard harness skips). The remaining one,
`ML_models/Web_app/app.py`, needs `streamlit`, which is an optional dependency
rather than a defect.

Two long-standing assertion failures were resolved, and in both cases the test
was wrong rather than the code:

* `compositional_validation_simple.py` asserted `qGs[0] > 0` for the injector.
  MRST's `setupSimpleCompositionalExample` sets `compi = [1, 0]` under the
  comment "Injection is pure gas", but the phases are (Oil, Gas), so `[1, 0]`
  selects oil - and MRST itself reports `INJ qOs=1.5828141e-05 qGs=0` when run.
  The assertion had encoded MRST's stale comment rather than its behavior. The
  same test also pinned `available == ("MRST",)`, asserting that the optional
  E300/AD-GPRS datasets are absent; they live in MRST's `simplecomp` archive
  outside this repo and are installed here, so the check now asserts the
  invariant (available exactly when loaded) rather than one machine's state.
* `model.py`'s separator test failed on convergence, not on physics. The
  separator conversion is correct - 1 kg/s of methane returns 1.531 m3/s
  against an ideal-gas 1.5345 - and the mass stream fed to it matches the plain
  model's mass balance exactly. It differs from the non-separator path by ~117x
  only because the separator does a real flash at 1 atm (~1.1 kg/m3) where the
  plain path uses the model's nominal `rho_s[1] = 100 kg/m3`. What failed was
  the default `tol_well = 1e-10`: with a separator flash inside the residual
  the attainable floor sits just under 1e-10, so the model converges in 6
  iterations at 1e-9 and never at 1e-10, regardless of iteration count or
  timestep (30 days down to 0.1 does not help). The test now runs both models
  at 1e-9.

That last one is the same signature as the Norne case (`tol_well = 1e-10`
missed by a small factor while every other residual passes) and is worth
treating as one repo-wide tolerance decision rather than case by case.

## Explicitly known blockers to “complete”

1. The new dedicated `tests/compositional` suite is only a baseline and does
   not yet cover every owner and option family.
2. All 82 executable rows do not yet have individual focused-test and MATLAB
   evidence in a machine-readable ledger.
3. Full-size example comparison, restart/summary, and complete performance
   acceptance is missing.
4. ~~SPE1 passes a broad rather than tight numerical envelope.~~ **Resolved.**
   The cause was missing rock and water compressibility in the compositional
   model, not interpolation; the overall-composition model now matches MATLAB
   to 4.6e-04 bar and the envelope is tightened to match. See the phase 114
   notes above. A second discrepancy found during the same work (the
   natural-variable model at 7.0e-02 bar) was traced to the MATLAB reference
   being under-converged, not to PyMRST; the reference has been regenerated and
   both formulations now agree with it to ~6e-05 bar.

## Completion declaration rule

Do not mark this module fully complete until the machine-readable audit reports
82/82 executable owners complete and validated, all explicit blockers above
are resolved or formally accepted exclusions, the standard pytest suite
discovers compositional tests, the full MATLAB regression suite passes, and
full-size end-to-end deck/example results satisfy documented engineering
tolerances and conservation limits.
