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

Important qualification: phase 114 currently accepts pressure differences of
up to 5 bar and observed about 4.6 bar. This is useful workflow evidence, but
the native SPE1 numerical gap should be diagnosed and the acceptance envelope
tightened before declaring full parity.

## Explicitly known blockers to “complete”

1. The new dedicated `tests/compositional` suite is only a baseline and does
   not yet cover every owner and option family.
2. All 82 executable rows do not yet have individual focused-test and MATLAB
   evidence in a machine-readable ledger.
3. Full-size example comparison, restart/summary, and complete performance
   acceptance is missing.
4. SPE1 passes a broad rather than tight numerical envelope.

## Completion declaration rule

Do not mark this module fully complete until the machine-readable audit reports
82/82 executable owners complete and validated, all explicit blockers above
are resolved or formally accepted exclusions, the standard pytest suite
discovers compositional tests, the full MATLAB regression suite passes, and
full-size end-to-end deck/example results satisfy documented engineering
tolerances and conservation limits.
