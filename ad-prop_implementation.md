# ad-props end-to-end implementation plan

## Objective

Complete `pymrst/modules/ad_props` against the MRST 2024a `ad-props/` tree to
the standard already applied to `gridprocessing`, `ad-core` and `vemmech`:

- every relevant MRST production file is inventoried;
- **every implementation sits at its MRST-shaped path**, not behind a
  re-export;
- every numerical API has a direct or clearly documented consolidated Python
  counterpart;
- MATLAB-only and catalog-only files are explicitly classified rather than
  silently omitted;
- Python behavior is covered by focused unit and invariant tests;
- numerical behavior is validated against independently generated MATLAB/MRST
  reference data;
- a full ECLIPSE deck builds a fluid object end to end and agrees with MRST;
- the parity audit reports zero pending and zero ported-but-unvalidated items.

The target MRST source tree is:

```text
mrst-2024a-Geomech/modules/ad-props
```

The target PyMRST package is:

```text
PyMRST/pymrst/modules/ad_props
```

## The layout contract

This is the first constraint, not an afterthought, because the current package
violates it.

MRST's `ad-props/` has exactly **two files at its root** - `initSimpleADIFluid.m`
and `imposeRelpermScaling.m` - and everything else under `props/`, `simple/`
and `utils/`. PyMRST must mirror that directory for directory, with the two
mechanical transforms used everywhere else in the port:

- hyphens become underscores (`ad-props` -> `ad_props`);
- MATLAB's one-function-per-file convention may collapse into grouped modules
  where that reads better, but **never across a subdirectory boundary**.

**The package partly inverts this.** Three of the largest modules are invented
root-level files holding implementation that belongs in subdirectories, and two
MRST-named files are shims forwarding back to them:

| file | lines | what it is |
|---|---:|---|
| `blackoil_fluid.py` | 268 | implementation at an invented root path; belongs under `props/` |
| `tables.py` | 210 | implementation at an invented root path; belongs under `utils/` |
| `simple_adi_fluid.py` | 105 | implementation, misnamed for `initSimpleADIFluid` |
| `props/init_deck_adi_fluid.py` | 15 | **shim** - forwards to `blackoil_fluid` |
| `utils/interp_pvt.py` | 10 | **shim** - forwards to `tables` |

A reader looking for `interpPVT` finds a ten-line file that forwards to
`tables.py`. That is the opposite of the contract, and Phase 2 exists to undo
it. Until it is undone, a parity audit matching on filenames alone reports this
module as more complete than it is - which is why the ledger detects shims by
parsing each module rather than trusting its name.

Three other MRST-named files are **not** shims and are already correct:
`utils/extend_tab.py` and `simple/corey_phase_relperm_ad.py` implement their
symbol at the right path, and `simple/table_by_pressure_linear_ad.py` does real
work while importing one helper from `tables.py` - so it moves only because
`tables.py` does.

## Current implementation status

Measured against the MRST tree, not estimated:

| | MRST | PyMRST |
|---|---:|---:|
| files | 67 `.m` (4 are `Contents.m`, so 63 production) | 12 `.py` |
| lines | 3,714 | 708 |

Of the 63 production files:

- **3** are implemented at their MRST path (`coreyPhaseRelpermAD`,
  `tableByPressureLinearAD`, `extendTab`);
- **2** have a file of the right name that only forwards (`initDeckADIFluid`,
  `interpPVT`), so they are counted as shims, not ports;
- **1** is consolidated into another module (`initSimpleADIFluid`, in
  `simple_adi_fluid.py`);
- **57 are absent**, including every `assign*` function bar the four whose
  behavior is inlined in `blackoil_fluid.py` without a counterpart file.

`AD_PROPS_PARITY.md` is the live count: **4 of 63 production files
implemented**.

What does exist works: 36 validation scripts name `ad_props` and all 36 pass,
and eight other modules import it, so the black-oil path it does cover is
exercised and load-bearing. The gap is breadth, not soundness.

### The three structural gaps

Counting files understates this. There are three architectural gaps behind the
53 missing files.

**1. No keyword dispatch.** MRST's `initDeckADIFluid` (166 lines) reads
`deck.PROPS` and dispatches to ~47 `assign<KEYWORD>` functions. PyMRST's
`init_deck_blackoil_fluid` hard-codes a handful of properties inline and
asserts its way out of the rest - `assert w["viscosibility"] == 0.0` is a
literal line. There is no dispatch table to extend.

**2. No regions.** `getRegMap`, `handleRegions`, `interpReg`, `interpRegPVT`
and `dinterpReg` are all missing, and nothing in the package understands
`SATNUM`/`PVTNUM`. Every property is global. This is the single largest
functional gap: multi-region decks cannot be represented at all, and it is a
prerequisite for the `assign*` family rather than an extra.

**3. The deck itself is a different object.** MRST dispatches on
`deck.PROPS.<KEYWORD>` and reads regions from `deck.REGIONS`, both produced by
`deckformat`'s general ECLIPSE reader. PyMRST's `deckformat` has none of that
layer - no `readEclipseDeck`, `readPROPS`, `readREGIONS`, `initializeDeck` or
`convertDeckUnits` - only GRDECL reading plus five bespoke per-deck readers
(SPE1, SPE3, SPE9, Egg, Norne) emitting a flat lowercase dict, `deck["pvto"]`,
`deck["swof"]`, with no PROPS section and no region keywords. MRST has no
per-deck readers at all; it reads every deck with
`convertDeckUnits(readEclipseDeck(fn))`. `initDeckADIFluid` cannot be ported
faithfully onto a deck that does not carry what it dispatches on. Phase 4
addresses this and is the plan's main dependency risk.

## Definition of done

ad-props is complete only when all the following are true:

1. A reproducible parity scanner inventories every MRST `.m` file under
   `ad-props/`.
2. Every file is classified as `ported`, `ported-consolidated`,
   `excluded-catalog`, or `excluded-matlab-only`, with an explicit reviewed
   reason for every exclusion.
3. **No implementation file sits outside its MRST-shaped path**, and no
   MRST-named file is a re-export of an invented module.
4. No entry is `pending-review`, and no ported entry lacks a naming test or
   reference validator.
5. A multi-region ECLIPSE deck builds a fluid whose properties agree with
   MRST's, region for region.
6. A `tests/ad_props/` pytest suite covers the module's invariants, not only
   its saved reference values.

## Compatibility conventions

### Region indices

MRST region numbers are 1-based and appear directly in deck keywords
(`SATNUM`, `PVTNUM`, `IMBNUM`). PyMRST stores them **0-based**, converting once
at the deck boundary, and every MATLAB reference dump converts on the MATLAB
side so the validator compares like with like. This is the convention already
fixed by `gridprocessing`, and regions are where it is easiest to get wrong.

### Table extrapolation

`extendTab` pads a table so interpolation outside its range is well defined.
Whether a property is extended, and by how much, changes answers near the
endpoints without raising anything. Each ported `assign*` must extend exactly
what MRST extends - this is a per-keyword decision, not a global policy.

### Units

Deck values are in field or lab units; MRST converts to SI on read. PyMRST's
existing readers already convert, so `assign*` receives SI. Any new PROPS
section must preserve that, and the validators must compare in SI to avoid a
unit error cancelling against a conversion error.

### AD compatibility

Every property returned must accept both plain arrays and AD objects, because
these functions are called inside residual assembly. A property that silently
drops derivatives produces a converging-but-wrong Newton solve, so the tests
check derivative propagation, not just values.

### Numerical tolerances

Property values compare at `rtol=1e-10`. Table interpolation compares at
`rtol=1e-12, atol=1e-14`. Where a comparison must be looser the validator
states why inline.

## Phase 1: parity ledger and audit tooling

**Why first.** `ad_core`, `ad_blackoil`, `gridprocessing` and `vemmech` each
have a ledger; `ad_props` does not, so "finished" is not measurable. It also
has to be built *before* the layout refactor, so the refactor can be shown not
to have lost anything.

**Status: done.** `scripts/audit_ad_props_parity.py`,
`docs/ad_props/parity.csv` and `AD_PROPS_PARITY.md` are in place. The scan
reports 67 files: 3 ported, 1 consolidated, 2 shims, 57 pending, 4 catalogs.

The ledger immediately corrected an over-claim in an earlier draft of this
plan, which listed five files as re-export shims. Parsing them showed only two
are; the other three implement their symbol at the right path.

**Two properties the scanner must have**, both learned from the vemmech audit:

- match on *code*, not docstrings, or a module that merely mentions
  `assignPVTO` gets counted as implementing it;
- flag a file as `ported` only when the counterpart is an implementation. A
  re-export shim satisfies a filename match while adding nothing. The scanner
  reports shims as a distinct status, detected by parsing the module - it
  defines nothing, or every function it defines is a one-line forward - so
  Phase 2's progress is visible.

**Size.** Small. Half a day.

## Phase 2: layout refactor

**Why before the ports.** Every phase after this adds files. Adding them
around an inverted layout means moving more later.

**Moves**

| from | to | note |
|---|---|---|
| `simple_adi_fluid.py` | `init_simple_adi_fluid.py` | MRST root file, renamed to its symbol |
| `tables.py` | `utils/interp_pvt.py`, `utils/extend_tab.py`, `utils/preprocess_table_pvt.py` | split by MRST symbol |
| `blackoil_fluid.py` | `props/init_deck_adi_fluid.py` + the `assign*` files it inlines | the bulk of it becomes Phase 5 |

`imposeRelpermScaling.py` is created at the root in Phase 7.

**Constraint.** The re-export shims disappear; they do not get inverted. After
this phase `utils/interp_pvt.py` *contains* the interpolation, and nothing
imports `ad_props.tables`.

**Risk.** Eight modules outside `ad_props` import from it, and 36 validators
name it. The refactor must keep the public surface working: re-export from
`ad_props/__init__.py` if needed, but never from an MRST-named file back to an
invented one.

**Exit criterion.** The ledger reports zero re-export shims and no
implementation outside its MRST path, the 36 validators still pass, and the
eight importing modules keep working.

**Size.** Medium. 1-2 days.

## Phase 3: the interpolation and region core

**Why this is the keystone.** Every `assign*` function returns a closure over
one of these interpolators. Porting the `assign*` family first would mean
writing 47 functions against an interpolation layer that does not exist.

**Files** (line counts exclude comments and blanks)

| MRST file | lines | destination |
|---|---:|---|
| `utils/interpPVT.m` | 234 | `utils/interp_pvt.py` |
| `utils/interpRegular2DTable.m` | 102 | `utils/interp_regular_2d_table.py` |
| `utils/interp2DTable.m` | 86 | `utils/interp_2d_table.py` |
| `utils/handleRegions.m` | 81 | `utils/handle_regions.py` |
| `utils/getRegMap.m` | 66 | `utils/get_reg_map.py` |
| `utils/interp2DPVT.m` | 63 | `utils/interp_2d_pvt.py` |
| `utils/interpReg.m` | 52 | `utils/interp_reg.py` |
| `utils/interpRegPVT.m` | 40 | `utils/interp_reg_pvt.py` |
| `utils/preprocessTablePVT.m` | 30 | `utils/preprocess_table_pvt.py` |
| `utils/extendTab.m` | 25 | `utils/extend_tab.py` |
| `utils/dinterpReg.m` | 24 | `utils/dinterp_reg.py` |

Total ~803 lines. `interpPVT` already has a partial counterpart in `tables.py`
that Phase 2 relocates; it must be reconciled against the full MRST version,
including the saturated/undersaturated branch handling.

**Where the risk is.** `interpPVT` is 234 lines because it handles the
saturated-to-undersaturated transition and the flag array that selects between
them. Getting the branch selection subtly wrong gives plausible properties that
are wrong only near the bubble point - exactly where a black-oil solve spends
its time.

**Validation.** A MATLAB dump per interpolator on fixed tables and query
points, including points deliberately outside the table range so extrapolation
is pinned, and multi-region cases for the `*Reg*` family.

**Size.** Large. 3-4 days.

## Phase 4: the deck reader, in `deckformat` where MRST puts it

**The dependency.** MRST's `initDeckADIFluid` dispatches on
`deck.PROPS.<KEYWORD>` and reads its region partition from `deck.REGIONS`.
PyMRST has neither.

**This is settled by MRST's own architecture, not by preference.** MRST places
the deck reader in `deckformat/deckinput/`, not in `ad-props`:

- `readEclipseDeck.m` builds the seven-section deck, calling `readPROPS.m` and
  `readREGIONS.m`;
- `convertDeckUnits.m` converts it to SI, with a dedicated `convertPROPS`;
- `ad-props` only *consumes* the result.

So `ad_props` defines no deck contract of its own and gains no adapter. The
work belongs to `deckformat`, and it lands at MRST's paths. This phase is on
ad-props' critical path but is not ad-props' code, and should be tracked
against `deckformat` when that module gets its own plan.

**MRST reads every deck with one reader.** There are no per-deck readers in
the MRST tree - its own examples call
`convertDeckUnits(readEclipseDeck(fn))` for SPE1, SPE9 and the rest.
PyMRST's five bespoke readers (`spe1/`, `spe3/`, `spe9/`, `egg/`, `norne/`)
have no MRST counterpart. They are what stands in for the general reader today,
and the general reader supersedes them.

**Files** (line counts exclude comments and blanks)

| MRST file | lines | destination |
|---|---:|---|
| `deckinput/readPROPS.m` | 403 | `deckformat/deckinput/read_props.py` |
| `deckinput/convertDeckUnits.m` (`convertPROPS` part) | 280 | `deckformat/deckinput/convert_deck_units.py` |
| `deckinput/readREGIONS.m` | 81 | `deckformat/deckinput/read_regions.py` |
| `deckinput/readRelPermTable.m` | 73 | `deckformat/deckinput/read_rel_perm_table.py` |
| `deckinput/readMisciblePVTTable.m` | 66 | `deckformat/deckinput/read_miscible_pvt_table.py` |
| `deckinput/readEclipseDeck.m` | 60 | `deckformat/deckinput/read_eclipse_deck.py` |
| `deckinput/initializeDeck.m` | 52 | `deckformat/deckinput/initialize_deck.py` |
| `deckinput/readDefaultedRecord.m` | 35 | `deckformat/deckinput/read_defaulted_record.py` |
| `deckinput/readDefaultedKW.m` | 33 | `deckformat/deckinput/read_defaulted_kw.py` |
| `deckinput/readImmisciblePVTTable.m` | 27 | `deckformat/deckinput/read_immiscible_pvt_table.py` |

Roughly 1,110 lines for a PROPS- and REGIONS-capable reader. `readPROPS`
alone covers **59 keyword cases** and `readREGIONS` the nine region keywords
(`SATNUM`, `PVTNUM`, `IMBNUM`, `EQLNUM`, `FIPNUM`, `ROCKNUM`, `SURFNUM`,
`ENDNUM`, `EOSNUM`).

**Scope discipline.** The full `convertDeckUnits.m` is 895 lines covering every
section. Only `convertPROPS` is on this critical path; the RUNSPEC, GRID,
SOLUTION and SCHEDULE converters belong to `deckformat`'s own plan and should
not be pulled in here to avoid an unbounded phase.

**Migration, not replacement.** The five existing readers stay working while
the general reader is built. Once SPE1, SPE3 and SPE9 parse identically through
`read_eclipse_deck`, the bespoke readers become thin wrappers over it and are
then removed. The test that both routes produce the same deck is what makes
that switch safe, and it is the deliverable that matters most in this phase.

**Deliverables**

- `deckformat/deckinput/` at MRST's layout, with the ten files above;
- `deck.PROPS` and `deck.REGIONS` populated and converted to SI;
- a test that SPE1, SPE3 and SPE9 parse to the same values through the general
  reader as through their current bespoke readers;
- MATLAB references for the parsed PROPS tables of at least one full deck.

**Size.** Large, and the largest single risk in this plan. 4-6 days.

### Phase 4 status: complete

`deckformat/deckinput/` now holds `read_defaulted_record.py` (also
`readDefaultedKW` and `readFixedNumRecords`, which are one file here because
they share the cursor), `read_rel_perm_table.py`, `read_pvt_tables.py` (both
PVT readers, plus `fill_usat_invlinear`), `read_regions.py`, `read_props.py`,
`read_runspec.py`, `initialize_deck.py`, `read_eclipse_deck.py` and
`convert_deck_units.py`, with `unit_conversion_factors.py` beside them.

`read_runspec.py` was not in the file table above and turned out to be
required: TABDIMS carries NTSFUN and NTPVT, and a PROPS reader that assumes
one region reads the first table of a multi-region deck and drops the rest.

**Validation.** Two independent checks, and the second is the one that
mattered:

- `tests/deckformat/test_deckinput_matches_bespoke_readers.py` (39 tests)
  compares against PyMRST's SPE1/SPE3/SPE9 readers;
- `examples/validation/validate_deckinput_props.py` compares against
  `ref_deckinput_props.mat`, dumped by MRST's own
  `convertDeckUnits(readEclipseDeck(...))`. **All PROPS values for SPE1, SPE3
  and SPE9 match MRST to 1e-10 relative.**

The bespoke-reader comparison passed while two real defects were still
present, because both readers were PyMRST code and shared the gap. The MRST
reference caught them:

- `ROCK` is a six-field record defaulting to NaN, not a two-field record
  defaulting to zero. Several other record templates were wrong the same way
  (`SDENSITY`, `RSCONSTT`, `PLYROCK`, `PLMIXPAR`, `SURFROCK`, `SHRATE`), and
  are now transcribed field-for-field from `readPROPS.m`.
- PVTO branches that state only a bubble point must be extended by
  `fill_usat_invlinear`. SPE9 has ten such branches, so MRST produces 22 rows
  where the bespoke reader produces 12.

**A defect this exposed in existing PyMRST code.** `spe9/read_spe9_deck.py`
does not apply the undersaturated fill, so it hands the simulator PVTO
branches that stop at their bubble point. Any SPE9 result computed through it
is affected. This is recorded, not worked around, in
`test_spe9_pvto_gains_the_undersaturated_rows_the_bespoke_reader_lacks`; it is
fixed when that reader is retired in favour of `read_eclipse_deck`.

**Deliberately out of scope.** `read_props.py` covers 52 of readPROPS's 323
keyword cases. The remainder are almost entirely the endpoint-scaling family
(`SWL`, `SWCR`, `KRW`, `SGCR` and their directional variants), which are
per-cell arrays rather than tables and belong with Phase 7's relative
permeability scaling, plus the box and array-arithmetic keywords (`BOX`,
`ADD`, `COPY`, `EQUALS`, `MULTIPLY`) which belong with the GRID reader.
Unrecognised keywords are appended to an `unhandled` list rather than dropped
silently, so the gap is visible from a parsed deck.

## Phase 5: core PVT and saturation-function properties

The properties a black-oil deck cannot run without.

| group | files | lines |
|---|---:|---:|
| core PVT (`PVTO`, `PVTG`, `PVTW`, `PVDO`, `PVDG`, `PVDS`, `PVCDO`, `RSCONSTT`, `DENSITY`, `SDENSITY`, `GRAVITY`, `ROCK`, `ROCKTAB`) | 13 | 375 |
| saturation functions (`SWOF`, `SGOF`, `SLGOF`, `SWFN`, `SGFN`, `SGWFN`, `SOF2`, `SOF3`, `SSFN`, `MSFN`, `RelPerm`, `RelPermScal`, `EHYSTR`) | 13 | 522 |

Each lands at `props/assign_<keyword>.py`.

**Where the risk is.** `assignSWOF` and `assignSGOF` build both relative
permeability *and* capillary pressure, with endpoint handling that decides
what happens at residual saturation. `assignRelPermScal` and `EHYSTR`
(scaling and hysteresis) are the hardest and should come last in the phase.

**Validation.** Property values on a fixed saturation and pressure sweep,
per region, against MATLAB - and derivatives, since these are called under AD.

**Size.** Large. 4-5 days.

## Phase 6: EOR and thermal property families

Everything a black-oil deck does not need, which is why it comes after Phase 5.

| group | files | lines |
|---|---:|---:|
| polymer (`PLYADS`, `PLYMAX`, `PLYROCK`, `PLYSHEAR`, `PLYSHLOG`, `PLYVISC`, `PLMIXPAR`, `TLMIXPAR`, `SHRATE`) | 9 | 168 |
| surfactant and miscible (`SURFADS`, `SURFCAPD`, `SURFROCK`, `SURFST`, `SURFVISC`, `MISC`, `PMISC`) | 7 | 150 |
| thermal (`SPECHEAT`, `SPECROCK`, `WATERVIST`, `OILVIST`, `VISCREF`) | 5 | 111 |

These are mostly table lookups over the Phase 3 interpolators and are more
repetitive than difficult. They are, however, only meaningful if a consumer
exists: check whether PyMRST's polymer/surfactant/thermal models are ported
before porting their properties, and record any that have no consumer as
deliberately deferred rather than quietly skipping them.

**Size.** Medium. 2-3 days.

## Phase 7: dispatch and relperm scaling

**Files**

| MRST file | lines | destination |
|---|---:|---|
| `props/initDeckADIFluid.m` | 166 | `props/init_deck_adi_fluid.py` |
| `initSimpleADIFluid.m` | 85 | `init_simple_adi_fluid.py` |
| `imposeRelpermScaling.m` | 75 | `impose_relperm_scaling.py` |

`initDeckADIFluid` is the dispatch table the whole module hangs off: it builds
the region map, then calls each `assign*` whose keyword is present. It comes
after the assign functions because it is their consumer.

Its option surface (`region_method`, `pvtMethodOil`, `pvtMethodGas`,
`resamplePoly`, `ignoreErrors`, `optimize`, `useMex`) should be ported as
named arguments, with `useMex` classified as not applicable and the rest either
implemented or explicitly rejected with a clear message. Silently ignoring an
option that changes results is the failure mode to avoid.

**Size.** Medium. 2 days.

## Phase 8: tests and validation

`ad_props` has **no `tests/` directory**. `tests/` holds `ad_blackoil`,
`ad_core`, `gridprocessing` and `vemmech`. Its only coverage is the 36
`validate_phase*.py` scripts, which pin saved values but do not test
invariants, regions, error paths, or AD derivative propagation.

**Deliverables**

- `tests/ad_props/` covering:
  - monotonicity where physics requires it - `krw` non-decreasing in `Sw`,
    `kro` non-increasing, `Bo` behaviour across the bubble point;
  - endpoint behaviour: properties at and beyond table limits, so
    extrapolation is pinned rather than incidental;
  - region selection: two regions with different tables return different
    values for the same input, and the 0-based conversion is exercised;
  - AD propagation: every property returns derivatives when handed an AD
    object;
  - the full-deck path: SPE1, SPE3 and SPE9 build a fluid whose properties
    match MATLAB.
- registration in the Phase 1 ledger so every ported entry is named by a test,
  taking the "ported but unvalidated" table to zero.

**Naming note.** `tests/ad_core/` and `tests/gridprocessing/` already collide
on three basenames, which blocks whole-suite collection. Any new
`tests/ad_props/` parity tests must use distinct basenames so this phase does
not deepen an existing problem.

### Phase 8 status: complete

`tests/ad_props/` holds 72 tests in four files, all with distinct basenames:

| file | covers |
|---|---|
| `test_deck_to_fluid_end_to_end.py` | SPE1/3/9 from deck text to fluid, both keyword families, bounds and call signatures |
| `test_fluid_carries_derivatives.py` | AD propagation, checked against finite differences |
| `test_multi_region_deck.py` | TABDIMS, SATNUM/PVTNUM, per-region tables, driven from deck text |
| `test_deck_fluid_matches_mrst.py` | the MRST comparison, as a test rather than only a script |

**Three MATLAB-backed validators**, all passing:

- `validate_deckinput_props.py` - the reader, against `readEclipseDeck` +
  `convertDeckUnits`;
- `validate_ad_props_assign.py` and `validate_ad_props_assign2.py` - every
  assign function MRST can evaluate, on fixed probes;
- `validate_ad_props_deck_fluid.py` - **the decisive one**: a deck file through
  reader, unit conversion and dispatcher, against MRST doing the same. Every
  property matches to 1e-10 relative, and the *field sets* match too, so
  neither side has a property the other lacks.

**Region coverage came from a synthetic deck, not the benchmarks.** None of
SPE1, SPE3 or SPE9 partitions its field, so a fluid built from any of them
exercises no region code at all. `test_multi_region_deck.py` states two
saturation and two PVT regions with deliberately different curves, which is
what makes a cell reading the wrong region's table visible.

**The "ported but unvalidated" table did not reach zero, and should not.** It
stands at 15 of 63, in three groups:

- **Four cannot be validated against MRST, because MRST cannot run them.**
  `assignWATERVIST` and `assignOILVIST` index `reg.SATKINX`, a field
  `handleRegions` never creates; `assignRelPermScal`'s three-phase branch has
  three separate defects; `assignPLYSHLOG` stores its record unchanged, so
  there is no computed value to compare. Each is documented in its own module.
- **Eight `utils/` interpolation files are exercised transitively** by every
  validator above - `interpPVT` through PVTO and PVTG, `interpReg` through
  SOF2, SGWFN and MISC, `extendTab` through every saturation table - but the
  auditor credits only files a validator *names*, and inferring transitive
  coverage statically is not something it should attempt. Their own self-tests
  cover them directly.
- **Three remain genuinely open**: `imposeRelpermScaling` (needs a model
  object to drive), and the two `simple/` fluids.

One thing this phase changed in the tooling: `validate_ad_props_assign.py`
originally imported the EOR assigners with a dynamic `__import__`, which hid
ten files' coverage from the auditor. Naming them explicitly took the
unvalidated count from 39 to 29 without testing anything new - a reminder that
the ledger measures what it can see, not what is true.

**Size.** Medium. 2-3 days.

## Risks and open questions

- **Phase 4 is the critical path and the largest risk.** It is settled in
  favour of MRST's architecture - the reader lives in `deckformat/deckinput`
  and `ad_props` only consumes it - but it is ~1,110 lines in a module eight
  other packages depend on, and it supersedes five existing readers. Nothing
  from Phase 5 onwards can be ported faithfully until `deck.PROPS` and
  `deck.REGIONS` exist.
- **Regions are pervasive, not additive.** Adding them after the `assign*`
  family would mean rewriting all 47. Phase 3 must land first.
- **`interpPVT`'s branch handling** is where a subtle error would hide, and it
  would show up only near the bubble point.
- **Existing behavior must not regress.** 36 validators pass today and eight
  modules import `ad_props`; the Phase 2 refactor and the Phase 5 rewrite of
  `init_deck_blackoil_fluid` both put that at risk. Each phase should re-run
  those 36 before it is called done.
- **EOR families may have no consumer.** Porting properties nothing can use is
  low value; Phase 6 should confirm consumers first and record deferrals
  explicitly in the ledger.
- **`useMex` and `optimize`** are MATLAB performance options. `useMex` is not
  applicable; `optimize`/`resamplePoly` change the *representation* of a
  property and so can change results at the last digit, and need a reviewed
  decision rather than a default.

## Effort summary

| phase | work | size |
|---|---|---|
| 1 | parity ledger and audit tooling | **done** |
| 2 | layout refactor onto MRST paths | 1-2 days |
| 3 | interpolation and region core (11 files, ~800 lines) | 3-4 days |
| 4 | the deck reader in `deckformat/deckinput` (10 files, ~1,110 lines) | 4-6 days |
| 5 | core PVT and saturation functions (26 files, ~900 lines) | 4-5 days |
| 6 | EOR and thermal families (21 files, ~430 lines) | 2-3 days |
| 7 | dispatch, simple fluid, relperm scaling (3 files, ~330 lines) | 2 days |
| 8 | `tests/ad_props` and full-deck validation | 2-3 days |

**Total: roughly 19 to 26 focused days**, materially larger than the vemmech
effort because the module is wider (63 files against 65, but starting from 12
rather than 55) and because Phases 2, 3 and 4 are structural rather than
transcription.

MATLAB R2021b and the MRST 2024a tree are both available locally, so every
phase can generate its own reference data - the same workflow that took
`gridprocessing` to 1e-14 agreement and `vemmech` to zero pending entries.
