# examples parity ledger

Parity status for MRST's top-level `examples/` tree, which holds the worked
tutorials that introduce the toolbox rather than any module's own examples.

## Scope

`mrst-2024a-Geomech/examples` contains 20 `.m` files, of which 3 are `Contents.m`
catalogs and 17 are executable. Total executable size is about 2920 lines. The
tree also carries 352 non-`.m` files, which are dataset payloads rather than
code and are not port targets.

Module-level examples such as `modules/ad-blackoil/examples` are outside this
ledger. They are tracked with their own modules, and several already have Python
owners.

## Current scan

| Measure | Count |
|---|---|
| MRST files in `examples/` | 20 |
| `Contents.m` catalogs | 3 |
| Executable examples | 17 |
| Executable lines | ~2920 |
| **Executable files with a Python owner** | **17** |
| Missing owners | 0 |

**All seventeen executable examples have Python owners.** Each runs
standalone, and between them they carry 125 self-tests. The folder layout
mirrors MRST exactly: root, `data/egg`, `datasets` and `grids`.

Two examples depend on datasets that are not installed, `showCaseB4` (0.5 MB)
and `showSAIGUP` (2.5 MB). Both are written, both skip cleanly, and both assert
that loading never triggers a download.
`gridTutorialCornerPoint` waits on the `coarsegrid` module, not on the deck
functions the plan originally named; see the correction below.

## Status by area

| Area | Files | Lines | Owners | Blocking dependency |
|---|---|---|---|---|
| root tutorials | 5 | 1015 | **5** | none remaining |
| `grids/` | 4 | 680 | **4** | complete |
| `datasets/` | 5 | 875 | **5** | complete |
| `data/Egg/.../MRST/` | 3 | 350 | **3** | none remaining |

## In-scope entries

| MRST path | Lines | Python owner | Status | Note |
|---|---|---|---|---|
| `flowSolverTutorial1.m` | 73 | `pymrst/examples/flow_solver_tutorial1.py` | ported | checked against the analytic hydrostatic profile |
| `flowSolverTutorialAD.m` | 334 | `pymrst/examples/flow_solver_tutorial_ad.py` | ported | mass balance closes to 3.6e-13 relative |
| `tutorialAD.m` | 151 | `pymrst/examples/tutorial_ad.py` | ported | AD matches the hand derivative to 1e-14 |
| `tutorialBasicObjects.m` | 277 | `pymrst/examples/tutorial_basic_objects.py` | ported | BedModel2 section optional, dataset absent |
| `tutorialPlotting.m` | 180 | `pymrst/examples/tutorial_plotting.py` | ported | plotting optional, computation checked |
| `grids/gridTutorialIntro.m` | 231 | `pymrst/examples/grids/grid_tutorial_intro.py` | ported | 7 factory routines, connectivity asserted |
| `grids/gridTutorialStruct.m` | 81 | `pymrst/examples/grids/grid_tutorial_struct.py` | ported | boundary preserved under node perturbation |
| `grids/gridTutorialUnstruct.m` | 89 | `pymrst/examples/grids/grid_tutorial_unstruct.py` | ported | `seamount` substituted, see note |
| `grids/gridTutorialCornerPoint.m` | 279 | `pymrst/examples/grids/grid_tutorial_corner_point.py` | ported | fault splits 60 blocks into 64 |
| `datasets/showCaseB4.m` | 71 | `pymrst/examples/datasets/show_case_b4.py` | ported | skips until the 0.5 MB dataset is fetched |
| `datasets/showJohansen.m` | 172 | `pymrst/examples/datasets/show_johansen.py` | ported | 88775 active cells, porosity 0.100-0.291 |
| `datasets/showNorne.m` | 222 | `pymrst/examples/datasets/show_norne.py` | ported | 44927 active cells, 42532 fault faces |
| `datasets/showSAIGUP.m` | 299 | `pymrst/examples/datasets/show_saigup.py` | ported | skips until the 2.5 MB dataset is fetched |
| `datasets/showSPE10.m` | 111 | `pymrst/examples/datasets/show_spe10.py` | ported | 1122000 cells, 7.5 decades of permeability |
| `data/Egg/.../Reading_deck_Egg_model_Fully_Implicit.m` | 110 | `pymrst/examples/data/egg/reading_deck_egg_model_fully_implicit.py` | ported | 18553 active cells verified |
| `data/Egg/.../Reading_deck_Egg_model_Sequential.m` | 201 | `pymrst/examples/data/egg/reading_deck_egg_model_sequential.py` | ported | 8 injectors placed and verified |
| `data/Egg/.../StoreProduction.m` | 39 | `pymrst/examples/data/egg/store_production.py` | ported | fractional-flow rate split |

## Blocking dependencies

Calls in these examples were cross-checked against MRST's function inventory and
against PyMRST's modules and definitions. Eight MRST functions the examples need
have no PyMRST counterpart under any spelling:

| MRST function | Lines | MRST module | Needed by | Status |
|---|---|---|---|---|
| `processWells` | 804 | `deckformat/params/wells_and_bc` | Egg examples | **already ported** |
| `initEclipseRock` | 336 | `deckformat/params/rock` | Egg examples | **ported** |
| `initEclipseGrid` | 236 | `deckformat/grid` | Egg examples | **ported** |
| `getSPE10rock` | 134 | `spe10` | `showSPE10` | **ported** |
| `processPartition` | 117 | `coarsegrid` | `showSAIGUP`, **corner-point** | **ported** |
| `partitionLayers` | 100 | `coarsegrid` | `showSAIGUP`, **corner-point** | **ported** |
| `explosionView` | 102 | `coarsegrid/plotting` | `showSAIGUP`, **corner-point** | **ported** |
| `compressRock` | 56 | `deckformat/params/rock` | Egg, **`tutorialBasicObjects`** | **ported** |

That was 1885 lines of dependency against 2920 lines of example, so the
dependencies are close to two thirds of the visible work. Three are now ported,
`compressRock`, `initEclipseGrid` and `initEclipseRock`, together with
`mapAxes` (48 lines) which `initEclipseGrid` needs and which the original scan
missed entirely. `processWells` turned out to be ported already, and the three
`coarsegrid` entries are now ported too, as part of a full port of that module
rather than an extraction of the three functions listed here. **No dependency
remains outstanding.**

**Two attributions in the original scan were wrong**, both in the same
direction: the scan found which functions are missing but not which files need
them.

* `gridTutorialCornerPoint` does **not** call `initEclipseGrid` or
  `initEclipseRock` at all. It needs `processPartition`, `partitionLayers`,
  `explosionView`, `generateCoarseGrid`, `outlineCoarseGrid` and
  `plotBlockAndNeighbors`, so it is blocked by `coarsegrid`, the same module
  that blocks `showSAIGUP`. Porting the two deck functions did not unblock it.
* `tutorialBasicObjects` has a closing section the first pass missed. MRST
  names `initEclipseGrid` and `processWells` in a comment there, which the scan
  attributed to the Egg examples; the code below that comment calls
  `initEclipseProblemAD`, a different function that PyMRST already had.
* **`processWells` was never missing.** PyMRST carries the whole pipeline in
  `modules/compositional/facility.py`, split into
  `wells_from_deck_completion_records` for `WELSPECS`/`COMPDAT` and
  `apply_deck_well_controls` for `WCONPROD`, `WCONINJE`, `WCONINJH`,
  `WCONHIST`, `WELOPEN` and `WELTARG`. Verified by execution rather than by
  name: a producer at deck `(5, 5)` over layers 1 to 3 in a 5x5x3 grid lands on
  cells 24, 49 and 74, well indices are positive, and producer/injector signs
  and controls come through. The scan missed it because the port lives under a
  different name in a different module. At 804 lines this was the single
  largest item in the plan.

A thin `modules/deckformat/params/wells_and_bc/process_wells.py` now delegates
to that implementation, so the MRST-shaped path exists without a second copy of
the logic. One of its self-tests asserts the wrapper and the direct calls agree,
so it cannot drift into being a separate implementation.

`compressRock` moved into Phase 1 during implementation. The original plan
listed it as an Egg dependency only, but `tutorialBasicObjects` calls it twice,
so the claim that the root tutorials had no blockers was wrong. Static call
analysis found the function; it did not find every caller. The correction is
recorded in `examples_implementation.md`.

**`coarsegrid` was absent from PyMRST entirely.** It owned three of the eight.
It is now ported in full, 34 of 37 files with 3 recorded exclusions; see
`coarsegrid_parity.md`. `showSAIGUP` turned out not to need it at all — that
attribution belonged to `modules/coarsegrid/examples/coarsegridExampleSAIGUP.m`,
a different file with a similar name — and waits only on its dataset.

Present and usable, despite an earlier automated scan reporting otherwise:
`process_grdecl`, `read_grdecl`, `grdecl2rock`, `incomp_tpfa`,
`init_variables_adi`, `init_deck_adi_fluid`, and `make_norne_grdecl`. The scan
missed them because naive camel-to-snake conversion mangles acronyms, turning
`processGRDECL` into `process_g_r_d_e_c_l`. Any future audit script for this
tree needs acronym-aware name matching, or it will over-report the gap.

## Dataset availability

The examples read real field datasets. Descriptors exist in PyMRST for all of
them under `pymrst/utils/dataset_manager/datasets/`, so the download machinery is
in place, but the payloads differ in whether they are present locally.

| Dataset | Local size | Status |
|---|---|---|
| Johansen | 233 MB | present |
| Egg | 166 MB | present |
| SPE10 | 106 MB | present |
| Norne | 18 MB | present |
| SAIGUP | absent | download required |
| CaseB4 | absent | download required |

## Validation approach

These are tutorials, so most produce figures rather than numbers, and figure
comparison is not parity evidence. Validation should target the computed
quantities each tutorial passes to its plotting calls: grid geometry, rock
properties, pressure and flux fields, and well solutions. The existing
`dump_*_reference.m` and `validate_*.py` pattern applies unchanged.

Three examples carry genuine numerical content and deserve MATLAB validators:
`flowSolverTutorial1` (gravity column pressure), `flowSolverTutorialAD`
(compressible single-phase history), and both Egg deck runs. The four grid
tutorials validate against grid structure checksums, which is the pattern already
used for the MATLAB file bridge, where cell and face counts, volumes, centroids,
areas and permeability matched to a relative 5.6e-14.

## Completion declaration rule

Do not mark this tree complete until all 17 executable examples have Python
owners at MRST-shaped paths, the eight blocking dependencies are ported or
formally excluded, `coarsegrid` exists or `showSAIGUP` is excluded with a stated
reason, the two missing datasets are downloadable through the existing dataset
manager, every example runs end to end in the standard test sweep, and the
numerically meaningful examples pass MATLAB validators.

Against that rule: all 17 examples have owners, all eight dependencies are
ported, and `coarsegrid` exists in full. What remains is the two datasets.
`showSAIGUP` is the only example that cannot run end to end here, and it waits
on a 2.5 MB download rather than on any code.

## Known gap surfaced by the port

`init_eclipse_problem_ad` accepts only PyMRST's SPE1-style parsed-deck subset,
carrying `x`/`y`/`z`/`poro`/`perm`, and raises `NotImplementedError` on a full
ECLIPSE deck unless the caller supplies a ready-made `model=`. The SPE9 deck
reads and converts correctly; construction is what fails.

`tutorial_basic_objects.run_eclipse_case_section` therefore returns a reason
rather than a result, and its self-test asserts that reason. Reporting the limit
keeps it visible and makes the test flip the day full-deck support arrives,
where returning a bare `None` would have made a real capability gap look like a
missing dataset.

## Deck-reader defects found while porting

**Comma separators (fixed).** ECLIPSE treats a comma as a separator, and the
Egg model writes `0.1000,  0.0000e+00,  8.0e-01  0` in `SWOF`. The tokenizer
matched bare `\S+`, gluing the comma onto the number, so the whole deck failed
on `float('0.1000,')`. Fixed centrally in
`utils/inout/eclipse/deckinput_simple/read_defaulted_record.py` by excluding
commas from unquoted tokens; quoted strings keep theirs because the quoted
alternatives are tried first. Regression test added, no regressions across
`deckformat` or `inout`.

**No GRID-section reader (fixed).** The Egg `GRID` section returned only
`cartDims`, and INCLUDE was not the cause. `read_eclipse_deck` lists `GRID`
among its `SKIPPED_SECTIONS` by design, and `read_grdecl` asserts
`SPECGRID`/`COORD`/`ZCORN`, which a block-centred deck such as Egg does not
carry. PyMRST therefore had no path at all to a block-centred GRID section.

Added `modules/deckformat/deckinput/read_grid.py`, a port of MRST's `readGRID`
scoped to the keywords real decks use. It splices INCLUDE files and parses the
combined text once, which matters because Egg declares `SPECGRID` in the parent
deck and applies `COPY`/`MULTIPLY` inside an include: parsing the files
separately leaves the operator without grid dimensions and it fails.
`ARRAY_KEYWORDS` gained the block-centred group (`DX`, `DY`, `DZ`, `DXV`,
`DYV`, `DZV`, `TOPS`, `DEPTHZ`) alongside the corner-point one. A companion
`grid_section_coverage` reports arrays, operators, ignored and **unrecognised**
keywords, so a skipped keyword is visible rather than silent.

Egg now reads end to end: `cartDims` (60, 60, 7), every array present,
`PERMZ/PERMX` exactly 0.1 from the deck's `MULTIPLY`, 18553 active cells of
25200, and **zero unrecognised keywords**. `init_eclipse_grid` builds an
18553-cell grid from it and `init_eclipse_rock` the matching rock.

**`COPY` ran backwards (fixed).** ECLIPSE writes `COPY source target`; MRST
spells the direction out in its own implementation as `y <- x` with
`x = rec{1}`. `apply_operator_simple` took `record[0]` as the target for every
operator, which is right for `ADD`, `MULTIPLY` and `EQUALS` but reversed for
`COPY`, so `COPY 'PERMX' 'PERMY'` would overwrite PERMX from PERMY. It raises
when the target does not exist, which is the lucky case; where both arrays
exist it copies the wrong way in silence. The existing self-test passed because
it encoded the same reversal as the code. Both are corrected, with a regression
test that fails on either convention alone.

## Egg examples: what reading a real deck cost

The three Egg files are 350 lines of example resting on four defects and two
gaps, none of which a static dependency scan could have predicted. In order of
discovery:

1. **Comma separators** in the tokenizer, so the deck would not parse at all.
2. **No GRID-section reader** for a block-centred deck.
3. **`COPY` reversed**, so `COPY 'PERMX' 'PERMY'` wrote the wrong array.
4. **ECLIPSE box operators ignored** entirely, found earlier in the same effort.
5. **`init_corey_fluid` lacked `sr` and `kwm`**, MRST's residual saturations and
   end-point relative permeabilities. Without them a fluid stays mobile down to
   zero saturation. Added with defaults that reproduce the previous behaviour
   exactly, so no existing caller changes, and checked term by term against
   MRST's `modified_saturations`.
6. **`processWells` was already ported** under another name, which is a
   bookkeeping error rather than a defect but cost the largest single line
   estimate in the plan.

Three of the six are silent-wrongness defects: they produce a plausible model
rather than an error. All were found by pointing the port at a real field deck,
not by reading code, which is the strongest argument for doing that earlier and
more often.

The examples assert the Egg model's published geometry, 18553 active cells of
25200, rather than merely that a grid was built. An ACTNUM read in the wrong
order still yields a perfectly reasonable grid.

## Dataset examples: what each one verifies

These read production models, so each asserts the model's published
characteristics rather than merely that something loaded. A grid built from a
mis-read ACTNUM, or a field sampled without its index map, is perfectly
plausible until compared against a known number.

| Example | Verified against |
|---|---|
| `show_norne` | 44927 active of 104970 cells; 42532 fault faces; 64 degenerate cells, all inactive |
| `show_johansen` | NPD5 at (100, 100, 11); 88775 active; porosity 0.100 to 0.291, inside MRST's own colour range |
| `show_spe10` | 1122000 cells; 7.5 decades of permeability; Kz 2.8 decades below Kx; Tarbert and Upper Ness distinguishable |
| `show_case_b4` | skipped until the dataset is fetched; asserts that loading never triggers a download |

Two further reader defects surfaced here, both in the `read_keywords`
tokenizer and both fixed with regressions:

* **INCLUDE not followed.** `NORNE.GRDECL` spreads its arrays across fourteen
  include files, so the box operators lost the `SPECGRID` they need. The
  `read_grid` written for Egg solves this, which is the same gap appearing in a
  second dataset. `read_grdecl` should probably delegate to it, but cannot:
  `read_grid` lives in `modules` and the import-health test asserts that
  `utils` never imports `modules`.
* **Trailing commentary after a record slash.** Norne writes
  `'MULTZ' 0.005 6 14 11 18 10 10 /  C-3H`, where `C-3H` names the fault with
  no comment marker. It leaked into the next record, which then had one field
  too many and was rejected. The main deck tokenizer already handled this and
  even cites Norne in its comment; this reader has its own simpler tokenizer
  that did not.

One test was deliberately weakened, and the reason is worth keeping. Asserting
that all Norne cell volumes are positive fails at -531828 m3. Checking where
showed 64 degenerate cells, every one inactive and no active cell affected.
That is precisely why MRST passes `'checkgrid', false` here: forcing all cells
active to keep the inactive region visible also admits the degenerate geometry
that region contains. The test now asserts the real invariant and still fails
if a degenerate cell ever appears among the active ones.


## A correction: showSAIGUP never needed coarsegrid

The dependency scan attributed `processPartition`, `partitionLayers` and
`explosionView` to `datasets/showSAIGUP.m`. It uses none of them. Its complete
set of MRST calls is `readGRDECL`, `convertInputUnits`, `getUnitSystem`,
`processGRDECL`, `grdecl2Rock`, `computeGeometry`, `convertTo`,
`getDatasetPath` and four plotting routines, **every one of which was already
ported**. The only thing blocking it was the missing dataset.

The partitioning belongs to `modules/coarsegrid/examples/coarsegridExampleSAIGUP.m`,
a different file with a similar name in a different tree. The two were
conflated.

This was the fifth attribution error in this port, and the pattern held every
time: the scan reliably found which *functions* were absent and was unreliable
about which *files* needed them. It also meant the `coarsegrid` decision was
presented as unblocking two examples when it unblocked one. That work stands on
its own merits, having produced a real module and uncovered three duplicate
`CoarseGrid` implementations, but the premise given for it was partly wrong.

`show_saigup` carries a test that parses its own imports and fails if a
`coarsegrid` dependency is ever added, so the correction cannot quietly reverse.
