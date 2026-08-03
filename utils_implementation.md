# Finishing `pymrst/utils`

What it would take for `pymrst/utils` to match MRST's `utils/` end to end, and
in what order.

`UTILS_PARITY.md` is the ledger this plan retires; `scripts/audit_utils_parity.py`
regenerates it. `docs/utils/parity.csv` is authoritative.

## The scope question comes first, because the raw number is misleading

MRST's `utils/` holds **469 MATLAB files**. Counting all of them would produce
a parity figure that means nothing: PyMRST is never going to port a vendored
Boost Graph Library binding, MATLAB's module-path machinery, or a bibliography
of MRST papers.

**309 files are excluded**, by rule, with the reason recorded per rule in the
ledger rather than applied as a silent filter:

| Exclusion | Files | Why |
|---|---:|---|
| `3rdparty/matlab_bgl` | 122 | vendored Boost Graph Library MEX bindings; Python has `scipy.sparse.csgraph` and `networkx` |
| `dataset_manager` | 59 | dataset download registry and GUI |
| `paper_manager` | 44 | bibliography of MRST publications |
| `octave_only` | 16 | Octave compatibility shims |
| `mex` | 6 | MEX compilation helpers |
| `settings_manager` | 3 | MATLAB settings persistence |
| `Contents.m` | 16 | package catalogs |
| MATLAB environment | 43 | `mrstModule`, `mrstPath`, `merge_options`, `dispif`, `ROOTDIR` and the rest of the plumbing |

The exclusion worth arguing about is **`merge_options`**. Every MRST function
takes `varargin` and parses it with `merge_options`; Python has keyword
arguments, so no PyMRST function needs it, and porting it would invite a
calling convention the language already provides. That is a deliberate
divergence, not an oversight, and it is the reason PyMRST signatures look
different from MRST's throughout.

**That leaves 160 files in scope.**

## Where it stood when this plan was written

**These are the starting numbers, kept as the baseline the phases are measured
against - they are not current.** For where it stands now, read
`UTILS_PARITY.md`, which is regenerated from the ledger, or the per-phase
status sections below. At the time of writing this section: 2 of 160. As of the
most recent phase: **87 of 156**, with `ported-elsewhere` at zero.

| | count |
|---|---:|
| At an MRST-shaped path under `pymrst/utils` | 2 |
| Ported, but somewhere else | 13 |
| Consolidated into another file | 54 |
| Not implemented | 91 |

Two of 160 is the honest headline, and it is also the least useful number
here. **69 of 160 exist in some form.** The gap is dominated by two things:
work genuinely not done (91), and work done but filed where MRST's layout does
not point (13).

| Subsystem | In scope | Ported | Elsewhere | Consolidated | Missing |
|---|---:|---:|---:|---:|---:|
| `(root)` | 29 | 1 | 8 | 7 | 13 |
| `units` | 51 | 0 | 0 | 42 | 9 |
| `gridtools` | 39 | 1 | 2 | 3 | 33 |
| `inout` | 24 | 0 | 3 | 2 | 19 |
| `tensorAssembly` | 14 | 0 | 0 | 0 | 14 |
| `equil` | 2 | 0 | 0 | 0 | 2 |
| `ad_utils` | 1 | 0 | 0 | 0 | 1 |

## Phase 1: put what exists where MRST points

**Six files, no new numerics.**

Four PyMRST files sit in `utils/gridtools/` that MRST keeps at `utils/` root:

| MRST | PyMRST now | should be |
|---|---|---|
| `utils/rldecode.m` | `utils/gridtools/rldecode.py` | `utils/rldecode.py` |
| `utils/rlencode.m` | `utils/gridtools/rlencode.py` | `utils/rlencode.py` |
| `utils/mcolon.m` | `utils/gridtools/mcolon.py` | `utils/mcolon.py` |
| `utils/blockDiagIndex.m` | `utils/gridtools/block_diag_index.py` | `utils/block_diag_index.py` |

The last one is mine, added during `vem` Phase 4 - I put it beside `rldecode`
without checking that `rldecode` was itself misplaced. MRST's
`utils/gridtools/` is for *grid* queries (`gridCellNo`, `findEnclosingCell`);
run-length encoding and index arithmetic are general.

Two more need a decision rather than a move:

- **`value.m`** is at MRST's `utils/` root; PyMRST has it at
  `utils/adi/value.py`. It is genuinely AD-specific - it strips derivatives -
  so the current placement reads better than MRST's. Either move it or record
  it as a reviewed deviation; do not leave it unexplained.
- **`ADI.m`** is one MATLAB classdef; PyMRST has a package `utils/adi/` with
  the class, its operators and `init_variables_adi`. A classdef with dozens of
  methods becoming a package is reasonable. Record it as reviewed.

**Acceptance**: `utils/__init__.py` and `utils/gridtools/__init__.py` re-export
everything, so no call site changes; the full suite passes unchanged.

**Size.** Half a day.

### Phase 1 status: complete

All four files moved to `pymrst/utils/`, with 31 files' imports rewritten.
`utils/gridtools/__init__.py` still re-exports them so nothing external broke,
and its docstring says where they went and why. The ledger moved from **2 to 9
of 160** at MRST-shaped paths, and `ported-elsewhere` from 13 to 6.

`value`, `numelValue`, `ADI` and `initVariablesADI` are recorded as reviewed
deviations rather than moved: MRST keeps all four at `utils/` root, but each is
AD-specific, and a MATLAB classdef with dozens of methods reasonably becomes
the `utils/adi/` package. The reasons are in the auditor, so they appear in the
ledger rather than living only here.

**Six entries remain `ported-elsewhere`, and each needs a decision this phase
did not make:**

| MRST | PyMRST | Decide in |
|---|---|---|
| `utils/computeWellIndex.m` | `params/wells_and_bc/compute_well_index.py` | phase 2 |
| `utils/gridtools/translateGrid.m` | `gridprocessing/translate_grid.py` | phase 4 |
| `utils/gridtools/computeGridSlicePolygons.m` | `gridprocessing/compute_grid_slice_polygons.py` | phase 4 |
| `deckinput_simple/cutGrdecl.m` | `gridprocessing/cut_grdecl.py` | phase 5 |
| `deckinput_simple/readGRDECL.m` | `deckformat/grdecl/read_grdecl.py` | phase 5 |
| `deckinput_simple/readDefaultedRecord.m` | `deckformat/deckinput/read_defaulted_record.py` | phase 5 |

MRST keeps the first three **only** in `utils/` - they are not duplicated in
its `gridprocessing` module - so on the layout contract they should move. The
last three are the `deckinput_simple` duplication question, which Phase 5
settles.

**Regression**: vem 61, gridprocessing 250, ad_props 72, deckformat 39,
vemmech 92 pass. The two failures in `ad_core` and `ad_blackoil` are the
pre-existing parity-ledger tests, unchanged by this work.

## Phase 2: the core numerics at `utils/` root

**13 files, ~700 lines.**

The ones that carry weight:

| File | What it does |
|---|---|
| `dinterpTable`, `dinterpq1` | the *derivatives* of `interpTable`, which PyMRST has without them |
| `geomspace`, `discspace1` | graded spacings for near-well refinement |
| `subsetPlus`, `subsetMinus` | subset arithmetic on index sets |
| `getSortedCellNodes` | cyclically sorted cell nodes - a prerequisite for several `gridtools` entries |
| `cellDimsCG`, `isCoarseGrid` | coarse-grid queries |
| `incompHydrostaticPressure` | hydrostatic initialisation |
| `numelValue`, `reduceToDouble` | AD-aware size and value reduction |
| `formatTimeRange` | human-readable durations |

**The three flux converters are already done**, all inside
`utils/flux_conversion.py`: `cellFlux2faceFlux`, `faceFlux2cellFlux` and
`faceFlux2cellVelocity`. Splitting that file into MRST's three is a Phase 1
style move, not new work.

An earlier draft of this plan listed all three as missing, because the auditor
did not know MRST writes "to" as a digit - `faceFlux2cellFlux` against
`face_flux_to_cell_flux`. The auditor now matches both spellings, which moved
three entries from *missing* to *consolidated*. Worth remembering when reading
any of these ledgers: a name-matching scan understates what exists, and the
direction of that error is always the flattering one for the work remaining.

**Where the risk is.** `getSortedCellNodes` is a dependency of the `gridtools`
phase and of `computeVEMGeometry`, which PyMRST already implements by its own
route. Porting it means checking the two agree, not just that the new one
works.

**Size.** 3–4 days.

### Phase 2 status: complete

**`(root)` is now 23 ported, 0 elsewhere, 0 missing**, and the ledger reads
**24 of 160** overall. Thirteen new files, each with its own tests:
`get_sorted_cell_nodes` (with `sort_edges`), `dinterpq1`, `dinterp_table`,
`geomspace`, `discspace1`, `subset_arithmetic`, `numel_value`,
`reduce_to_double`, `is_coarse_grid`, `cell_dims_cg`, `format_time_range`,
`incomp_hydrostatic_pressure`.

`flux_conversion.py` was split into MRST's three files, with the shared
cell-face indexing in a private `_cell_face_signs.py` rather than triplicated.
`compute_well_index` moved from `params/wells_and_bc/` to `utils/`: MRST does
have a `params/wells_and_bc/` directory, but it holds `addWell`, `addBC` and
`addSource` - not `computeWellIndex`, which is in `utils/`.

**A Python problem MRST does not have, which broke the whole suite for a
while.** A module and a function cannot share a name safely: writing
`from pymrst.utils.mcolon import mcolon` anywhere rebinds the *package
attribute* `pymrst.utils.mcolon` to the module, so a later
`from pymrst.utils import mcolon` yields a module and fails with
`'module' object is not callable`. Phase 1's import rewrite had collapsed call
sites to the package form, and 119 gridprocessing tests failed the moment
Phase 2 added more same-named modules.

Every call site now imports through the submodule explicitly. The package
`__init__` still re-exports for convenience, but **anything whose module name
equals its function name must be imported as
`from pymrst.utils.<name> import <name>`**. That is the cost of following
MRST's one-function-per-file layout in Python, and it is worth knowing before
Phase 4 adds 33 more such files.

**Two things the tests caught that I had asserted wrongly:**

- I documented `dinterpq1` as *inconsistent* with `interp_table` outside the
  table - value clamped flat, slope extrapolated. It is not: `interp_table`
  extrapolates linearly, so the two agree. The test now asserts the agreement.
  (`extend_tab` is the mechanism that flattens, and it is a different thing.)
- `reduceToDouble` is an **empty function** in MRST. The plausible guess - that
  it strips AD derivatives - is wrong and would break every caller that
  differentiates through it. It is a passthrough hook, and `value` is the
  function that strips derivatives.

**Regression**: gridprocessing 250, ad_props 72, deckformat 39, vem 61,
vemmech 92 pass; all 30 `validate_phase4xx` and the flux validator pass.
`validate_phase170` fails on the pre-existing `compositional.model` circular
import, which `import pymrst.modules.compositional.model` reproduces on its
own.

## Phase 3: units

**9 files, and one decision.**

42 of MRST's 51 unit files are already present as constants in
`pymrst/utils/units/units.py`. Missing: `ampere`, `farad`, `gal`, `site`,
`avogadro`, `faradayConstant`, `gasConstant`, `vacuumpermittivity`, and
`getUnitSystem`.

The first eight are one line each. `getUnitSystem` returns a named unit system
and is the one with behaviour; `deckformat`'s `unit_conversion_factors.py`
already does the equivalent job for decks, so port `getUnitSystem` only after
checking whether the two should be one function.

**The decision: do not split `units.py` into 51 files.** MRST has one file per
unit because MATLAB resolves bare `darcy` as a function call. Python does not
need that, and 51 single-constant modules would be worse to read and to
import. This is the one place in this plan where MRST's layout should be
declined outright, and the ledger records the 42 as `consolidated` rather than
`ported` so the divergence stays visible.

**Size.** Half a day.

### Phase 3 status: complete

**`units` is now 0 missing.** Eight constants added to `units.py` -
`AMPERE`, `FARAD`, `GAL`, `AVOGADRO`, `GAS_CONSTANT`, `FARADAY_CONSTANT`,
`VACUUM_PERMITTIVITY`, `SITE` - plus `units/get_unit_system.py` at MRST's path.
The ledger reads **25 of 160**, with `units` at 1 ported and 50 consolidated,
which is the intended divergence.

**The `getUnitSystem` decision: keep the two tables separate.** It carries
seven quantities and names four systems; `deckformat`'s
`unit_conversion_factors` carries 25, handles temperature offsets, and converts
*between* systems for deck reading. Merging would give deck reading a
vocabulary it does not need. They do overlap, so a test compares the four
shared fields across all three non-SI systems - and they agree, which is now
checked rather than assumed.

**Three defects in MRST's own unit files, found while porting:**

- **`avogadro.m` cannot run.** It declares output `T` and assigns `An`, so
  MATLAB raises "Output argument T is not assigned". It is also dead: nothing
  in the MRST tree references it except its own `Contents.m`.
- **MRST has two values of Avogadro's number.** `avogadro.m` says
  6.022140857e23; `site.m` divides by 6.00221413e23. They differ in the sixth
  digit, so `site * avogadro` is 1.0000000095 rather than 1. This port derives
  `SITE` from the single `AVOGADRO`, and a test asserts the product is exactly
  1.
- **`gal` is an acceleration, not a volume.** 1 cm/s^2, and MRST also has
  `gallon`. The one-word filenames make them easy to swap, which is off by
  five orders of magnitude; a test states the distinction.

The physical constants are checked against each other rather than only
transcribed: `R = k_B * N_A` holds to 1e-6, which is an independent check on
two of them.

## Phase 4: `gridtools`

**33 files, ~2,000 lines. The largest phase.**

Grid queries and manipulations. The natural groups:

- **connectivity**: `gridCellNo`, `gridCellFaces`, `gridCellNodes`,
  `gridFaceNodes`, `getCellNoFaces`, `getConnectivityMatrix`,
  `getNeighbourship`, `neighboursByNodes`, `connectedCells`
- **queries**: `findEnclosingCell`, `getFacesCloseToSegment`,
  `gridLogicalIndices`, `indirectionSub`, `sampleFromBox`
- **validation**: `checkGrid`, `compareGrids`
- **construction and repair**: `createGridMappings`, `createAugmentedGrid`,
  `addBoundingBoxFields`, `sortGrid`, `sortEdges`, `removeNodes`,
  `makePlanarGrid`, `transform3Dto2Dgrid`
- **horizons and layers**: `convertHorizonsToGrid`, `sortHorizons`,
  `makeLayeredHorizonGrid`, `refineGrdeclLayers`,
  `computeVerticalGridIntersection`, `volumeByGaussGreens`

Three already exist as symbols inside other files (`createAugmentedGrid` in
`vemmech/vem.py`, `removeNodes` in `gridprocessing/pebi.py`, `sortEdges` in
`gridprocessing/make_layered_grid.py`). Those are *not* free wins: each is a
private helper shaped for its caller, and promoting it to a general utility
means checking it handles cases its caller never produced.

**Start with the connectivity group.** It is nine small functions that
everything else in the phase uses, and `gridCellNo` alone appears in most MRST
grid code.

**Size.** 5–7 days.

### Phase 4 status: complete on substance - 30 of 34 ported; the 4 left are MRST's test scaffolding

Nine files, each with its own tests: `grid_cell_no`, `get_cell_no_faces`,
`grid_cell_faces`, `grid_cell_nodes`, `grid_face_nodes`, `get_neighbourship`,
`get_connectivity_matrix`, `neighbours_by_nodes`, `connected_cells`. The
ledger reads **34 of 160**, with `gridtools` at 10 ported and 24 missing.

This was the right place to start: everything else in the phase is expressed in
terms of these, and `grid_cell_no` alone appears throughout MRST's grid code.

**`connectedCells` answered the plan's SciPy question.** MRST reaches it
through `dmperm`, MATLAB's Dulmage-Mendelsohn decomposition - a general tool
used for a special case, and one that lives in the vendored Boost binding this
plan excludes. `scipy.sparse.csgraph.connected_components` answers the actual
question directly, so that is what the port calls. Groups are returned sorted
by smallest member, because `connected_components` promises no order and a
caller comparing two runs would otherwise see spurious differences.

**Two conventions that will trip a reader porting from MATLAB**, both now
tested rather than assumed:

- **Outside is -1, not 0.** `get_neighbourship(include_boundary=True)` marks
  the outer side -1 where MRST marks it 0 - and 0 is a real cell index here.
- **A subset renumbers positionally.** `grid_cell_no(g, [2, 4])` numbers cell 4
  as *1*, not 4, because the result is meant to index a subset-sized array.
  Using it as a global cell index is the mistake the interface invites.

**`grid_cell_nodes` is not `get_sorted_cell_nodes`**, and a test asserts they
disagree. The former returns each cell's nodes sorted by index; the latter
returns them in cyclic boundary order. Using the sorted one where the cyclic
one is meant gives a polygon with crossed edges and an area that is too small.

**Three more since**: `grid_logical_indices`, `indirection_sub` and
`check_grid`. The ledger reads **37 of 160**, `gridtools` at 13 ported and 21
missing.

`grid_logical_indices` is the one with a trap. It maps through
`cells.indexMap`, not the cell number, and the difference only shows on a grid
with inactive cells removed - which a test built on `cart_grid` cannot reach.
Its test removes three cells first and asserts the naive answer differs.

**`check_grid` is an auditing tool, so it was pointed at the repository.** Run
over every grid type PyMRST can build - Cartesian 2-D and 3-D, twisted,
triangle, PEBI, tetrahedral - it reports:

| grid | result |
|---|---|
| cart 2-D, cart 3-D, twisted, triangle, tetra | clean |
| **pebi 2-D** | **36 of 348 faces have negligible area** |

The PEBI finding is real and worth knowing: generating points on a lattice are
cocircular, so their Voronoi vertices coincide and the construction emits
zero-area faces - areas around 1e-14 against a median of 83. It is the same
root cause as the sliver cells `vem_example_2` documents, seen from the face
side rather than the cell side.

**An earlier version of this check called those faces misoriented**, because it
compared a dot product against zero and a zero-area face has no meaningful
direction. The orientation of PyMRST's PEBI normals is in fact correct. The
check is now relative to the face area and centroid separation, and reports
degeneracy as degeneracy - which is both true and more useful, since a caller
can act on it.

**Two more since**: `find_enclosing_cell` and `compare_grids`. The ledger
reads **39 of 160**, `gridtools` at 15 ported and 19 missing.

`find_enclosing_cell` **assumes convex cells**, and MRST does not say so. A
point is placed by testing it against every face plane, which coincides with
"inside the cell" only for a convex cell; a re-entrant corner gives a wrong
answer rather than an approximate one. Stated in the module, because every
grid PyMRST builds happens to be convex cell by cell and the assumption is
therefore invisible until it is not.

`compare_grids` **did not compare numbering**, and could not until `sortGrid`
was ported: two grids may describe the same geometry with cells in a different
order, and a field-by-field comparison then reports differences everywhere.
That limit was in its docstring rather than discovered later, and `sort_grid`
has since lifted it - see below.

**One more since**: `sort_grid`. The ledger reads **40 of 160**, `gridtools` at
16 ported and 18 missing. It is validated against MATLAB
(`scripts/validate_sort_grid.py`), and every ordering field - node coordinates,
`faces.nodes`, `faces.nodePos`, `cells.faces`, `cells.facePos` - agrees with
MRST's `sortGrid` exactly on four Cartesian grids in 2-D and 3-D.

`compare_grids` **now takes `sort=True`**, which puts both grids into canonical
order first and so answers "are these the same grid?" rather than "are these
the same arrays?". Off by default, because sorting is much the more expensive
of the two and because a grid whose numbering *should* have survived a round
trip is exactly the case where you want to be told that it did not.

**Two defects in MRST's `sortGrid`, found by porting it.** Neither is
speculative; both are pinned by tests that fail if the fix is reverted.

- **In 2-D the sort reverses faces and MRST does not notice.** The algorithm
  rotates each face's node list to start at its lowest node. A 2-D face has
  exactly two nodes, so that "rotation" is a *reversal* half the time - 16 of
  the 4×3 grid's faces - which flips the face normal and breaks the invariant
  that a normal runs from `neighbors[0]` to `neighbors[1]`. MRST gets away with
  it because it leaves the derived geometry in place and stale, so no normal is
  ever recomputed to contradict it. `sort_grid` swaps the neighbour columns of
  exactly the reversed faces, and drops the stale geometry rather than keeping
  it. The MATLAB validator asserts on this divergence rather than hiding it:
  in 3-D `faces.neighbors` matches MRST entry for entry, and in 2-D it matches
  only as an unordered pair, with the swap count checked to be non-zero.
- **The face-tag column is left mismatched.** Where `cells.faces` has a second
  column naming each face's direction, MRST rotates that column but then writes
  the *sorted* face numbers beside it. Its own source closes `sortCells` with
  the comment `% cellFaces(:,2) buggy`. Here the tags are permuted by the same
  index as the faces. PyMRST's own builders emit no tag column, so this matters
  only for a grid read back from MATLAB - which is precisely the case the
  function exists to serve.

One deliberate match rather than divergence: MATLAB pads the node lists into a
rectangular array with `NaN`, and `sortrows` puts `NaN` last, so where one node
list is a prefix of another the **longer** one sorts first. Python tuples order
the shorter one first. That would give a different canonical form on any grid
whose faces differ in node count, so `_order` pads with infinity to reproduce
MATLAB's answer.

**The auditor was undercounting validation.** It scanned only `tests/` and
`examples/` for evidence, but this project's MATLAB-reference validators live
in `scripts/validate_*.py` and are the strongest evidence available. It now
scans those too - restricted to that glob, since the auditors themselves name
every symbol they classify and would otherwise mark the whole tree validated.
14 of the 45 ported files now carry reference validation.

**One more since**: `add_bounding_box_fields`, also validated against MATLAB
(`scripts/validate_add_bounding_box_fields.py`). The ledger reads **41 of
160**, `gridtools` at 17 ported and 17 missing.

**Its field is not a bounding box.** `faces.bbox` holds `max - min` of the
face's node coordinates - the box's *size*, not the box. MRST's only consumer,
`getFacesCloseToSegment`, supplies the missing half from `faces.centroids` and
treats the face as spanning `centroid ± bbox/2`. A centroid is not a box
centre, so that reconstruction can fall short of the true box on a skewed face
- a triangle is enough to show it, and a test does. For a rejection test the
error has to point outward and this one does not always. The port therefore
takes `extent=False`, which stores `bboxMin` and `bboxMax` as well and makes an
overlap test exact; `bbox` is still written unchanged, so a later port of
`getFacesCloseToSegment` finds what it expects.

The validator compares boxes **as multisets, not row by row**: PyMRST and MRST
number faces differently on the same Cartesian grid, so an entry-wise
comparison would fail for a reason unrelated to the function. What the function
promises is invariant under renumbering, and that is what is checked.

**One more since**: `volume_by_gauss_greens`, validated against MATLAB
(`scripts/validate_volume_by_gauss_greens.py`) and agreeing **bit for bit** on
all three cases. The ledger reads **42 of 160**, `gridtools` at 18 ported and
16 missing.

It computes cell volumes a second way - by integrating over the cell boundary
rather than by splitting the cell into tetrahedra - which makes it a real check
on `compute_geometry` rather than a duplicate of it.

**It is exact only for planar faces**, and the tests now say so instead of
assuming otherwise. The boundary integral uses one centroid and one normal per
face, which is the true integral only if the face is flat. On a twisted
hexahedral grid - most real reservoir geometry - it disagrees with
`compute_geometry` by about **0.4% per cell**. MRST's own implementation shows
the same 9.4e-03 disagreement on the same grid, so this is a property of the
method, not of the port; the validator asserts that rather than tolerating it.

The **totals still match to machine precision**, and the reason is the useful
part: every interior face enters two cells with opposite signs, so its error
cancels between them. The error redistributes volume between neighbours rather
than losing any. A total-only check therefore proves nothing about the per-cell
volumes - and a total is exactly the check most likely to be reached for.

Two things fell out of writing the tests:

- **The `x`-column is blind to any face lying in the plane `x = 0`**, where the
  centroid's `x` multiplies the normal away. The first version of the
  broken-normal guard perturbed face 0 of a grid cornered at the origin, which
  is such a face, and the guard passed while detecting nothing. It now picks
  the face by its actual contribution, and a second test pins the blindness.
- **MRST computes an entire block-diagonal matrix product to read its
  diagonal.** The quantity is a segmented sum; the port writes it as one, which
  is `O(cell-faces)` instead of `O(cells × cell-faces)`.

**One more since**: `create_grid_mappings`, validated against MATLAB
(`scripts/validate_create_grid_mappings.py`). The ledger reads **43 of 160**,
`gridtools` at 19 ported and 15 missing.

It returns one row per (cell, node, half-face, face) incidence, and everything
`createAugmentedGrid` builds is a projection of it. The **half-face** column is
the part worth reading carefully: a half-face is a `cells.faces` *row index*,
not a face number, so face 7 seen from cell 3 and face 7 seen from cell 5 are
different half-faces. That is what makes the table useful for assembly, where a
face carries a different contribution from each side.

**This surfaced that `create_augmented_grid` is only partly implemented.**
PyMRST has it inside `pymrst/modules/vemmech/vem.py`, where it computes
`cells.nodes` and `cells.nodePos` by its own route and stops - MRST's also
builds `nodes.faces`/`nodes.facePos` and `nodes.cells`/`nodes.cellPos`, and
neither exists here. The ledger records the file as `consolidated`, which reads
as done; it is not. A test asserts the two agree on the part they do share, so
the gap is bounded rather than guessed at.

**Completing it is left as a decision, not made silently.** The vemmech version
is used and reference-validated across that whole module, and rewiring it to go
through this table would touch the most heavily exercised code in the
repository for no behaviour anyone has asked for. The honest state is: the
mapping now exists at MRST's path and is validated; the consumer still does not
use it.

The validator compares **numbering-independently in the faces**, because MRST
and PyMRST number the faces of the same `cartGrid` differently. Each row is
fingerprinted as `(cell, node, sorted nodes of the face)`, which names the same
incidence under either numbering, and the multisets must match. Node and cell
numbering *are* asserted to agree first, so the fingerprint means something -
the script fails rather than proceeding if they do not.

**One more since**: `sample_from_box`, validated against MATLAB
(`scripts/validate_sample_from_box.py`) on four Cartesian cases including a
cell subset and a lattice that does not match the grid. The ledger reads **44
of 160**, `gridtools` at 20 ported and 14 missing.

It samples a uniform Cartesian data array at cell centroids - the usual way to
populate a grid from a porosity or permeability field generated independently
of it. Sampling is **nearest-slab, not interpolated**, which is right for a
piecewise-constant property but means the result does not converge under
refinement of the input array.

**MRST's bounding box can be too small.** It takes the nodes of the cells with
the extreme *centroids*, which is not the grid's bounding box: on a distorted
grid a cell whose centroid is unremarkable can own the furthest-out node.
Centroids outside the resulting box map to a negative or over-large index,
which is an out-of-bounds error in MATLAB. The port uses every node of the
selected cells.

Constructing a witness took two attempts, and the reason is worth recording:
**the stretched node has to be interior.** Stretching a boundary node puts it
in a cell that is extreme in some *other* direction - a node on the top edge
belongs to a top-row cell, picked for its `y` centroid - so MRST's box happens
to include it and nothing goes wrong. With an interior node of a 3×3 grid
pulled to `x = 3.9`, MRST's box stops at `x = 3.0`.

The two boxes coincide on Cartesian grids, which is why the reference agrees:
the validator checks the sampling arithmetic - the `ceil` convention, the axis
order, the subset handling - and the divergence is exhibited by a test in the
module, since no reference can settle a case where MRST indexes out of bounds.

**Two more since**: `make_planar_grid` and `search_for_boundary_faces`, both
validated against MATLAB. The ledger reads **46 of 160**, `gridtools` at 22
ported and 12 missing.

`make_planar_grid` turns selected faces of a 3-D grid into a surface grid -
each face becomes a cell, each edge a face. The result has `griddim == 2` with
three-dimensional coordinates, so `Grid.spatial_dim` reports 3 and anything
assuming `griddim == coords.shape[1]` is wrong about it.

Two things about it are worth knowing before use:

- **The parent's nodes are carried over whole**, as in MRST. A surface built
  from nine faces of a 3×3×2 grid still carries all 48 of the parent's nodes,
  16 of them used. Node numbers stay comparable with the parent's, which is
  usually the point, but `nodes.num` is not the surface's node count.
- **A non-manifold edge is rejected rather than silently mis-resolved.** The
  neighbour construction gives each edge one cell per direction, which is right
  for two faces meeting at an edge and wrong for three. MRST overwrites and
  keeps whichever it visited last. Three faces on an edge is normal in a
  fracture network or at a fault, so this is reachable and a wrong answer is
  worse than a stop. Two faces with *opposite* winding are not non-manifold -
  they traverse the edge once each way - and a test pins that distinction.

`search_for_boundary_faces` is **not a duplicate of `boundary_face_indices`**,
though the two look alike enough to be folded together by mistake.
`boundary_face_indices` takes the two ranges *in the plane of the requested
side*, so which axes they mean depends on the side; this takes `i1`, `i2`, `i3`
as `I`, `J`, `K` always. A test asserts they agree unrestricted and diverge
under restriction, so the distinction is pinned rather than described.

Its trap is the depth convention: **`top` means `ZMin` and `bottom` means
`ZMax`**, because z increases downwards. Ranges are 0-based here and 1-based in
MRST, and an out-of-range index says so in the error, since passing a 1-based
index out of habit is the likely cause.

The validator includes a case that must come back **empty** - west faces all
have `I = 0`, so restricting to `I = 1` selects nothing. A validator comparing
only non-empty results would not catch an off-by-one in exactly the conversion
most likely to have one.

Both validators compare **geometrically rather than by index**, because MRST
and PyMRST number the faces of the same `cartGrid` differently: faces by
centroid for one, and edges, cells and cell-separations by node identity for
the other. `make_planar_grid`'s neighbour *columns* are deliberately not
compared - they encode winding direction, which follows from the parent's face
node order and is itself numbering-dependent; the unordered pair is the claim
that has to hold.

**One more since**: `get_faces_close_to_segment`, validated against MATLAB on
nine cases - all matching exactly. The ledger reads **47 of 160**, `gridtools`
at 23 ported and 11 missing. **The queries group is now complete.**

It is a **conservative** filter: it returns every face a segment could
plausibly touch and some it could not, to cut a grid's worth of faces down to a
handful before a real intersection test. Treating the result as "the faces the
segment crosses" is the way to misuse it. Its tests check the direction that
matters - no false negatives - by walking the segment cell by cell with
`find_enclosing_cell` and asserting every genuinely crossed face comes back.
On a diagonal through a 6×6×4 grid it returns 48 faces of 516, covering all 10
that are actually crossed.

Two things came out of the port:

- **Projecting a segment along its own direction collapses it to a point**, and
  MRST divides by zero there, gets `NaN`, and silently returns no faces - which
  reads as "nothing is nearby" rather than "the question was malformed". A
  vertical well with `projVec = [0 0 1]` is exactly how someone would hit it.
  The port raises, naming both the segment and the projection.
- **MRST's `radius` option does nothing.** It is declared in the options struct
  and never read. It is not implemented here, because accepting a parameter
  that silently has no effect is worse than not accepting it.

The `fac = 2/3` default is worth understanding rather than copying: `1/2` would
be exact for a Cartesian grid, and the extra third is slack for faces whose
centroid is not their bounding box's centre - the same approximation
`add_bounding_box_fields` documents. That is why the conservatism is a
heuristic and not a guarantee.

The validator compares face **sets** exactly rather than accepting a superset.
This is a deterministic geometric predicate, not a heuristic with room to
differ, so "a superset is fine" would be the wrong standard: it would pass a
port whose slack calculation was wrong in the safe direction.

**One more since**: `transform_3d_to_2d_grid`. The ledger reads **48 of 160**,
`gridtools` at 24 ported and 10 missing. **The construction and repair group is
now complete** apart from `createAugmentedGridTEDFM`.

It collapses a grid that is a single cell thick in x, y or z down to the 2-D
problem it actually is - dropping the two cap faces, dividing volumes and areas
by the thickness, keeping the other two coordinates.

**MRST's version does not run.** It reads `G.faces.numNodes` and
`G.cells.numFaces`, which grids stopped carrying in favour of `nodePos` and
`facePos`; on `computeGeometry(cartGrid([1,4,3],[1,4,3]))` it fails on its third
line with `Unrecognized field name "numFaces"`. This is the same class of
finding as `assignRelPermScal` in the ad-props work: shipped code that cannot
execute. Supplying the two legacy fields by hand revives it, and that is how the
reference for this port was produced - the validator says so, since a reader
should not have to discover that the reference came from a patched function.

Even revived, its output has two further defects:

- **No `griddim` and no `type`.** `griddim` is never assigned, and the line
  meant to record the transform reads `G = [G.type, mfilename]` - assigning to
  the *input* variable, dead at that point, instead of to `G2.type`. The result
  is not a usable grid structure, since nearly everything downstream branches
  on `griddim`. The port emits a complete `Grid`, and a test runs `check_grid`
  over the result for all three collapse axes.
- **Face tags are remapped in one branch of three.** Collapsing x renumbers the
  surviving tags 3..6 to 1..4; collapsing y or z leaves them alone. Whichever
  is intended, all three cannot be right. The port renumbers uniformly.

Two choices worth recording. The three MATLAB branches differ only in which
coordinate is dropped, so this **writes them once**, parameterised by the
collapsed axis - about 90 lines against MRST's 230. And **geometry is
recomputed rather than scaled**: MRST scales the stored areas and volumes and
then re-sorts each face's node pair with two loops to fix orientation, whereas
this builds the topology, orients the pairs against the retained 3-D normals,
and lets `compute_geometry` do the rest. That turns the areas into a
cross-check instead of a restatement, and the validator compares them against
MRST's scaled ones - all 28 checks pass across three axes and a non-unit
thickness.

**Two more since**: `sort_horizons` and `refine_grdecl_layers`, both validated
against MATLAB (`scripts/validate_horizons.py`). The ledger reads **50 of
160**, `gridtools` at 26 ported and 8 missing.

`sort_horizons` orders surfaces top to bottom, which everything that builds a
grid from horizons assumes. It ranks by each surface's **shallowest** point,
`min(z)`, which is MRST's choice and not the only reasonable one - ranking by
mean would disagree, and a test exhibits a tilted surface where it does.
Interpenetrating horizons have no well-defined order at all, and ranking by
minimum resolves that arbitrarily rather than reporting it: if your surfaces
cross, sorting is not the step that will tell you.

`refine_grdecl_layers` splits a band of layers into `ref` sub-layers each,
leaving the rest alone - the case being one interval that needs more vertical
resolution than the overburden. `COORD` is untouched, which is what makes it
cheap and also bounds it to purely vertical refinement. All three dependencies
(`grdecl_xyz`, `cut_grdecl`, `refine_grdecl`) were already in PyMRST.

Its band is **0-based and inclusive** where MRST's is 1-based, so MRST's
`[2 3]` is `(1, 2)` here. The validator's cases deliberately include a band at
the very start and one covering the whole grid, where an off-by-one runs off an
end rather than landing somewhere plausible. `ZCORN` is compared entry for
entry - a corner-point deck is indexed by logical position, so there is no
numbering ambiguity to work around here as there was for the unstructured grids.

**One bug of mine, worth recording because the failure was informative.** I
first returned the reassembled `ZCORN` as a 3-D array; it travels flat in
Fortran order, and `grdecl_xyz` is what unpacks it. `process_grdecl` did not
reject the malformed deck - it warned about "non-monotone corner-point depths"
and returned **24 disconnected grids** instead of one. That the return type
changes shape (a bare `Grid` when connected, a list when not) is what made the
breakage visible, and the test now asserts connectedness explicitly rather than
indexing blindly into whatever came back.

**Two more since**: `convert_horizons_to_grid` and
`make_layered_horizon_grid`, both validated against MATLAB. The ledger reads
**52 of 160**, `gridtools` at 28 ported and 6 missing. **The horizons group is
complete.**

The two solve the same problem by different routes, and the difference is the
reason both exist. `convert_horizons_to_grid` builds a corner-point deck on a
**Cartesian** areal mesh over the horizons' bounding boxes;
`make_layered_horizon_grid` extrudes **whatever areal grid you already have** -
triangular, PEBI, fitted to wells and faults - onto the same surfaces.

Their interpolation differs accordingly, which is easy to miss: the first uses
`griddedInterpolant` and so **requires horizons on a rectilinear mesh**, while
the second uses `scatteredInterpolant` and accepts arbitrary point clouds. That
makes the second the one to reach for with raw picks.

`convert_horizons_to_grid` matched MATLAB **entry for entry on `COORD`, `ZCORN`
and `ACTNUM`** across six cases - a deck is indexed by logical position, so
there is no numbering ambiguity to work around. `make_layered_horizon_grid` is
compared order-independently (counts, total volume, sorted depths and volumes),
because the extrusion inherits the areal numbering and `remove_short_edges`
renumbers again after collapsing.

Findings:

- **`ndgrid` versus `meshgrid` orientation is a real trap** in
  `convert_horizons_to_grid`. Horizons must have `x` varying down the rows;
  MATLAB's `meshgrid` and NumPy's default `np.meshgrid` both give the
  transpose, and MRST's own example disguises this by writing
  `[y,x,z] = peaks(30)` - swapping the outputs rather than explaining. The port
  checks the orientation and rejects a transposed horizon with a message saying
  so, instead of silently building a transposed grid.
- **`makeLayeredHorizonGrid`'s `extrapMethod` option has no effect.** It is
  declared and documented by implication, but `scatteredInterpolant` is built
  with only the interpolation method. The port omits the parameter rather than
  accepting a dead one, and documents the behaviour that actually applies:
  nodes outside the horizons' hull are linearly extrapolated, so an areal grid
  overhanging the surfaces gets invented depths rather than an error.
- **MRST's monotonicity check is a bare `>= 0`, which fires on rounding.**
  Interpolating a surface that *touches* another returns values like `-5e-18`
  at the tangency rather than exactly zero - measured, not assumed - so a
  strict test reports "non-monotone data" on precisely the tangency case most
  likely to be deliberate. Both ports use a scale-relative tolerance, and the
  warning still fires for genuinely crossing horizons.
- **PyMRST's `remove_short_edges` returns no cell map**, where MRST's
  `removeShortEdges` returns one. Rather than change a shared function's
  signature, the map is recovered by numbering the cells into `indexMap` first
  and reading back what the removal left - and any pre-existing `indexMap` is
  restored through it.

Both keep MRST's habit of **sorting depths along each pillar**, which silently
repairs crossing horizons after warning about them. That is kept because the
alternative is a deck `process_grdecl` will reject; the warning is the part to
act on.

**Three more since**: `compute_vertical_grid_intersection`,
`create_augmented_grid` and `create_augmented_grid_tedfm`, all validated
against MATLAB. The ledger reads **54 of 160**, `gridtools` at 30 ported and 4
missing - **and the four that remain are MRST's own test and example
scaffolding**, so Phase 4 is functionally complete.

`compute_vertical_grid_intersection` extrudes a map-view polyline into a
vertical surface and returns the polygon it cuts from each cell - a fence
diagram, or which cells a well trajectory crosses. Both its dependencies
(`add_bounding_box_fields`, `get_faces_close_to_segment`) came from earlier in
this phase, and the coarse-then-fine filtering means the exact intersection
arithmetic only runs on a handful of faces. It matched MATLAB on all four cases
including a bend.

Two things about its output surprised me, and both are now asserted:

- **Every vertex appears twice**, because an intersection point sits on an edge
  shared by two of the cell's faces and both contribute it. MRST knows - its
  source carries a commented-out `order = order(1:2:end)` and a note that
  de-duplicating would lose neighbour information - and ships the duplicates.
- **The cells at either end of the polyline get clipped, open polygons**: two
  distinct corners where a fully-crossed cell has four. That is correct - the
  surface stops where the trace stops - but it reads as a bug, and my first
  test asserted the duplication on a *clipped* cell and failed for that reason
  rather than because the code was wrong.

**`create_augmented_grid` is now complete and at MRST's path.** PyMRST had only
the partial version inside `pymrst/modules/vemmech/vem.py`, which the
`create_grid_mappings` work exposed. This adds the node-to-face and
node-to-cell maps and the 3-D edge structure (`edges.nodes`, `faces.edges`,
`faces.edgeSign`) that were missing entirely. A test checks Euler's formula,
`V - E + F - C = 1`, and another asserts it agrees with the vemmech version on
the fields they share. **vemmech still calls its own** - that module is
reference-validated throughout and rewiring it buys no behaviour anyone asked
for - so the decision flagged earlier is still open, but the complete version
now exists to switch to.

`createAugmentedGridTEDFM` is MRST's hundred-line copy of `createAugmentedGrid`
with every `G` rewritten to `G.Matrix`, kept in step by hand. Here it is the
three-line wrapper it should have been.

**MRST's position-array helper is wrong for a grid with an unreferenced node**,
and this was measured rather than read off the source. `map2Pos` builds each
`*Pos` array with `find(diff(keys) == 1)`, assuming the sorted keys have no
gaps. Adding one node that belongs to no cell to a 27-node grid makes MRST
return `nodes.cellPos` and `nodes.facePos` with **28 entries where 29 are
needed** - so the last node's range is unreadable and everything after the gap
is misaligned. This is reachable: `make_planar_grid` carries its parent's whole
node array, so most of its nodes are unreferenced. The port counts occurrences
instead, and the validator asserts both that MRST is short and that the port is
not.

**Still missing (4), all of it MRST's own test scaffolding:**

| file | what it is |
|---|---|
| `test_gridtools` | MRST's test script for this directory |
| `gridHelpersExample` | a worked example |
| `plotSingleCellInfo`, `plotSingleFaceInfo` | plotting helpers for the above |

Porting these is not obviously worth doing, and I would argue against it:
PyMRST's per-file tests already cover the same ground more thoroughly than a
single script can, and two of the four exist only to draw pictures. Recorded as
a recommendation rather than acted on.

**Superseded list** - an earlier version named five files that were already
ported:

| group | files |
|---|---|
| construction and repair | `createAugmentedGridTEDFM` |
| geometry | `computeVerticalGridIntersection` |
| MRST's own examples and tests | `gridHelpersExample`, `test_gridtools`, `plotSingleCellInfo`, `plotSingleFaceInfo` |

The last group is MRST's test scaffolding, and porting it is not obviously
worth doing: PyMRST's per-file tests already cover the same ground, and two of
the four are plotting helpers. Flagging it rather than assuming.

Two entries - `translateGrid` and `computeGridSlicePolygons` - remain `ported-elsewhere` in
`pymrst/gridprocessing/`, and MRST keeps both only in `utils/gridtools/`, so on
the layout contract they should move here.

## Phase 5: `inout`

**19 files - but read the overlap note before starting.**

`utils/inout/eclipse/deckinput_simple/` is MRST's *simple* deck reader, and it
**duplicates** functions that also live in `modules/deckformat/deckinput/`:
`readDefaultedRecord`, `readDefaultedKW` and `readVector` exist in both places
in MRST itself. PyMRST already has the `deckformat` versions, ported and
validated against MRST during the `ad-props` work.

So this phase is not 19 new files. It is:

1. deciding whether PyMRST wants a second, simpler deck reader at all -
   probably not, in which case `deckinput_simple` is excluded with that reason
   recorded, the way `merge_options` is;
2. porting what is genuinely absent and not duplicated: `readGRDECL`'s
   simple variant, `writeGRDECL`, `readEQUIL`, `readFaults`, `readMultfelt`,
   `readSimData`/`writeSimData`, `readSimRock`, `convertInputUnits`,
   `readCache`/`writeCache`/`cacheDir`, `splitString`.

`writeGRDECL` is the interesting one: PyMRST can read corner-point grids but
cannot write them, so it cannot round-trip a modified grid to a file another
tool reads.

### Phase 5 status: the premise above was wrong; the layout is fixed and two utilities are ported

**`deckinput_simple` is not superseded, and the framing above is a mistake
worth leaving visible.** Its `readGRDECL` is referenced from **72 files**
across MRST - it is the standard way to read a GRDECL *grid file*, as opposed
to a full ECLIPSE *deck*. `readDefaultedKW` has 9 callers. Only
`applyOperatorSimple` has none. MRST does not treat the two readers as versions
of each other: it **duplicates** `readVector.m` into both trees rather than
sharing it.

So the 19 were never one decision. They are two unrelated groups: twelve
belonging to `deckinput_simple` (several of them `readVectorOld`,
`readVector_textscan`, `readSCHEDULE_orig` - superseded *within MRST itself*),
and seven ordinary file-I/O utilities with nothing to do with deck readers.

**The layout is now MRST's.** `read_grdecl`, `read_defaulted_record` and
`cut_grdecl` moved from `modules/deckformat/` and `gridprocessing/` to
`utils/inout/eclipse/deckinput_simple/`, with re-exporting shims left at all
three old paths so none of the 13 call sites changed. `ported-elsewhere` drops
from 5 to 2 and the ledger reads **61 of 160**.

Two things the move taught, neither of which was visible beforehand:

- **It exposed a genuine import cycle, not just a tidiness question.**
  `read_grdecl` needs `read_keywords`, which stayed in `modules/deckformat`;
  that made `pymrst/utils` import from `pymrst/modules`, and since
  `modules/deckformat/__init__` imports the shim, the shim imported back into
  a half-initialised module. `read_keywords` had to move too. The layering rule
  - utils never depends on modules - is not stylistic here; violating it did
  not work. MRST has the same duplication for the same reason.
- **Two moved modules located their test data by counting parent directories**
  (`Path(__file__).parents[5]`), which the move silently invalidated - the path
  resolved to `PyMRST/mrst-2024a-Geomech/...`, one level short. They now search
  upward for the sibling tree, which survives the next move.

**`write_grdecl` and `convert_input_units` are ported**, the second validated
against MATLAB. `write_grdecl` is tested by **round-tripping through
`read_grdecl`** rather than against a fixture: the file is a means, and what
matters is that reading it back rebuilds the same grid. Where MRST silently
drops any keyword it cannot write, this returns the list of them - a deck that
quietly loses `MULTX` or a fault definition is worse than one that says so.

`convert_input_units` keeps MRST's refusal to guess: an unrecognised keyword
raises rather than passing through unconverted, because silently leaving a
permeability in millidarcy is a factor-of-1e15 error that announces itself
nowhere. `known_only=True` is offered for callers who would rather be told than
stopped. MRST errors there too, and the validator asserts both do.

**A silent data-corruption bug, found by re-checking an exclusion.**
PyMRST had **no support for ECLIPSE's box operators** - `ADD`, `COPY`,
`EQUALS`, `MAXVALUE`, `MINVALUE`, `MULTIPLY` - anywhere, and `read_keywords`
skips unknown keywords without a word. So a deck reading

    PERMX
      4*100 /
    MULTIPLY
      'PERMX' 0.1 /
    /

came back with **PERMX = 100 where the deck says 10**. A 10x permeability
error, no warning, numbers that look entirely reasonable.

It was found by re-examining files this plan had dismissed as "superseded" - a
claim made from filename patterns rather than caller counts. Counting them:
`readRecordString` has **15 callers**, `readOperator` 2, `applyOperatorSimple`
2. Not dead code. `apply_operator_simple` is now ported with all six operators
and wired into the reader, and `read_operator`, `read_operator_mod_field` and
`read_record_string` with it.

`readOperatorModField` turned out to handle **only `MULTIPLY` and `EQUALS`**
despite taking the keyword as an argument - the other four fall through its
if-chain and are silently ignored. It also carries the **fourth** copy of the
dead `if(~length(split)==8)` check, after `readFaults`, `readMultfelt` and
`readOperator`: one mistake, pasted four times, never fired once.

**The cache group was mis-categorised too.** Three of the six were not caches:
`cacheDir` returns a *path*, and `readSimRock`/`readSimData` read plain-text SAM
simulator files. The three real ones are ported, storing pickles rather than
`.npz` because what is worth caching here is `Grid` objects, which `savez`
cannot represent - with the consequence stated in both docstrings, that a
pickle executes code on load and only a cache this process wrote may be read.

**SAM simulator I/O is now complete at the MRST layout.** `readSimData`,
`readSimRock`, and `writeSimData` use MRST's actual plain-text formats:
`-geom.dat`, `-topo.dat`, optional `-map.dat`, `-perm.dat`, `-poro.dat`, and
`-wells.dat`. The grid reader reconstructs zero-based half-face topology and
neighbor orientation, geometry, optional Cartesian maps, and validates the
resulting `Grid`. Rock permeability is written in mD and returned in SI units.
Round-trip tests cover 2-D and 3-D grids, anisotropic rock, map data, malformed
record counts, sources, and wells.

After this increment, `inout` has 18 ported owners and zero missing runtime
entries. The five remaining MRST equilibrium/gridtools test and example files
have also been ported at matching layout paths. `gridAddHelpers` is now a real
MRST-shaped owner with lazy plotting closures and grid-query helpers rather
than an environment exclusion. Headless diagnostic ports for
`plotSingleCellInfo` and `plotSingleFaceInfo` return inspectable geometry and
optionally render on supplied Matplotlib axes.

The ledger now reports 97 ported owners, 60 consolidated owners, and zero
pending in-scope entries. The utility test suite contains 173 passing tests;
its executable-module self-test gate independently reports 95 passing modules.
The core MATLAB fixture additionally covers `blockDiagIndex`, `interpTable`,
`dinterpTable`, `dinterpq1`, `discspace1`, `formatTimeRange`, dimension-aware
`rlencode`, subset arithmetic, `numelValue`, `reduceToDouble`, and
`uniqueStable`. This validation exposed and closed the missing spline option,
dimension argument, stable-unique owner, and MRST-specific time-limit semantics.
The second MATLAB fixture validates blockwise `multiEig`/`multiSymmEig`,
projection, sorting, concatenation and field replacement for `IndexArray`, the
`TensorProd` dispatch/evaluation path, and the final `SparseTensor` matrix.
It also records MRST 2024a's reproducible `compIndexArray` crash while testing
PyMRST's corrected documented behavior. Overall reference coverage is now
**127 of 157** in-scope utilities.

The grid-query MATLAB fixture then validates packed cell-face, cell-node and
face-node extraction, indirection lookup, logical Cartesian indices,
geometrical/topological/NNC neighbourships, connectivity matrices,
node-sharing neighbours, connected subsets, and translated geometry. It
explicitly accounts for MRST's cell-zero versus PyMRST's outside-cell `-1`
convention. Every in-scope `gridtools` entry now has reference evidence and
overall coverage is **139 of 157** utilities.

The remaining-numerics fixture validates `lMultdiag`, `cellDimsCG`, cyclic
`getSortedCellNodes`, `incompHydrostaticPressure`, `isCoarseGrid`, and all
fields of all four `getUnitSystem` variants. MATLAB exposed two observable
contracts that are now reproduced: coarse status depends on the presence of
`parent` and `partition` even when empty, and the hydrostatic sampler stores
its second depth at two increments while advancing pressure by one. Reference
coverage is now **145 of 157**; the final twelve entries are all deck/cache
I/O utilities.

## Expanded infrastructure scope

The earlier 157-file runtime scope excluded MRST's entire `dataset_manager`
and `settings_manager` trees. That exclusion was too broad for literal layout
and end-to-end reservoir-workflow parity and has been removed from the audit.
The new Python packages contain direct implementations—not forwarding
wrappers—of persistent typed settings, mapping migration, dataset metadata and
discovery, availability/path resolution, download policy, and safe ZIP/TAR
installation. Reservoir-critical definitions currently exist at MRST-shaped
paths for SPE1, SPE3, SPE9, SPE10, EGG, Norne, and the Norne field model.

That expanded backlog is now retired. All 39 MRST dataset definitions have
direct metadata modules; the four pinned Norne/SPE10 installers, bulk-download
and example-list utilities, testable dataset browser, and runnable example are
also implemented. `dataset_manager` and `settings_manager` now report **55
ported, 2 consolidated, and zero pending** entries together. Downloads use
atomic temporary files and reject archive path traversal; persistent settings
are written atomically and validated before use.

## Phase 6: `tensorAssembly`

**14 files, ~1,200 lines.**

`IndexArray`, `SparseTensor`, `TensorMap`, `TensorProd` and their helpers: a
small algebra for assembling sparse operators by named index contraction, used
by `mpsaw` and the finite-volume biot code.

PyMRST has `SparseMultiArray` in `modules/vemmech/ad_version/`, which the
`vemmech` ledger already records as the consolidated counterpart of MRST's
`ssparsemul`, `tbinaryop` and `tcontract`. **Establish whether
`SparseMultiArray` is the same algebra before porting anything**: if it is,
this phase is mostly moving it to `utils/tensorAssembly/` and giving it MRST's
names; if it is not, it is 1,200 lines of new code.

That check is the first task of the phase and it decides the size, so do not
schedule the rest until it is done.

**Size.** 1 day to answer the question; 1–5 days after, depending.

### Phase 6 status: ported in full - the recommendation against it was wrong

**All 14 files are ported**, with three MATLAB validators. The analysis below is
left standing because it was right about the *facts* and wrong about the
*conclusion*: `SparseTensor` and `SparseMultiArray` really are different
classes, and `mpsaw` really does assemble without either - but "no consumer in
PyMRST yet" is a poor reason to skip a subsystem when the goal is end-to-end
parity and `vag`, `nfvm` and `mpfa` all use it.

The port took one session. `crossIndexArray` - the inner join everything else
rests on - matches MRST on **rows, row order, column names and both index
maps**; order matters there because those maps are positional, so a join with
correct pairings in the wrong order would silently corrupt its callers.
`TensorMap` matches on results and dispatch vectors for the same reason.

**Four MRST defects, all verified in MATLAB rather than read off the source:**

- **`compIndexArray` always returns true.** It cannot run at all - `isvirtual`
  is `[]` for any ordinary array and `[] || []` raises - and forced past that,
  it compares `tbl1` against a permutation of *itself*: `sortrows(tbl1.inds(:,
  fdind2))` where `tbl2` was meant. Two arrays with entirely different contents
  compare equal.
- **`CrossIndexArrayGenerator`'s constructor discards every argument**, parsing
  into a local struct and never assigning to the object. `eval()` then fails
  with "Dot indexing is not supported".
- **`crossindconstruct` has two failure modes**, one silent: a trailing
  `nb2 = 0` raises, and an *interior* `nb1 = 0` makes write positions collide so
  later groups index into the wrong block - answering `[5,6]` where the
  definition gives `[8,9]`.
- **`sortIndexArray`'s `keepAllFields` does nothing**, building a list from
  `fieldnames(tbl)` (property names, not index names) and never using it.

### Phase 6 (superseded analysis, kept for the reasoning)

**`SparseMultiArray` is not MRST's `SparseTensor`, and never was.** They are
two different classes, in two different places in MRST:

| | `utils/tensorAssembly/SparseTensor.m` | `modules/vemmech/ad_version/SparseMultiArray.m` |
|---|---|---|
| size | 340 lines, 8 methods | 1590 lines, ~50 methods |
| model | a sparse matrix built from `TensorMap` / `TensorProd` descriptors | named-index multi-array with contraction, AD, permutation |
| API | `getMatrix`, `setMatrix`, `setFromTensorMap`, `mtimes`, `plus`, `transpose` | `contract`, `tensor_product`, `changeIndexName`, `asMatrix`, `sub`, ... |
| callers in MRST | 18 | 8 |

PyMRST ported the **vemmech** one, and put it at the matching path
(`pymrst/modules/vemmech/ad_version/`). So the premise behind this phase - that
the two might be the same algebra wearing different names - is simply false,
and `utils/tensorAssembly` is genuinely unported.

**But nothing in PyMRST would use it.** `SparseTensor`'s main consumer is
`modules/mpsaw/assembly` (9 of its 18 callers), and **PyMRST's `mpsaw` is
already ported - in 7 files, with no tensor class at all.** It assembles
directly into sparse matrices rather than through the index-contraction layer.
The remaining callers are `vag`, `nfvm` and `mpfa`, none of which PyMRST has.

So porting these 14 files would add ~1,200 lines of machinery with **no caller
in this repository**, and its only plausible purpose would be re-porting
`mpsaw`'s assembly to go through it - which is a change to working, ported code
for no behaviour anyone has asked for.

**Recommendation: leave it.** The 14 files stay `pending-review` in the ledger
rather than being reclassified as excluded, because they *are* unported and
calling them excluded would flatter the number. If PyMRST ever ports `vag`,
`nfvm` or `mpfa`, this phase becomes real work and should be scheduled then -
those modules use the abstraction throughout, and reimplementing it per module
would be worse than porting it once.

## Phase 7: the remainder, tests and the ledger

- `equil/simpleEquilibrium` (2 files) - hydrostatic equilibration, pairs with
  `incompHydrostaticPressure` from Phase 2.
- `ad_utils/lMultdiag` (1 file) - a diagonal left-multiply for AD.
- **`tests/utils/`**, which does not exist. `pymrst/utils` has 21 files and no
  test directory; its coverage is whatever other modules' tests happen to
  exercise. At minimum: `rldecode`/`rlencode` round-trip, `mcolon` against
  concatenated ranges, `blockDiagIndex` against an explicitly built matrix,
  `interpTable` against its derivative `dinterpTable`, and the units against
  documented conversions.
- Ledger integrity test, with a basename distinct from the existing
  `test_parity_ledger.py` collision - `test_utils_parity_ledger.py`.

**Size.** 3 days.

### Phase 7 status: complete but for MRST's own test scripts

Three test files, 96 tests, plus the two library items - `equil` and
`ad_utils/lMultdiag` - now ported and at their MRST paths.

**`test_utils_selftests.py` collects what nobody was collecting.** Every file
in `pymrst/utils` carries its tests in an `if __name__ == '__main__'` block,
which keeps them next to the code but means they only ran when someone
remembered to invoke the file by hand. This runs all 56 as subprocesses - not
imports, since the tests live under `__main__`, and not `runpy`, since several
mutate grids and a shared interpreter would let them interfere. The module list
is **discovered rather than listed**, so a new file is covered the moment it
lands; the failure mode being guarded against is someone adding a module and
forgetting to wire up its tests. A second check asserts each one actually
prints its "All tests passed" line, because a module whose tests silently do
nothing would exit 0 either way.

**`test_utils_parity_ledger.py` checks the ledger against the filesystem**
rather than against itself. The two that carry weight: every `ported` row must
name a file that exists, so a rename cannot quietly turn a port into a claim;
and every `ported` row must sit under `pymrst/utils`, which is the layout
contract this whole effort is about. It also asserts `UTILS_PARITY.md`'s
headline matches the CSV, so a hand-edited number fails rather than misleads,
and that Phase 4's "complete on substance" claim above still holds.

**`test_utils_integration.py` found a real bug in Phase 2's work.** Its first
test checks that every name in `__all__` resolves to a callable rather than to
a module - the shadowing trap this codebase has already been bitten by, where
`mcolon.py` defining `mcolon` means the package attribute is whichever was
bound last. It failed immediately: `pymrst/utils/__init__.py` imported
`cell_flux_to_face_flux`, `face_flux_to_cell_flux` and
`face_flux_to_cell_velocity` with `from pymrst.utils import ...`, which binds
the **modules**, so `from pymrst.utils import face_flux_to_cell_flux` handed
back something that fails on call. Fixed to import from the submodules. Nothing
depended on the broken binding, which is why it survived: the flux converters
are reached by their full paths everywhere else.

The rest of that file covers seams no single module can: `sort_grid` preserving
geometry, `compare_grids(sort=True)` seeing through a renumbering,
`volume_by_gauss_greens` agreeing with `compute_geometry`, the augmented maps
being mutual transposes, and `get_faces_close_to_segment` missing no crossed
face - each across four grid types (Cartesian 2-D and 3-D, twisted, triangular)
rather than the one grid each module happened to use.

**Phase 7's two library files are done.** `simple_equilibrium`
(`utils/equil/`) and `l_mult_diag` (`utils/ad_utils/`), both at MRST's paths;
the first validated against MATLAB, the second checked against the explicit
`diag(d) @ J` product it replaces. The ledger reads **56 of 160**. Only
`equil/test_simpleEquilibrium.m` remains in those two directories, and it is
MRST's own test script.

`simple_equilibrium` distributes phases geometrically from contact depths - no
capillary pressure, no fluid physics, just the fraction of each cell's height on
each side of a contact. Three things about it are worth knowing, and all three
are now asserted:

- **MRST's version cannot run on a 2-D grid at all.** Its default direction is
  `[0 0 1]`, which cannot multiply a 2-D grid's `(n, 2)` centroids; MATLAB
  fails with "Arrays have incompatible sizes for this operation". The port
  defaults to the last coordinate, sized to `griddim`, and works. The validator
  records MATLAB's error and asserts both halves.
- **Columns come back heaviest-first**, reversing the order the contacts were
  given in - the opposite of what the argument order suggests.
- **A cell with no volume comes back all zeros.** Its centroid is `NaN`, so the
  zero-height fallback cannot place it, and its saturations do not sum to one.
  MRST has the same gap; my first test constructed exactly such a cell and
  failed, which is how it was found. Documented as a limitation rather than
  papered over.

`l_mult_diag` scales Jacobian rows. MRST builds an `n x n` sparse diagonal and
runs a full sparse product per block; scaling the stored entries directly is
the same operation in one pass. For a million-cell model with three primary
variables that is three million-square matrices and three sparse products
avoided.

**The whole test suite can be collected again.** `pytest tests/` failed with
four collection errors from three duplicated basenames -
`test_parity_ledger.py` in three directories, `test_compute_cp_geometry.py` and
`test_parity_audit.py` in two each - because without `__init__.py` pytest
cannot tell same-named modules apart. Adding the 22 missing `__init__.py` files
fixes it, and collection goes from *2165 tests + 4 errors* to **2274 tests, no
errors**. Worth knowing: every parity claim in this repository rests on a suite
that, until now, could not be run in one command.

**And running it turns out to matter.** The first full run - 68 minutes - gives
**2000 passed, 273 failed**. Those failures are not new; they had simply never
been visible, because the collection error aborted the run before anything
executed. `tests/test_all.py`, which replays the saved MATLAB references in
`examples/validation/`, had never once run.

Every failure sampled is the **same pre-existing circular import**:
`pymrst.modules.compositional.model` cannot import `CompositionalState` from
itself. The traceback runs entirely through `pymrst/modules/ad_core/utils/` and
`pymrst/modules/compositional/utils/` - module-level subpackages, unrelated to
`pymrst/utils/`. It is one defect blocking a large block of validators, and
fixing it is the single highest-value piece of work left in this repository:
until it is fixed, roughly an eighth of the suite cannot report anything.

**Since fixed, and it was one line of layout.** `expandMatrixToCell.m` lives in
MRST's `modules/ad-core/utils/`. PyMRST had the implementation in
`compositional/utils/` with `ad_core` re-exporting it - a lower-level module
importing a higher-level one, which closed the cycle:

    compositional/model -> ad_core/__init__ -> ad_core/utils/__init__
      -> ad_core/utils/expand_matrix_to_cell -> compositional/utils/__init__
      -> compositional/utils/equations_compositional
      -> compositional/model   (still half-initialised)

Moving the implementation to its MRST path and leaving the shim in
`compositional` points the dependency downward and breaks the cycle. **The
MATLAB validation set went from 273 failures to 3**: 421 of 421 now pass.

The three that survived were real bugs the import error had been masking:

- `compositional/facility.py` raised `UnboundLocalError` on any `WCONINJH`
  history-rate record, because `status` was read inside the `else` branch of a
  two-way choice and used after it. The producer loop ten lines above reads it
  before the branch; the injector loop had it nested by mistake. Two validators.
- `validate_phase320.py` built `operators = SimpleNamespace()` with no
  attributes and passed it to `ReservoirModel`, which now constructs a
  `PressureGradient` needing `operators.grad`. A fixture that had fallen behind
  the model, not a library defect - the checks around it only exercise name
  aliases, so the operator is never called.

Both are the same lesson as the cycle itself: **a test that cannot run cannot
tell you anything**, and three defects sat behind one import error for as long
as the suite was uncollectable.

**A ratchet now guards the class of bug that caused all this.**
`tests/test_import_health.py` builds the module-level import graph of `pymrst`
(1170 files, 2656 edges) and asserts three things:

- **every module in an import cycle still imports from a cold interpreter** -
  the exact check the compositional bug failed. There are 5 cycles today,
  spanning 19 modules, and all 19 import cleanly; the compositional one was the
  only cycle that had actually broken.
- **the cycle count does not grow.** The five that remain are mostly the benign
  "package `__init__` imports its submodule, submodule imports back" shape,
  which works until it does not - so they are ratcheted rather than forbidden.
  Unpicking six interlocking `vemmech` modules is a real refactor, not a test's
  business; the point is that the *next* one is noticed the day it lands.
- **`pymrst/utils` never imports `pymrst/modules`.** Verified non-vacuous: 50
  utils modules do import from `gridprocessing`, `params` and `utils`, and none
  from `modules`. This is the invariant `read_keywords` had to move to restore,
  and there the upward import was not untidiness - it deadlocked the import.

**The last two red tests were the ledger tests themselves**, and both were
asserting stale facts rather than catching real ones:

- `tests/ad_blackoil` asserted `eggExample` was still `pending`. It has since
  been ported *and* reference-validated - `validate_ad_blackoil_egg.py`
  reproduces the MRST setup and production regression and passes. The
  expectation was updated, not the ledger; the test is named
  "remain_honestly_pending", so flipping it without checking the port was real
  would have been exactly the flattery it exists to prevent.
- `tests/ad_core` asserted `len(rows) == 345` under the name
  `test_ad_core_ledger_covers_reference_tree`. A hardcoded count is not
  coverage: it had been failing at 346 against 345 and telling nobody anything,
  and bumping the number would only re-arm the trap. It now compares the
  ledger's paths against the MRST tree on disk - 346 files, each listed once -
  which is the property the name promises and cannot drift.

**Final full-suite state: 2282 passed, 11 failed** (73 minutes), from
*2000 passed, 273 failed* before this work - and before that, from a suite that
could not be collected at all.

All 11 survivors are `test_module_selftests`, none in code touched here, and
they fall into three kinds:

| failures | cause |
|---:|---|
| 3 | `get_flux_and_props_{gas,oil,water}_bo.py` use **relative imports in their `__main__` block**, so they import fine as modules and fail as scripts. A harness mismatch rather than a defect - either the blocks move to real tests, or the harness runs them with `-m`. |
| 2 | `_pyvista_grid.py` and `plot_saved_states.py` raise `TypeError: stat: path should be ... not PlotResult` - a real bug in their own self-tests. |
| 1 | `multisegment_equations.py` raises `AttributeError: 'Model' object has no attribute 'raw_fluid'`. |
| 5 | the SPE1/SPE3/SPE9 and Norne example drivers, which run full simulations; SPE9 alone exceeds ten minutes. These may be failing or merely slow, and telling which needs a longer budget than a self-test sweep should have. |

**Six of the eleven are since fixed, one is correctly reclassified, and four
remain.** In order:

| was | now |
|---|---|
| 3 x `get_flux_and_props_*_bo.py` | **fixed** - relative imports made absolute. There were 8 relative imports in all of `pymrst` against 2656 absolute ones; all 8 are now absolute, so no outlier is left to trip the next person. |
| `_pyvista_grid.py` | **fixed** - the test asserted on `plot_grid`'s return value, which became a `PlotResult` rather than a path. It already knew the filename; it now stats that. |
| `timestep_control_demo.py` | **fixed** - imported `build_spe1`, which never existed. The setup step it wanted is `setup_spe1_tutorial`, returning the same `(deck, model, state)` triple. Runs in 4s. |
| `plot_saved_states.py` | **reclassified** - its `__main__` is an argparse CLI needing a `state_file`, so running it bare prints usage and exits 2. That is the tool working. The harness now skips exactly that signal, rather than skipping every module mentioning argparse: two of the three that do have defaults and run their tests fine. |
| `multisegment_equations.py` | **partly fixed, still failing.** It required `model.raw_fluid` outright, where `flux_and_props_bo`, `get_b_o_bo` and `get_polymer_shear_multiplier` all fall back to `fluid`; it now follows the same idiom. That exposed a second problem - the self-test's `Fluid` stub has no `b_w` - which is a fixture that does not model a black-oil fluid and needs someone who knows what it should be. |
| SPE3, SPE9 x2, Norne | **still failing, and genuinely so.** These are not merely slow: SPE3 runs 146 seconds and then raises `RuntimeError: SPE3 step 20 failed to converge`. A numerical convergence failure in a flagship example is worth someone's attention, and would have been hidden had these been moved behind an opt-in marker - which is why they were not. |

The last group raises a question for whoever owns `tests/test_all.py`: a sweep
that runs every module as a script will keep colliding with modules whose
`__main__` is a demo rather than a test. Splitting "modules with unit tests"
from "example drivers" would make the suite both faster and more honest.

The `utils` work is unaffected - all 56 module self-tests pass, and the six
suites exercised directly (gridprocessing, vem, vemmech, ad_props, deckformat,
utils) are at 610 passing - but the honest headline for the repository as a
whole is 2000 of 2273, not "everything passes".

## Order and cost

| Phase | Work | Size |
|---|---|---|
| 1 | move four files to `utils/` root; decide `value` and `ADI` | 0.5 d |
| 2 | 16 core numeric utilities | 3–4 d |
| 3 | 9 units and `getUnitSystem` | 0.5 d |
| 4 | 33 `gridtools` files | 5–7 d |
| 5 | `inout`, after the duplication decision | 2–4 d |
| 6 | `tensorAssembly`, after the `SparseMultiArray` question | 2–6 d |
| 7 | remainder, `tests/utils/`, ledger test | 3 d |

**Roughly 16–25 days**, and the spread is real: Phases 5 and 6 each turn on a
question that should be answered before their work is scheduled.

**Both questions have since been answered, and both premises were wrong.**
Phase 5 assumed `deckinput_simple` was superseded by `deckformat`; it is not -
`readGRDECL` has 72 callers, and MRST duplicates helpers between the two trees
rather than sharing them. Phase 6 assumed `SparseMultiArray` might be MRST's
`SparseTensor` under another name; they are different classes in different
modules, and PyMRST already has the vemmech one correctly placed.

The pattern is worth naming, because it recurs across this plan and the ad-props
and vem ones before it: **a plan written from file names guesses wrong about
what the files are.** Every such guess here has been resolved by reading MRST's
own call graph - how many files reference a function, which modules consume a
class - and every time the answer changed what the work was. Check the callers
before scheduling the port.

Phases 1, 2 and 3 are independent and could run in any order. Phase 4 depends
on `getSortedCellNodes` from Phase 2. Phase 7's tests should grow with each
phase rather than waiting for all of them.

## What this plan does not propose

**It does not port `merge_options`, `dispif`, or the `mrst*` path functions.**
Those exist because MATLAB lacks keyword arguments, a logging module and a
package system. Porting them would import MATLAB's constraints into a language
that does not have them.

**It does not split `units.py` into 51 modules**, for the reason given in
Phase 3.

**It does not touch `3rdparty`.** If PyMRST ever needs graph algorithms -
`components`, `dmperm`, maximum flow - `scipy.sparse.csgraph` provides them
directly, and the right move is to call it rather than to port a MATLAB
binding to a C++ library.
