# Porting `modules/coarsegrid`

A plan for porting MRST's coarse-grid module properly, rather than extracting
the three functions the examples tree happens to need.

The conclusion first: 37 executable files and 3971 lines, of which 2 files and
252 lines are already ported, both under the wrong path. 489 of the remaining
lines rest on METIS, an external native library, and need a disposition before
they are worth touching. The rest divides cleanly into a combinatorial core, a
coarsening layer built on it, and plotting.

Inventory and status live in `coarsegrid_parity.md`. This document covers what
to build and in what order.

> **Status: all six phases complete.** 34 of 37 files owned, 3 excluded with
> reasons, 323 self-tests across 31 files, and three MATLAB validators covering
> 13 cases. The paragraph above is the plan as first written and is kept for the
> record; each phase now carries an outcome section recording what it actually
> took, including where the plan was wrong.

## Why this module rather than three functions

`showSAIGUP` and `gridTutorialCornerPoint` need only `processPartition`,
`partitionLayers` and `explosionView`, 319 lines between them. Porting just
those would finish the examples tree at 17 of 17.

It would also create `pymrst/modules/coarsegrid/` holding three of thirty-seven
functions, which reads as a complete module to anyone who does not check. The
`utils` port spent real effort driving `ported-elsewhere` deviations to zero
precisely so the layout could be trusted as a map. A three-function module
undoes that for a new reader.

Porting the module makes the two examples fall out at the end rather than
driving a partial port, and it gives upscaling and multiscale work a real
foundation.

## What makes this module unusually testable

Coarse grids are combinatorial. Almost every operation has an exact invariant
rather than a tolerance, so the tests can be assertions rather than comparisons:

* a partition assigns every fine cell to exactly one block;
* the number of distinct blocks equals the coarse cell count;
* summed fine volumes equal the coarse block volume;
* summed fine face areas equal the coarse face area;
* `invertPartition` round-trips;
* `compressPartition` preserves membership while removing gaps;
* `processPartition` splits exactly the disconnected blocks;
* `coarseDataToFine` then averaging recovers the coarse values.

That is worth stating up front because it changes what "done" means here. A
ported file without those assertions has not really been validated, however
plausible its output looks, and coarse-grid errors are the silent kind: a wrong
partition still produces a coarse grid.

## Phase 1: relocate what exists

Two files sit at `pymrst/modules/upscaling/coarsegrid/`:

| MRST | Now | Target |
|---|---|---|
| `generateCoarseGrid.m` | `modules/upscaling/coarsegrid/generate_coarse_grid.py` | `modules/coarsegrid/generate_coarse_grid.py` |
| `partitionCartGrid.m` | `modules/upscaling/coarsegrid/partition_cart_grid.py` | `modules/coarsegrid/partition_cart_grid.py` |

There is also a **third copy**: `upscaling/upscale.py` defines its own
`partition_cart_grid` and `generate_coarse_grid` plus a second `CoarseGrid`
class, importing neither of the files above. The three agree numerically, so
merging is safe, but two distinct `CoarseGrid` types mean any `isinstance`
check or annotation written against one rejects the other.

Move both files, fold `upscale.py` onto the single implementation, leave
re-exports so existing callers keep working, and check the import-health test
still passes. Do this first: every later file imports from the new location, and
moving them afterwards means touching everything twice.

Add the invariant self-tests listed above to both while they are being handled,
since neither carries any today.

## Phase 2: the combinatorial core

Nine files, no external dependencies, and the foundation everything else uses.

| MRST | Lines | Purpose |
|---|---|---|
| `compressPartition.m` | 58 | renumber to remove empty blocks |
| `partitionUI.m` | 94 | uniform partition in logical space |
| `partitionTensor.m` | 84 | tensor-product blocks |
| `partitionLayers.m` | 100 | uniform in I/J, layered in K |
| `processPartition.m` | 117 | split disconnected blocks |
| `cellPartitionToFacePartition.m` | 72 | cell partition to face partition |
| `processFacePartition.m` | 95 | connect coarse faces |
| `subFaces.m` | 71 | fine faces of a coarse face |
| `sortSubEdges.m` | 128 | order subfaces in 2-D |

`processPartition` and `partitionLayers` land here, so the examples tree is
unblocked at the end of this phase apart from `explosionView`.

Order within the phase: `compressPartition` first, since the partition
constructors all end by calling it; then the three constructors; then
`processPartition`, which needs connectivity; then the face-side four.

`private/coarseConnections.m` (96), `private/transposeConnections.m` (69) and
`private/sortedges.m` (156) support this phase and should be ported alongside
the functions that need them rather than as a separate step.

### Phase 2 outcome

Complete. `coarseConnections` and `transposeConnections` are folded into
`generate_coarse_grid` as `_coarse_topology`, which is where MRST calls them and
the only place they are used; `subFaces` is folded in as `CoarseGrid.sub_faces`
for the same reason. `sortedges` supports `sortSubEdges` only and is ported
alongside it, at `private/sort_edges.py`.

Three things came out of this phase that the plan did not anticipate.

**The indicator.** `generateCoarseGrid`'s coarse faces are groups of (block
pair, indicator), and the default indicator carries the half-face tag of every
exterior face. Grouping on block pairs alone gave 13 coarse faces on the
validation case where MATLAB gives 29, with no invariant to catch it. Ported.

**`processFacePartition` is 2D only,** which MRST states in its help text and
enforces with an error. The port refuses a 3D grid the same way rather than
guessing at a polygon-adjacency rule MRST has not written.

**Ordering had to be matched, not just grouping,** so that a coarse face
carries the same number in Python as in MATLAB. That meant reproducing
`coarseConnections`' sort on (block pair, indicator, fine face) and
`transposeConnections`' sort on (block, coarse face). The first version sorted
blocks stably by column, which put a block's faces in the wrong order without
changing any set.

Validated against MATLAB in five cases; see `coarsegrid_parity.md`.

Two more things came out of the face-ordering half, both departures recorded in
the module docstrings.

**`sortedges` drops entries on failure, and this port does not.** MATLAB ends
with `map = map(map > 0)`, so a section whose edges do not form one chain
returns fewer entries than it has edges. `sortSubEdges` then reindexes the
packed `fconn` array anyway, and every coarse face after the failed one reads
the wrong fine faces. Only a warning marks it. The port appends the unreachable
edges instead, so the permutation stays total and `success` is the sole signal.

**`sortSubEdges` is not idempotent, and cannot be.** A block's boundary is a
cycle with no natural first face, and the walk seeds from whichever face is
currently first, so calling it twice rotates `cells.faces`. `fconn` does settle,
because its direction is pinned by `faces.neighbors`. Both halves are asserted,
so the caveat cannot quietly stop being true.

## Phase 3: the coarsening layer

Eight files under `utils/`, all built on Phase 2.

| MRST | Lines | Purpose |
|---|---|---|
| `coarsenGeometry.m` | 77 | coarse volumes, centroids, areas |
| `coarseDataToFine.m` | 64 | expand coarse values onto fine cells |
| `invertPartition.m` | 71 | block-to-cell lookup |
| `isSamePartition.m` | 61 | partition equivalence |
| `fineToCoarseSign.m` | 58 | face orientation across scales |
| `coarsenBC.m` | 97 | coarsen boundary conditions |
| `coarsenFlux.m` | 48 | coarsen fluxes |
| `refineNearWell.m` | 172 | refine a partition near a well |

`coarsenGeometry` deserves a MATLAB validator: summed volumes and areas are
exactly the kind of mapping that produces plausible numbers when the index
arrays are subtly wrong.

### Phase 3 outcome

Complete. All seven non-METIS files in `utils/` are ported, plus
`refineNearWell`, and the validator `coarsenGeometry` deserved is written.

**The table above is wrong about two files, and the caveat below is why.**
`refineNearWell.m` is not in `utils/`; it is a root file. The eighth file in
`utils/` is `callMetisMatrix.m`, which belongs to Phase 5. The line count was
also wrong: `utils/` holds 640 lines, not 648. `refineNearWell` is ported
regardless, at `pymrst/modules/coarsegrid/refine_near_well.py`.

Two behaviours were carried over deliberately rather than fixed, because
changing either would break parity silently:

**`coarsenBC` samples pressure and drops faces.** A coarse face survives only
if its *first* fine face carries a condition, so conditions on the other fine
faces are lost without a warning, and the pressure it reports is that one
face's value rather than an average. MRST's own comment says the sampling
"could be improved". Both are asserted as tests so they stay visible.

**`coarsenBC` sums flux without the orientation sign.** It computes
`fineToCoarseSign` and then overwrites the result; the block is marked as a
hack in MRST. `coarsen_flux` does apply the sign. A `signed=True` flag offers
the corrected behaviour, and the default is MRST's.

One convention had to change. `refineNearWell` marks the masked far field with
block 0, which works because MRST numbers blocks from 1. PyMRST numbers from 0,
so 0 is a real block; the port marks the far field -1, the same sentinel
`faces.neighbors` uses. `refine_partition` performs the combination MRST's
"partitions can be created simply by addition" comment describes, which is the
only thing the marker was for.

## Phase 4: plotting

| MRST | Lines | Note |
|---|---|---|
| `plotting/explosionView.m` | 102 | unblocks the two examples |
| `plotting/plotCellNumbers.m` | 64 | |
| `plotting/plotFaceNumbers.m` | 59 | |
| `plotting/tests/test_newplotting.m` | 174 | MRST's own scaffold, likely excluded |

Follow the convention already used across `pymrst/examples`: compute in a
testable function, plot optionally, return `None` without a backend.

`outlineCoarseGrid` and `plotBlockAndNeighbors` are already ported into
`pymrst/plotting/`, matching MRST's placement of them outside the module.

### Phase 4 outcome

Complete. All three files are ported, each split into a placement function
(`block_displacements`, `cell_number_labels`, `face_number_labels`) and a
drawing function, so a label or block in the wrong position is caught by a test
rather than by looking at a figure that reads perfectly well and is wrong.

`test_newplotting.m` is MRST's own test scaffold and stays excluded; PyMRST's
equivalent is the `__main__` self-tests in each file.

Two deliberate departures, both recorded in the module docstrings:

**Labels are 0-based.** A figure whose numbers disagree with the indices the
user is typing is worse than one that differs from MATLAB. `base=1` reproduces
MATLAB's labels for a side-by-side comparison.

**MRST's private `computeCoarseGeometry` is not reproduced.** Both numbering
plots carry a copy of it, and both compute a coarse face centroid as
`sum(fine centroids) / sum(fine areas)`, with no area weighting in the
numerator. That is the centroid only when every fine face has unit area, which
is true of the unit Cartesian grids the examples use and false in general.
`plotCellNumbers`'s copy also assigns the result to a misspelled field, so its
face centroids stay zero and never reach a plot. The port calls
`coarsen_geometry`, which is MRST's own correct version of the same quantity,
and `plot_face_numbers` asserts on a graded grid that the two formulas differ
so the choice cannot quietly stop mattering.

## Phase 5: METIS, once a disposition is chosen

489 lines across four files. `coarsegrid_parity.md` sets out three options; the
recommendation there is an optional `pymetis` extra, matching how `pyamg` and
`torch` are handled, because a substitute partitioner changes the partitions and
so breaks parity rather than preserving it.

Nothing before this phase depends on METIS, and neither blocked example needs
it, so this can be deferred without holding anything up.

### Phase 5 outcome

The recommendation was taken. `metis = ["pymetis>=2023.1"]` sits in
`pyproject.toml` beside the existing extras, `partitionMETIS` and
`callMetisMatrix` are ported, and the two MEX files are recorded as
**excluded-not-applicable**: they exist to reach the same library through
MATLAB's compiled interface, which `pymetis` does directly.

The split is deliberate. `metis_graph` builds the weighted CSR graph and is
fully asserted — off-diagonal pattern, the `ceil(1e4 * |v| / max|v|)` integer
scaling, symmetry — because it needs no METIS installation. The partitioning
call is one function call behind an import guard, and the two tests that
exercise it print `SKIP`. Neither `pymetis` nor MATLAB's `gpmetis` binary is
installed here, so no reference partition exists on either side to compare
against; `pip install pymrst[metis]` is what closes that.

`connection_matrix` is written directly rather than through `incomp_tpfa`.
MRST calls `incompTPFA` with a no-op linear solver purely to get the matrix
out, which is code reuse rather than a step in the method; the matrix is
determined by the transmissibilities alone and needs no fluid, state or solver.

**A bug in `useLog` is reproduced rather than fixed**, and named in both the
docstring and the parity ledger. Taking `log10` of SI transmissibilities gives
negative values, and `callMetisMatrix` takes their absolute value, so the weight
ordering inverts and METIS cuts the *strongest* connections first. On a 1000:1
channel the strongest connection is weighted 7960 against the weakest at 10000.
The port warns whenever every transmissibility is below 1, which is exactly when
this happens, and asserts both the inversion and the warning.

**Two vacuous assertions were caught while writing this.** Transmissibilities
are around 1e-14 in SI, and `np.isclose`'s default `atol` of 1e-8 calls any two
of them equal, so a harmonic-versus-arithmetic check and a row-sum check both
passed without testing anything. Every comparison in that file is now relative,
with `atol=0`. Worth carrying to any other module handling raw SI
transmissibilities or permeabilities.

## Phase 6: examples

MRST's seven `coarsegrid/examples` files, 1043 lines, then the two waiting in
the examples tree:

* `showSAIGUP` needs `processPartition`, `partitionLayers` and `explosionView`;
* `gridTutorialCornerPoint` needs the same three plus `generateCoarseGrid`,
  `outlineCoarseGrid` and `plotBlockAndNeighbors`, all already present.

`coarsegridExampleSAIGUP.m` (231) and `coarsegridExampleNorne.m` (160) both need
datasets: SAIGUP is not downloaded, Norne is.

### Phase 6 outcome

All seven are ported, at `pymrst/modules/coarsegrid/examples/`, with 73
self-tests between them. Norne runs on the real 44,915-cell model: 216 logical
slots become 168 non-empty blocks and then 223 after the faults are split,
against the 225 MRST's own histogram is scaled to.

MRST's examples are scripts whose output is a figure, so ported literally they
would assert nothing. Each one here states what its figure is meant to show and
checks that instead: `numbering` returns label positions rather than drawing
them, `face_detail` returns the fine faces and blocks the last SAIGUP figure
colours, and the two pressure solutions example 1 plots side by side are
compared numerically (they agree to 2e-13 bar).

**One piece of infrastructure was missing.** Example 1 solves a pressure problem
*on the coarse grid*, which MRST supports because its coarse grid keeps its
topology in the same fields a fine grid uses. `CoarseGrid` did not, so
`CoarseGrid.to_grid()` was added: a nodeless `Grid` carrying the coarse topology
and geometry, which `compute_trans` and `incomp_tpfa` accept unchanged. MRST's
coarse grid has no nodes either — a block is defined by the cells it holds, not
by a polygon.

Three things the examples caught that the module's own tests had not:

**`_tensor_coarse_grid` read logical indices from cell order**, not from
`indexMap`. On a grid with cells removed the two disagree, so it read the
partition at the wrong positions, and an empty logical slice made it raise
outright. Found by example 2's semicircular cutout.

**MRST fixes the merge threshold against the starting mean.** The SAIGUP
merging loop compares block volumes against `0.1 * meanVol` where `meanVol` is
computed once, before the loop. Recomputing it each pass makes it a moving
target: every merge raises the mean, so the loop keeps finding new blocks below
it. The first version of the port recomputed it.

**MRST's two merging loops are not equivalent**, even given the same neighbour
rule. The incremental one reads adjacency once and maps it forward through its
redirection table, but merging changes adjacency — a block that absorbs another
inherits its neighbours. On the test fixture they agree under one rule and
differ by a single block under the other. The port asserts the sharper property
instead: the incremental loop's running volumes must equal those of a coarse
grid rebuilt from its own answer.

**A fixture correction worth carrying.** A *faulted* grid of uniform cells does
not reproduce SAIGUP's sliver problem: `simple_grdecl` at several throws and
partitions gave a volume spread under 4x and no block below a tenth of the mean,
so the merging loop had nothing to do. What produces tiny blocks is cell volumes
varying by orders of magnitude, which is what graded stratigraphy gives. The
fixture now grades the layers geometrically and notches a corner, for an 80x
spread and 8 tiny blocks of 42.

## Effort summary

| Phase | Files | Lines | Blockers | Status |
|---|---|---|---|---|
| 1, relocate and consolidate | 3 impls | 252 | none | **done** |
| 2, combinatorial core | 9 + 3 private | 1140 | none | **done** |
| 3, coarsening layer | 7 + `refineNearWell` | 640 | Phase 2 | **done** |
| 4, plotting | 3 | 225 | Phase 2 | **done** |
| 5, METIS | 2 + 2 excluded | 489 | disposition | **done** |
| 6, examples | 7 + 2 | 1043 | Phases 2 to 4; SAIGUP download | **done** |

Phases 1 to 4 came to about 2265 lines with no external dependency, and finished
the examples tree as a by-product.

**Complete.** 34 of 37 files owned, 3 excluded with reasons, 323 self-tests
across 31 files. Three of those tests skip without the optional `pymetis`
package, and `showSAIGUP` still waits on a 2.5 MB download; both are recorded in
`coarsegrid_parity.md` rather than counted as done.

## A caveat carried from the examples port

The dependency figures here are static: they count files and lines, not
behaviour. The examples work produced four attribution errors in a row, every
one in the same direction, and one function that turned out to be fully ported
under another name in another module.

Two specific risks apply here. `pymrst/modules/upscaling/` and
`pymrst/modules/msrsb/` are both plausible homes for coarse-grid material that
this audit's name matching would miss, and the scan already found the two
`ported-elsewhere` files only because they kept MRST's names. Before porting any
file in Phases 2 and 3, grep for its behaviour rather than its name.

**The caveat earned its sixth entry.** The Phase 3 grep found no pre-existing
owner for any of the eight functions, which was the risk it was written for. It
did not catch the other kind: this document put `refineNearWell.m` in `utils/`
when it is a root file, and left `callMetisMatrix.m` out of the group entirely.
The check to add is to list the directory rather than trust the table, which
costs one command and would have caught it.
