# coarsegrid parity ledger

Parity status for MRST's `modules/coarsegrid`, which builds coarse grids from
partitions of a fine grid and is the basis for upscaling and multiscale methods.

## Scope

`mrst-2024a-Geomech/modules/coarsegrid` holds 43 `.m` files, of which 6 are
`Contents.m` catalogs and 37 are executable, totalling 3971 lines.

Two of MRST's coarse-grid plotting routines live outside the module, in the core
`plotting/` directory, and are tracked here because the module's examples need
them: `outlineCoarseGrid` and `plotBlockAndNeighbors`. Both are already ported.

## Current scan

| Measure | Count |
|---|---|
| MRST files in `modules/coarsegrid` | 43 |
| `Contents.m` catalogs | 6 |
| Executable files | 37 |
| Executable lines | 3971 |
| **With a Python owner** | **34** |
| Excluded, recorded | 3 |
| Missing owners | 0 |
| Missing lines | 0 |

**All six phases of `coarsegrid_implementation.md` are complete.** Every
executable file has a Python owner at an MRST-shaped path or a recorded
exclusion. The module lives at `pymrst/modules/coarsegrid/` and carries 31 files
and 323 self-tests, 3 of which skip without the optional `pymetis` package:

| MRST | PyMRST | Tests |
|---|---|---|
| `generateCoarseGrid.m` | `generate_coarse_grid.py` | 19 |
| `partitionCartGrid.m` | `partition_cart_grid.py` | 8 |
| `compressPartition.m` | `compress_partition.py` | 9 |
| `partitionUI.m` | `partition_ui.py` | 10 |
| `partitionLayers.m` | `partition_layers.py` | 9 |
| `processPartition.m` | `process_partition.py` | 10 |
| `cellPartitionToFacePartition.m` | `cell_partition_to_face_partition.py` | 9 |
| `processFacePartition.m` | `process_face_partition.py` | 10 |
| `partitionTensor.m` | `partition_tensor.py` | 11 |
| `sortSubEdges.m` | `sort_sub_edges.py` | 10 |
| `private/sortedges.m` | `private/sort_edges.py` | 12 |
| `partitionMETIS.m` | `partition_metis.py` | 13 |
| `refineNearWell.m` | `refine_near_well.py` | 14 |
| `utils/coarsenGeometry.m` | `utils/coarsen_geometry.py` | 11 |
| `utils/coarseDataToFine.m` | `utils/coarse_data_to_fine.py` | 10 |
| `utils/invertPartition.m` | `utils/invert_partition.py` | 10 |
| `utils/isSamePartition.m` | `utils/is_same_partition.py` | 10 |
| `utils/coarsenBC.m` | `utils/coarsen_bc.py` | 9 |
| `utils/fineToCoarseSign.m` | `utils/fine_to_coarse_sign.py` | 7 |
| `utils/coarsenFlux.m` | `utils/coarsen_flux.py` | 7 |
| `utils/callMetisMatrix.m` | `utils/call_metis_matrix.py` | 9 |
| `plotting/explosionView.m` | `plotting/explosion_view.py` | 12 |
| `plotting/plotCellNumbers.m` | `plotting/plot_cell_numbers.py` | 10 |
| `plotting/plotFaceNumbers.m` | `plotting/plot_face_numbers.py` | 11 |
| `examples/coarsegridExample1.m` | `examples/coarsegrid_example1.py` | 8 |
| `examples/coarsegridExample2.m` | `examples/coarsegrid_example2.py` | 12 |
| `examples/coarsegridExample3.m` | `examples/coarsegrid_example3.py` | 11 |
| `examples/coarsegridExample4.m` | `examples/coarsegrid_example4.py` | 10 |
| `examples/coarsegridExample5.m` | `examples/coarsegrid_example5.py` | 10 |
| `examples/coarsegridExampleNorne.m` | `examples/coarsegrid_example_norne.py` | 10 |
| `examples/coarsegridExampleSAIGUP.m` | `examples/coarsegrid_example_saigup.py` | 12 |

Three more MRST files have owners without a file of their own, because MRST
calls each from exactly one place and the port folded them in there:
`private/coarseConnections.m` and `private/transposeConnections.m` are
`generate_coarse_grid._coarse_topology`, and `subFaces.m` is
`CoarseGrid.sub_faces`. Status `consolidated`, on the same basis as the three
merged `CoarseGrid` implementations.

**A location correction.** The implementation plan placed `refineNearWell.m`
under `utils/`. It is a root file, and the eighth file in `utils/` is
`callMetisMatrix.m`, which belongs to the METIS group below. That is the
sixth attribution error of this kind in this port, all in the same direction:
the plan asserted a location or a dependency without checking it. `refineNearWell`
is ported either way, at `pymrst/modules/coarsegrid/refine_near_well.py`.

The three duplicate implementations are merged, and every path that previously
exposed a `CoarseGrid` now exposes the same class: nine modules shared two
incompatible classes before, and share one now. The two pre-existing files had
no self-tests at all; both now assert the invariants below, including volume
conservation across the fine-to-coarse mapping.

`partitionLayers`, `processPartition` and `explosionView` are all three of the
functions the examples tree waited on, so `gridTutorialCornerPoint` is now
ported.

**`generate_coarse_grid` now handles any partition.** It previously formed the
coarse grid as a tensor product of coordinate lines, accepted only rectangular
Cartesian partitions, and asserted on anything else, so no coarse grid could be
built for a faulted model at all. It now builds MRST's coarse topology directly:
`faces.neighbors` giving the block pair of each coarse face, `connPos`/`fconn`
listing the fine faces behind it, and `facePos`/`faces` mapping blocks to their
coarse faces. The corner-point tutorial completes as a result, with 64 blocks
and 252 coarse faces.

The geometric grid `.g` is kept as an addition rather than a port, populated
only when the partition is a tensor product, because nine call sites use it for
block geometry. It is `None` otherwise, and the topology is complete either way.

`subFaces` is provided as `CoarseGrid.sub_faces`, since MRST's version only
reads `fconn`/`connPos` off the coarse grid.

**A coarse face is not a block pair.** The first version of the topology
grouped fine faces by block pair alone. MRST groups by the pair *and* an
indicator, and its default indicator already carries the half-face tag of every
exterior face, so a block's boundary is one coarse face per direction it faces.
On the 6x4x2 case in 3x2x1 blocks that is the difference between 13 coarse
faces and MATLAB's 29. Nothing raised and no invariant failed: boundary
conditions would simply have been applied to the wrong connections. The
indicator is now built, `generate_coarse_grid` takes MRST's optional `pf`
argument as a further column, and `face_indicator` records the result, which is
MRST's `ind` and the second column of its `cells.faces`.

`processFacePartition` is ported as `process_face_partition`. It is **2D only**,
as MRST states and enforces: in two dimensions a face is a segment and
"connected" means sharing an endpoint, whereas in three dimensions adjacency
runs along shared polygon edges, a construction MRST has not written. The port
refuses a 3D grid rather than inventing a rule.

Two implementation notes. `lbLinDist`, the load-balanced distribution that
spreads a remainder over the leading blocks, is duplicated as a private function
in both `partitionUI.m` and `partitionLayers.m`; the port defines it once in
`partition_ui` and imports it. And `processPartition` uses
`scipy.sparse.csgraph.connected_components` over one graph for the whole grid
where MRST loops over blocks calling `dmperm`, which is the same computation
done once rather than per block.

## Status by area

| Area | Files | Lines | Owners | Note |
|---|---|---|---|---|
| root | 14 | 1453 | **13** | complete; `mexPartitionMETIS` excluded |
| `utils/` | 8 | 640 | **8** | complete |
| `private/` | 4 | 436 | **3** | complete; the MEX implementation excluded |
| `plotting/` | 3 | 225 | **3** | complete |
| `plotting/tests/` | 1 | 174 | — | MRST's own scaffold; **excluded**, replaced by per-file self-tests |
| `examples/` | 7 | 1043 | **7** | complete; Norne runs on the real model, SAIGUP's algorithm on a synthetic one |

## Where the existing owners sit

Both ported files are under `pymrst/modules/upscaling/coarsegrid/` rather than
at an MRST-shaped `pymrst/modules/coarsegrid/`:

| MRST | PyMRST | Status |
|---|---|---|
| `generateCoarseGrid.m` | `modules/upscaling/coarsegrid/generate_coarse_grid.py` | ported-elsewhere |
| `partitionCartGrid.m` | `modules/upscaling/coarsegrid/partition_cart_grid.py` | ported-elsewhere |

That is a layout deviation of the same class the `utils` port drove to zero, and
it should be resolved when the module is built out: either relocate both to
`pymrst/modules/coarsegrid/` and leave a re-export under `upscaling`, or record
the nesting as deliberate. Relocating matches precedent.

**There is a third copy.** `pymrst/modules/upscaling/upscale.py` defines its own
`partition_cart_grid` and `generate_coarse_grid`, importing neither of the
above, and carries a second `CoarseGrid` class. Checked directly, the two
implementations agree: identical partitions, identical field names, identical
values on a 6x6x4 case partitioned 3x3x2. So consolidating them is safe.

The duplication still matters, because the two `CoarseGrid` classes are
different objects. Any `isinstance` check or type annotation written against
one silently rejects the other, and this codebase has already lost time to that
exact failure: `phase_stability_test` carries a comment explaining why it
checks a mixture by attribute rather than by `isinstance`, after class identity
made `eos.py` fail its own tests while working on import.

Consolidation is therefore Phase 1 work, and it is merging three
implementations across two classes rather than moving two files.

**Done.** `pymrst/modules/coarsegrid/` is the single implementation.
`modules/upscaling/coarsegrid` re-exports it and `upscale.py` imports it, so
every existing caller keeps working and all nine modules now share one
`CoarseGrid`.

The re-export needed care. That package contains submodules named
`generate_coarse_grid` and `partition_cart_grid`, the same names as the
functions in them, so importing a submodule binds it over the re-exported
function and `from ...upscaling.coarsegrid import generate_coarse_grid` returns
the *module*. Eight plotting tests failed with
`TypeError: 'module' object is not callable` until the `__init__` was made to
import the submodules first and rebind the function names afterwards. This is
the third appearance of that trap in this codebase, after
`phase_stability_test` and `convert_rs_rv_to_k_value`.

Related material already present and correctly placed:

| MRST | PyMRST |
|---|---|
| `plotting/outlineCoarseGrid.m` | `pymrst/plotting/outline_coarse_grid.py` |
| `plotting/plotBlockAndNeighbors.m` | `pymrst/plotting/plot_block_and_neighbors.py` |
| `utils/isCoarseGrid` equivalent | `pymrst/utils/is_coarse_grid.py` |

## The METIS question

Four files depend on METIS, a C graph-partitioning library MRST calls through a
MEX interface:

| MRST file | Lines |
|---|---|
| `partitionMETIS.m` | 111 |
| `mexPartitionMETIS.m` | 99 |
| `private/mexPartitionMETIS_Impl.m` | 115 |
| `utils/callMetisMatrix.m` | 164 |

489 lines, an eighth of the module, resting on an external native dependency.
Three dispositions were available and the choice should be recorded rather than
made silently:

1. port against `pymetis` or `networkx-metis` as an optional extra, mirroring
   how `pyamg` and `torch` are handled;
2. substitute `scipy.sparse.csgraph` or `networkx` spectral partitioning, which
   changes the partitions produced and therefore breaks numerical parity;
3. exclude with a stated reason, as the MATLAB BGL bindings were excluded from
   `utils`.

**Decision: option 1.** `metis = ["pymetis>=2023.1"]` is registered in
`pyproject.toml` alongside `amg`, `plotting`, `ml` and `mat73`. Option 2 was
rejected because a different partitioner gives different partitions, which is
the opposite of parity; option 3 because nothing else in the module needs to be
excluded.

Two of the four files are ported:

| MRST | PyMRST | Note |
|---|---|---|
| `partitionMETIS.m` | `partition_metis.py` | connection matrix and the METIS call |
| `utils/callMetisMatrix.m` | `utils/call_metis_matrix.py` | graph construction and weights |
| `mexPartitionMETIS.m` | — | a MEX wrapper with no Python counterpart |
| `private/mexPartitionMETIS_Impl.m` | — | likewise |

The two MEX files exist to reach the same library through MATLAB's compiled
interface, which `pymetis` does directly. They are recorded as
**excluded-not-applicable** rather than missing.

MRST shells out to the `gpmetis` binary through a temporary file;
`call_metis_matrix` calls the library in process. The graph construction is the
substantive half and is fully covered: the off-diagonal pattern, the
`ceil(1e4 * |v| / max|v|)` integer weight scaling, and the symmetry METIS
requires.

**The partitioning call itself is unexercised.** `pymetis` is not installed
here, and neither is the `gpmetis` binary MATLAB would need, so no reference
partition can be generated on either side. The self-tests covering that one call
print `SKIP` rather than passing quietly. Installing `pip install pymrst[metis]`
is what closes it.

### A bug in `use_log` worth naming

`partitionMETIS`'s `useLog` option takes `log10` of the transmissibilities
before building the graph, to compress a contrast spanning orders of magnitude.
A transmissibility in SI is around 1e-14, so its logarithm is negative, and
`callMetisMatrix` takes the absolute value before scaling to integers. The
ordering therefore inverts: a strong connection at 1e-11 becomes 11 and a weak
one at 1e-14 becomes 14, so the *strong* connection gets the smaller weight and
is cut first.

Measured on a 6x6 grid with a 1000:1 channel: the strongest connection is
weighted 7960 where the weakest gets 10000. The resulting partition follows the
low-permeability rock, and nothing about it looks wrong.

The port reproduces this rather than correcting it, because a different
weighting is a different partition. It warns when every transmissibility is
below 1, which is exactly when the inversion occurs, and the docstring says how
to avoid it. Both the inversion and the warning are asserted, so neither can
quietly stop being true.

## What the examples tree needs

`examples_parity.md` records two files waiting on this module. Their exact
requirement is smaller than the module:

| Function | Lines | Status |
|---|---|---|
| `processPartition` | 117 | missing |
| `partitionLayers` | 100 | missing |
| `plotting/explosionView` | 102 | missing |
| `generateCoarseGrid` | 194 | ported |
| `plotting/outlineCoarseGrid` | 101 | ported |
| `plotting/plotBlockAndNeighbors` | 204 | ported |

319 lines would unblock both examples. The decision taken was to port the module
properly rather than stop there, so those three are early targets rather than
the whole job.

## Validation approach

Coarse grids are combinatorial, so most of this module validates exactly rather
than to a tolerance, which makes it unusually well suited to assertions:

* a partition covers every fine cell exactly once;
* block counts and the coarse cell count agree;
* `compressPartition` leaves no empty blocks and preserves membership;
* `processPartition` splits exactly the disconnected blocks and no others;
* coarse face area equals the summed area of its fine faces;
* coarse block volume equals the summed volume of its fine cells;
* `invertPartition` round-trips;
* `coarseDataToFine` followed by averaging recovers the coarse values.

MATLAB validators are worth writing for `generateCoarseGrid`,
`processPartition` and `coarsenGeometry`, where the fine-to-coarse mappings are
easy to get subtly wrong and hard to see.

### Written

`examples/validation/dump_coarsegrid_faces_reference.m` and
`validate_coarsegrid_faces.py` cover the coarse face topology in five cases:

| Case | What | Result |
|---|---|---|
| A | 3D, default indicator, 6x4x2 in 3x2x1 blocks | 29 coarse faces, every array identical |
| B | the same in 2D, 4x4 in 2x2 | 12 coarse faces, every array identical |
| C | `showCoarseGrid.m`'s stripe partition through `processFacePartition` | 32 coarse faces, same grouping |
| D | 3D face partition from a finer partition, on a grid with cells removed | 88 coarse faces, every array identical |
| E | `processFacePartition` alone | 8 ids, same grouping |

A, B and D are compared array for array and index for index:
`faces.neighbors`, `faces.connPos`, `faces.fconn`, `cells.facePos`,
`cells.faces` and its tag column. That required matching MRST's orderings
exactly, both `coarseConnections` sorting on (block pair, indicator, fine face)
and `transposeConnections` sorting on (block, coarse face).

C and E compare the *grouping* rather than the labels, and the reason is worth
recording. MATLAB finds the connected components with `dmperm` and this port
uses `scipy.sparse.csgraph`; neither promises a particular numbering, so a
label comparison would fail on a correct port. The equivalence relation is what
carries meaning, and it is checked as a set of (block pair, fine-face set)
tuples. Counting groups instead would have been the easy check and would pass
on a wrong port.

`dump_coarsening_reference.m` and `validate_coarsening.py` cover the coarsening
layer, in four cases:

| Case | What | Result |
|---|---|---|
| A | uniform Cartesian, 6x4x2 in 3x2x1 | geometry, sign, flux and `invertPartition` all match |
| B | graded tensor grid | volume-weighted centroids match |
| C | an L-shaped block | one folded coarse face; areas match |
| D | `refineNearWell`, log and linear rings, masked far field | ring indices exact, groupings and mask match |

Geometry is compared at `rtol=1e-12`, since the two sides accumulate the same
sums in different orders; everything else is exact.

Case B exists because a uniform grid does not distinguish a volume-weighted
block centroid from a plain mean of cell centroids. The validator asserts that
the two differ on that case, so it cannot silently stop testing what it is for.
Case C does the same for `coarsenGeometry`'s area rule, which takes the length
of the summed normal rather than the summed area: the two agree on every flat
coarse face, so the case asserts that at least one face folds.

`refineNearWell` compares by grouping for the same reason as above, since MRST
numbers blocks from 1 and marks the masked far field 0 while this port numbers
from 0 and marks -1. Ring indices and the mask itself are compared exactly.

`dump_coarsegrid_sorting_reference.m` and `validate_coarsegrid_sorting.py` cover
`partitionTensor` and `sortSubEdges`:

| Case | What | Result |
|---|---|---|
| A | `partitionTensor` in 3D, uneven/even/unit widths | groupings and block counts match |
| B | the same on a grid with cells removed | grouping matches, including the block the hole splits |
| C | `partitionTensor` in 2D | grouping matches |
| D | `sortSubEdges` on 6x6 in 2x2 blocks | `fconn` and `cells.faces` match element for element, before and after |

`sortSubEdges` produces an *order*, so it is compared exactly rather than by
grouping, and case D also asserts that the sort reordered something: a validator
that passes because nothing moved is worse than none.

**Matching it exactly took one correction.** MRST grows the chain forward and
backward *interleaved*, offering each candidate edge to the forward end first
and the backward end second. The port originally ran the forward walk to
exhaustion first. On an open chain the two agree. On a closed cycle, which is
what a block's boundary is, running forward to exhaustion consumes the whole
loop and leaves the seed edge first, while the interleaved walk meets in the
middle and leaves the seed partway along. Both are correct cycles and both pass
every invariant test written for them; only MATLAB distinguishes them.

## Completion declaration rule

Do not mark this module complete until all 37 executable files have Python
owners at MRST-shaped paths or formally recorded exclusions, the METIS
disposition above is chosen and recorded, the two `ported-elsewhere` entries are
resolved, every ported file carries `__main__` self-tests, the combinatorial
invariants above are asserted rather than assumed, and `showSAIGUP` and
`gridTutorialCornerPoint` run end to end.

Against that rule, with all six phases done:

| Condition | Status |
|---|---|
| 37 files owned or excluded | **yes** — 34 owned, 3 excluded with reasons |
| METIS disposition recorded | **yes** — optional `pymetis` extra, ported |
| `ported-elsewhere` entries resolved | **yes** — one implementation, re-exported |
| every ported file has self-tests | **yes** — 323 across 31 files |
| combinatorial invariants asserted | **yes** — and MATLAB-validated in 13 cases |
| `gridTutorialCornerPoint` runs | **yes** — 64 blocks, 252 coarse faces |
| `showSAIGUP` runs | **no** — waiting on a 2.5 MB dataset, not on this module |

Six of seven conditions are met. The module is **complete as a port**; what
stands between it and an unqualified declaration is a download, not code.

Two qualifications, both stated rather than buried:

**The METIS partitioning call is unexercised.** Neither `pymetis` nor MATLAB's
`gpmetis` binary is installed here, so no reference partition exists on either
side. The three tests covering that one call print `SKIP`. Everything up to it —
the connection matrix, the weighted graph, the integer scaling — is asserted.

**SAIGUP's example runs on a synthetic model.** Its dataset is not installed, so
`coarsegrid_example_saigup.py` exercises the merging algorithm, which is the
substance of the script, on a graded model built for the purpose. The loaders
return `None` and say so. Norne, by contrast, runs on the real 44,915-cell
model.

The three exclusions:

| MRST | Reason |
|---|---|
| `mexPartitionMETIS.m` | a MEX wrapper; `pymetis` reaches the library directly |
| `private/mexPartitionMETIS_Impl.m` | likewise |
| `plotting/tests/test_newplotting.m` | MRST's own test scaffold; PyMRST's equivalent is the `__main__` self-tests in each file |
