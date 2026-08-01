# Gridprocessing end-to-end implementation plan

## Objective

Complete `pymrst/gridprocessing` against the MRST 2024a
`gridprocessing/` tree with the same standard used for PyMRST's AD-core:

- every relevant MRST production file is inventoried;
- every numerical API has a direct or clearly documented consolidated Python
  counterpart;
- MATLAB-only, deprecated, and compatibility-only files are explicitly
  classified rather than silently omitted;
- Python behavior is covered by focused unit and invariant tests;
- numerical behavior is validated against independently generated MATLAB/MRST
  reference data;
- realistic ECLIPSE grid-processing workflows run end to end;
- the parity audit reports zero pending and zero ported-but-unvalidated items.

The target MRST source tree is:

```text
mrst-2024a-Geomech/gridprocessing
```

The target PyMRST package is:

```text
PyMRST/pymrst/gridprocessing
```

## Current implementation status

The parity audit now inventories all 68 MATLAB files, including the complete
`testgrids/` and `deprecated/` directory layout:

- 50 have direct filename counterparts in PyMRST;
- 12 are consolidated into another documented Python implementation;
- all 15 named `testgrids` fixtures/examples have Python counterparts;
- `computeGeometryOrig` is available through an explicitly deprecated
  compatibility module backed by the maintained geometry kernel;
- the catalog, documentation-only, and legacy MATLAB-storage entries remain
  explicit exclusions;
- no entry is pending and no ported entry is unvalidated;
- 56 focused gridprocessing tests and 62 gridprocessing module self-tests
  pass.

This establishes file-layout, API, invariant, and saved-reference parity.
The final system-level gate remains the realistic deck and downstream
simulation comparison described in Phases 9 and 10.

## Definition of done

Gridprocessing is complete only when all the following are true:

1. A reproducible parity scanner inventories every MRST `.m` file.
2. Every file is classified as one of:
   - `ported`
   - `ported-consolidated`
   - `excluded-catalog`
   - `excluded-documentation`
   - `excluded-deprecated`
   - `excluded-matlab-compatibility`
   - `test-fixture`
3. The scanner reports no `pending-review` files.
4. Every ported numerical entry has direct test or validation evidence.
5. Public exports in `pymrst/gridprocessing/__init__.py` are intentional and
   documented.
6. All Python gridprocessing self-tests and focused pytest tests pass.
7. All saved MATLAB-reference validators pass.
8. At least one real deck containing faults, inactive cells, pinch-outs, and
   explicit NNCs can be processed in both MRST and PyMRST.
9. The processed grids agree on topology, geometry, mappings, tags, faults,
   and NNCs within documented tolerances.
10. A downstream transmissibility calculation and reservoir simulation run
    successfully on the processed PyMRST grid.

## Compatibility conventions

### Indexing

MRST uses one-based indices and `0` for a missing neighbor. PyMRST uses
zero-based indices and `-1` for a missing neighbor.

Every port must explicitly convert:

```text
MRST cell, face, node index: 1..N  -> PyMRST: 0..N-1
MRST missing neighbor: 0          -> PyMRST: -1
```

Raw deck Cartesian indices should retain their documented ECLIPSE/MRST
meaning until they cross the Python public API boundary.

### Array ordering

ECLIPSE and MATLAB grid arrays are interpreted in column-major/Fortran order.
Reshape and flatten operations involving `COORD`, `ZCORN`, `ACTNUM`, Cartesian
cell numbers, or region arrays must use `order="F"` where appropriate.

### Grid topology invariants

Every returned `Grid` must satisfy:

- `cells.facePos` has length `cells.num + 1`;
- `faces.nodePos` has length `faces.num + 1`;
- all referenced cells, faces, and nodes are in range;
- boundary faces have exactly one nonnegative neighbor;
- internal faces have two nonnegative, distinct neighbors;
- each cell-face incidence agrees with `faces.neighbors`;
- face node rings contain no invalid consecutive duplicates;
- geometry arrays, when present, match topology sizes;
- `cells.indexMap` maps active cells back to global Cartesian cells;
- direction tags retain their MRST semantics after zero-based conversion.

### Numerical tolerances

Validation should use quantity-specific tolerances:

- topology and mappings: exact equality;
- coordinates: approximately `1e-12` relative, with a scale-aware absolute
  tolerance;
- face areas and cell volumes: approximately `1e-10` relative;
- centroids and normals: approximately `1e-10` relative;
- transmissibilities: approximately `1e-9` relative;
- simulator states: model-specific tolerances established by the downstream
  validation case.

Tolerance relaxation must be justified by geometric degeneracy or a documented
algorithmic difference, not used to conceal indexing or topology errors.

## Phase 1: parity ledger and audit tooling

Create:

```text
scripts/audit_gridprocessing_parity.py
docs/gridprocessing/parity.csv
GRIDPROCESSING_PARITY.md
tests/gridprocessing/test_parity_audit.py
tests/gridprocessing/test_parity_ledger.py
```

The audit script should:

1. recursively inventory `mrst-2024a-Geomech/gridprocessing/**/*.m`;
2. convert MATLAB symbols to snake case;
3. find direct Python filenames;
4. find explicit consolidated symbol references;
5. scan focused tests and validation scripts independently;
6. assign port and validation status separately;
7. emit an authoritative CSV and human-readable Markdown summary;
8. fail CI if a previously classified file disappears or becomes pending.

The CSV should contain:

```text
mrst_path
symbol
subsystem
port_status
python_counterpart
port_evidence
validation_status
validation_files
notes
```

Explicit initial exclusions:

| MRST file | Classification | Reason |
|---|---|---|
| `Contents.m` files | `excluded-catalog` | MATLAB package catalogs |
| `deprecated/computeGeometryOrig.m` | `excluded-deprecated` | Superseded geometry implementation |
| `extended_grid_structure.m` | `excluded-documentation` | Structure documentation, not an algorithm |
| `removeIntGrid.m` | `excluded-matlab-compatibility` | Converts legacy MATLAB `int32` fields to doubles |
| `testgrids/*.m` | `test-fixture` | Reference grid generators, to be ported only when needed for validation |

Private helpers may be classified as consolidated only when the Python
counterpart genuinely performs the same operation and is tested through its
public consumer.

## Phase 2: foundational missing APIs

### `buildCornerPtPillars`

Create:

```text
pymrst/gridprocessing/build_corner_pt_pillars.py
tests/gridprocessing/test_build_corner_pt_pillars.py
examples/validation/dump_grid_build_corner_pt_pillars_reference.m
examples/validation/validate_grid_build_corner_pt_pillars.py
```

Implementation requirements:

- reshape `COORD` into one top/bottom line per pillar;
- return `X`, `Y`, and `Z` arrays with two columns;
- detect collapsed pillars using `coincidence_tolerance`;
- replace collapsed-pillar Z extents with the global `ZCORN` range;
- support MRST's `Scale` option;
- preserve pillar ordering exactly.

Validation cases:

- regular vertical pillars;
- sloping pillars;
- collapsed pillars;
- `Scale=False` and `Scale=True`;
- nonzero and negative depth coordinates.

### `cart2active`

Create:

```text
pymrst/gridprocessing/cart_to_active.py
tests/gridprocessing/test_cart_to_active.py
```

Implementation requirements:

- map global Cartesian cell numbers through `cells.indexMap`;
- preserve input ordering;
- omit inactive cells in the MRST-compatible mode;
- reject out-of-range and negative indices;
- expose clearly documented zero-based Python semantics;
- optionally provide a strict mode that returns `-1` for inactive cells when
  downstream callers require shape preservation.

Validate regular, inactive, reordered, empty, repeated, and invalid inputs.

### `checkAndRepairZCORN`

Create:

```text
pymrst/gridprocessing/check_and_repair_zcorn.py
tests/gridprocessing/test_check_and_repair_zcorn.py
examples/validation/dump_grid_repair_zcorn_reference.m
examples/validation/validate_grid_repair_zcorn.py
```

Implementation requirements:

- validate `cartDims` and `ZCORN` size;
- detect upper corners below the same cell's lower corners;
- detect overlap with the cell below on the same pillar;
- repair using the same assignment order as MRST;
- honor `ACTNUM`, so inactive cells do not create false repairs;
- leave already valid arrays bitwise unchanged when possible;
- return repair diagnostics for Python users without altering default
  MRST-compatible behavior.

Validation must include isolated inversion, vertical neighbor overlap, multiple
layers, inactive cells, and an already valid grid.

## Phase 3: fault processing

### `processFaults`

Create:

```text
pymrst/gridprocessing/process_faults.py
tests/gridprocessing/test_process_faults.py
examples/validation/dump_grid_faults_reference.m
examples/validation/validate_grid_faults.py
```

Define a typed Python result, for example:

```python
@dataclass
class GridFault:
    name: str
    faces: np.ndarray
    multiplier: float
```

Implementation requirements:

- accept the processed grid and ECLIPSE `FAULTS`/`MULTFLT` data;
- interpret I/J/K boxes and direction strings exactly;
- map Cartesian cells to active cells;
- use cell-face direction tags to locate fault faces;
- combine multiple records with the same fault name;
- apply case-insensitive fault-name matching consistent with MRST;
- apply `MULTFLT` values;
- remove empty or invalid fault entries using MRST's rules;
- return both the fault list and a stable name-to-index mapping.

### `removeFaultBdryFaces`

Implement in the same module or a dedicated file:

```text
pymrst/gridprocessing/remove_fault_boundary_faces.py
```

Requirements:

- remove fault faces that are external grid boundaries;
- update per-fault face counts;
- drop faults that become empty;
- return the keep mask corresponding to the input list;
- preserve fault order.

Fault validation cases:

- one I-direction fault;
- one J-direction fault;
- multiple named faults;
- repeated records;
- boundary-only faults;
- inactive cells intersecting a fault;
- missing and partial `MULTFLT`;
- finite, zero, and non-unit multipliers.

## Phase 4: PINCH and NNC processing

### `processPINCH`

Create:

```text
pymrst/gridprocessing/process_pinch.py
tests/gridprocessing/test_process_pinch.py
examples/validation/dump_grid_process_pinch_reference.m
examples/validation/validate_grid_process_pinch.py
```

Implementation requirements:

- require compatible `cartDims`, `ACTNUM`, and `PINCH`;
- identify inactive cells between active cells in each vertical column;
- enforce the PINCH thickness threshold;
- support MRST's documented `TOPBOT` transmissibility option;
- apply `MULTZ`/PINCH multipliers consistently;
- return active-cell pairs and corresponding geometric face pairs;
- exclude duplicates and already connected cell pairs;
- compose correctly with `process_nnc`;
- use zero-based active-cell indices in the Python result.

Validation cases:

- no inactive cells;
- one inactive cell between active layers;
- several consecutive inactive layers;
- thickness below, equal to, and above the threshold;
- inactive top or bottom cells with no valid bridge;
- multiple columns;
- `MULTZ`;
- coexistence with explicit `NNC`.

The end-to-end corner-point processor should optionally invoke this function
when the deck contains `PINCH`, matching MRST's public workflow.

## Phase 5: deck refinement

### `refineDeck`

Create:

```text
pymrst/gridprocessing/refine_deck.py
tests/gridprocessing/test_refine_deck.py
examples/validation/dump_grid_refine_deck_reference.m
examples/validation/validate_grid_refine_deck.py
```

Reuse `refine_grdecl` for the raw grid. The deck-level function must also:

- update `RUNSPEC.DIMENS` and `cartDims`;
- copy untouched deck sections without aliasing mutable input state;
- expand supported cell fields in Fortran order;
- refine `SOLUTION` fields such as pressure, saturation, RS, and RV;
- refine `REGIONS` fields such as SATNUM, PVTNUM, ROCKNUM, and EQLNUM;
- refine supported `PROPS` endpoint arrays;
- update well completion I/J/K ranges;
- update explicit WI and KH values according to refinement direction;
- support the `default_well` option;
- preserve schedule control structure and well ordering;
- validate dimensions of every expanded field.

Validation should use a small deck containing:

- heterogeneous permeability and porosity;
- multiple region arrays;
- initialized pressure and saturation;
- a vertical and a horizontal well;
- explicit and defaulted completion properties;
- nonuniform refinement factors such as `[2, 3, 2]`.

The refined deck should be processed and simulated, not merely checked for
array sizes.

## Phase 6: alternative corner-point processor

### `processGRDECLOMO`

Before implementing, make an explicit scope decision:

1. Port it as an independent alternative algorithm; or
2. classify it as a compatibility alias to `process_grdecl` after proving that
   the supported inputs and outputs are equivalent; or
3. exclude it with a documented reason if it is an obsolete experimental
   implementation.

Do not mark it complete based only on a name alias. If retained, validate it on:

- regular corner-point grids;
- sloping pillars;
- faults;
- inactive cells;
- pinched cells;
- geometry and topology agreement with MRST's own `processGRDECLOMO`.

## Phase 7: finish partial existing implementations

### `compute_grid_slice_polygons`

Add MRST option parity for:

- multi-segment polylines;
- bounding-box face preselection;
- node snapping;
- repeated-point handling;
- radius or ellipsoid clamping;
- multiple cut polygons per cell where applicable;
- deterministic polygon orientation and ordering;
- degenerate intersection diagnostics.

### `slice_grid`

Finish:

- cuts coincident with existing faces;
- zero-volume and near-zero-volume degeneracies;
- multiple cut components;
- MRST-compatible `gix` mapping for new/old cells, faces, and nodes;
- face-status values used by `markCutGrids`;
- property propagation for cell and face arrays.

### `markCutGrids`

Create:

```text
pymrst/gridprocessing/mark_cut_grids.py
```

Implement graph-based connected-component marking after excluding cut-plane
faces. Support the legacy start-cell behavior only if it provides compatibility
value; otherwise document it as an excluded slow MATLAB fallback.

### `glue_2d_grid`

Finish internal-boundary support:

- duplicated coincident boundary faces;
- orientation reversal;
- nonconformal node insertion;
- topology cleanup;
- geometry repair;
- explicit mapping of original to glued entities.

### Existing editing functions

Add direct MATLAB validation for:

- `extractSubgrid`
- `removeCells`
- `addFaces`
- `removeFaces`
- `emptyGrid`
- `makeInternalBoundary`
- `removeInternalBoundary`
- `splitDisconnectedGrid`

Round-trip properties should be tested where applicable:

```text
make internal boundary -> find matches -> remove boundary
add faces -> remove same faces
extract all cells -> topology-equivalent input grid
split components -> union of components preserves all input cells
```

## Phase 8: consolidated private helpers

Review and explicitly document mappings for:

| MRST private helper | Expected PyMRST location |
|---|---|
| `insertInPackedData` | `private/packed_data.py` |
| `removeFromPackedData` | `private/packed_data.py` |
| `copyFaces` | internal helper in `make_internal_boundary.py` |
| `sortEdges` | edge/cell-face ordering logic in `pebi.py` or shared helper |
| `extractGridInfoFromPolygons` | slicing polygon extraction pipeline |
| `computeCpGeometry` | existing PyMRST corner-point geometry utility |

Where two public algorithms need the same helper, move it into a clearly named
private Python module instead of maintaining duplicated implementations.

Each consolidated mapping must have:

- an explicit MRST symbol reference in its docstring;
- focused unit tests for the helper behavior;
- public-consumer validation showing it is integrated correctly.

## Phase 9: MATLAB reference suite

Create deterministic MATLAB dump scripts under:

```text
examples/validation/
```

Each dump script should:

1. add only the required MRST modules;
2. build a small deterministic grid or deck;
3. call the MRST function directly;
4. convert variable-length structures into portable numeric arrays;
5. save with `-v7`, not `-v7.3`;
6. include topology, mappings, tags, and geometry needed for a strict check;
7. print the output path and key sizes.

Each Python validator should:

1. rebuild the identical input independently;
2. call the Python implementation;
3. compare shapes before values;
4. compare integer topology exactly;
5. compare floating geometry with scale-aware tolerances;
6. print maximum absolute and relative error per field;
7. exit nonzero on failure.

Reference data should cover at minimum:

- Cartesian 1D, 2D, and 3D grids;
- nonuniform tensor grids;
- triangles, tetrahedra, hexahedra, and tessellations;
- PEBI grids;
- layered grids;
- regular and faulted corner-point grids;
- inactive cells;
- collapsed and repaired `ZCORN`;
- geometric pinch removal;
- deck `PINCH` NNC generation;
- explicit `NNC`;
- named faults and `MULTFLT`;
- subgrid extraction and cell removal;
- internal-boundary round trips;
- slicing and gluing;
- deck and GRDECL refinement.

## Phase 10: real-deck end-to-end validation

Build a compact deterministic ECLIPSE case containing:

- a 3D corner-point grid;
- sloping pillars;
- at least two named faults;
- fault transmissibility multipliers;
- inactive cells;
- a pinched layer;
- explicit NNC data;
- multiple rock/fluid regions;
- an injector and producer.

Run the following pipeline in both MRST and PyMRST:

```text
read deck
  -> repair ZCORN
  -> process GRDECL
  -> process faults
  -> process PINCH
  -> process explicit NNC
  -> compute geometry
  -> compute transmissibilities
  -> initialize state
  -> simulate schedule
```

Compare:

- active-cell index map;
- nodes and coordinates;
- face-node topology;
- face neighbors;
- cell-face topology and direction tags;
- centroids, normals, areas, and volumes;
- fault face lists and multipliers;
- PINCH and explicit NNC cell pairs;
- transmissibilities;
- initial pore volume;
- final pressure and saturation;
- well rates and BHP.

This case is the final integration gate. Unit-level parity is insufficient if
the processed grid cannot drive the reservoir simulator.

## Testing structure

Add a dedicated test package:

```text
tests/gridprocessing/
    test_parity_audit.py
    test_parity_ledger.py
    test_grid_invariants.py
    test_corner_point_helpers.py
    test_fault_processing.py
    test_pinch_processing.py
    test_refine_deck.py
    test_slicing.py
    test_grid_editing.py
    test_end_to_end_deck.py
```

Keep fast deterministic unit tests in pytest. Continue allowing module
`__main__` self-tests, but do not rely on symbol mentions or embedded tests as
the only evidence for parity.

Recommended verification commands:

```bash
PYTHONPATH=. pytest -q tests/gridprocessing
PYTHONPATH=. pytest -q tests/test_all.py -k gridprocessing
python3 scripts/audit_gridprocessing_parity.py
```

For MATLAB-enabled validation:

```bash
/Applications/MATLAB_R2021b.app/bin/matlab -batch \
  "run('examples/validation/dump_grid_<case>_reference.m')"

PYTHONPATH=. python3 examples/validation/validate_grid_<case>.py
```

## Implementation sequencing and commit boundaries

Use small, reviewable commits:

1. `add gridprocessing parity audit`
2. `port corner point pillar and active-cell helpers`
3. `port ZCORN validation and repair`
4. `port grid fault processing`
5. `port PINCH non-neighbor connections`
6. `port deck refinement`
7. `complete grid slicing and cut marking`
8. `complete 2D grid gluing parity`
9. `validate grid editing utilities against MRST`
10. `add real-deck gridprocessing integration case`
11. `close gridprocessing parity ledger`

Every implementation commit should include:

- the implementation;
- public exports;
- unit tests;
- MATLAB dump and Python validator when numerical;
- parity-ledger regeneration;
- focused and regression-test results.

## Risk areas

### Corner-point indexing

Most subtle errors will come from MATLAB Fortran ordering, repeated pillar
coordinates, direction tags, and active/global cell conversions. Small regular
grids are not enough; faulted and inactive cases must be validated field by
field.

### Degenerate geometry

Collapsed pillars, zero-thickness cells, coincident cut planes, short edges,
and repeated nodes need explicit policies. Avoid silently perturbing geometry
unless MRST performs the same repair.

### Mutable deck structures

Deck refinement must not mutate the caller's input unexpectedly. Nested
structures require controlled copying, and expanded fields must retain their
intended dtype and ordering.

### Fault and PINCH interaction

Fault faces, geometric pinch removal, `PINCH`-generated NNCs, and explicit
`NNC` data are related but distinct. They must not create duplicate
connections or discard multipliers.

### Validation false positives

A function name appearing in a test is not proof of numerical validation.
The parity ledger should distinguish:

- import/smoke coverage;
- invariant/unit coverage;
- saved MATLAB-reference coverage;
- end-to-end simulation coverage.

## Final completion report

When the work is complete, report:

- total MRST files inventoried;
- count per port/exclusion classification;
- zero pending items;
- zero unvalidated numerical ports;
- Python gridprocessing test count;
- MATLAB-reference validator count;
- real-deck topology and geometry maximum errors;
- downstream reservoir-state maximum errors;
- documented exclusions;
- final commit identifier.

Only after these gates pass should `pymrst/gridprocessing` be described as a
complete end-to-end MRST 2024a gridprocessing implementation.

## Combined fault/PINCH/NNC simulation gate

The runnable Python case
`pymrst/gridprocessing/examples/fault_pinch_nnc_waterflood.py` and its MRST
counterpart `gridprocessing/examples/faultPinchNNCWaterflood.m` exercise the
features together instead of validating them only in isolation. The case has
a 5-by-4-by-3 corner-point grid, two inactive cells, two named faults with
different multipliers, two PINCH connections, one explicit long-range NNC,
heterogeneous rock, a three-completion rate injector, and a three-completion
BHP producer.

Generate and validate the MRST reference with:

```bash
/Applications/MATLAB_R2021b.app/bin/matlab -batch \
  "cd('../mrst-2024a-Geomech'); startup; makePlots=false; \
   outputFile='/tmp/fault_pinch_nnc_mrst_results.mat'; \
   run('gridprocessing/examples/faultPinchNNCWaterflood.m')"

PYTHONPATH=. python3 scripts/validate_fault_pinch_nnc_waterflood.py \
  /tmp/fault_pinch_nnc_mrst_results.mat
```

The July 2026 validation produced exact active-cell maps, NNC cell pairs,
NNC transmissibilities, fault face counts, and fault multipliers. Across six
two-day fully implicit report steps, maximum absolute differences were
0.0659 Pa for reservoir pressure, 0.0756 Pa for well BHP, 9.84e-4 for water
saturation, and 1.83e-13 m3/s for phase well rates.

`slice_grid(..., return_slice=True)` also exposes MRST's third `sliceGrid`
output: a two-dimensional grid assembled from the generated slice faces. For
multiple cuts the returned grid contains every retained slice surface, while
an empty cut returns `None`.

The `only_faces=True` option matches MRST's `onlyFaces` topology: intersected
faces and crossing nodes are inserted without splitting cells or creating
slice faces. PyMRST intentionally corrects an MRST 2024a off-by-one in the
`old.faces` mapping (`cutFaceIx < nf_orig`), which otherwise leaves the final
original face marked as retained when it was split.

For staged workflows, pass a previous mapping through `gix_input`. The next
single or multi-cut call composes `old`, `new`, and `parent` arrays back to the
original grid, matching MRST's `gix_input` behavior.

`make_internal_boundary(..., tag=value)` preserves corner-point directional
half-face tags and existing retained face tags while applying `value` to both
replacement boundary faces. Duplicate inputs remain supported and map back to
the same replacement pair; already-exterior faces retain `NaN` mappings.

The inverse `remove_internal_boundary` restores each side's directional
half-face tag on the merged face, preserves Cartesian/cell/custom metadata,
and follows MRST by assigning zero to the newly merged face tag.

`remove_cells(..., return_mapping=True)` returns MRST-compatible original cell,
face, and node indices for every retained entity. Logical masks are accepted;
preserved corner-point nodes are renumbered, and NNC rows touching removed
cells are filtered while surviving connections are renumbered.

`extract_subgrid(..., return_mapping=True)` provides the same original entity
maps and stores them on the returned grid as `cells.global`, `faces.global`,
and `nodes.global` (with Python-friendly `global_` aliases). Logical masks,
corner-point nodes, NNCs, geometry, and custom grid metadata are preserved and
renumbered consistently.

`split_disconnected_grid` accepts MRST's `verbose` option and includes both
dictionary- and structure-style NNCs in its adjacency graph. Unlike MRST 2024a,
which drops `nnc` while extracting components, PyMRST retains only each
component's local connections and renumbers them with the subgrid.

The primary `process_grdecl` entry point follows MRST's option flow: it can
repair ZCORN, round by tolerance, attach PINCH-generated NNCs, validate the
grid, and split disconnected components by default. Set
`split_disconnected=False` for a single grid. The OMO compatibility entry point
retains its locally documented modified default of `False`.

`tensor_grid(..., cellnodes=True)` and `cart_grid(..., cellnodes=True)` expose
MRST's top-level `cellNodes` table with identical local-corner ordering in 1D,
2D, and 3D.

Non-monotone tensor-grid axes follow MRST's warning-and-truncation algorithm:
non-increasing entries are repeatedly discarded until each retained axis is
strictly increasing. An axis that cannot retain two coordinates raises a clear
error instead of constructing a zero-cell grid.

`compute_geometry(..., cp_geometry=True)` implements MRST's `CpGeometry`
option for grids carrying `cells.cpnodes`. It adds nominal corner-derived cell
centroids, directional extents, and one nominal face centroid per half-face.

`compute_geometry(..., find_neighbors=True)` reconstructs face adjacency from
cell-face incidence and orients each pair from face-node geometry, matching
MRST's `findNeighbors`/`findNormalDirections` path. Missing neighbor data also
triggers reconstruction automatically; non-manifold faces are rejected.

`compute_geometry(..., hinge_nodes={"faces": ..., "nodes": ...})` implements
MRST's 3D `hingenodes` option. Face indices are zero-based in Python and each
row of `nodes` supplies the triangulation center for the corresponding curved
or non-planar face. The substituted centers affect face area, normal, and
centroid as well as the derived cell volumes and centroids. Attribute-style
objects with `.faces` and `.nodes` are also accepted.

`slice_grid` now handles the two degeneracies where the cut plane coincides
with a face. A pinched cell whose collapsed face lies in the plane is split
normally: coincident cut-polygon corners are merged by coordinate so the
boundary edges still close into a ring, which is what previously failed and
raised `NotImplementedError`. A cut snapped onto an existing column of faces
splits nothing but marks those faces as slices (`gix.new.faces == 3`), so the
third output returns the cross-section grid, matching MRST.

Cell splitting follows MRST's `splitCells` acceptance test: a cell is divided
only when both halves keep at least `griddim` faces. A cell that fails is left
whole - if the odd half is a single in-plane face that face becomes the slice,
otherwise the cell is skipped with a warning, mirroring MRST's "Cutting of %d
out of %d cells failed". Faces are still split for every cell the plane fully
traverses, as MRST's `splitFaces` runs before the cell-split decision.

Against MRST, every face with area matches exactly. MRST additionally retains a
zero-area sliver face where it splits the collapsed face; PyMRST does not
create it, so the sliced grid has one fewer degenerate face and identical
geometry.

`add_faces` and `remove_faces` now carry a grid's optional data across instead
of rebuilding a topology-only grid. MRST's `addFaces`/`removeFaces` mutate `G`
in place, so everything they do not touch survives; the ports rebuilt `Grid`
from four fields each and silently dropped `cartDims`, face and cell tags, and
every geometry or custom field. Since `cart_grid` populates both `faces.tag`
and `cells.tags`, this affected ordinary grids, not just augmented ones.

`remove_faces` restricts each per-face array - `tag`, `areas`, `normals`,
`centroids`, `diameters` and the packed `edges`/`edgePos`/`edgeSign`/
`edgeNormals` set - to the surviving faces, and drops cell-face tags together
with their half-faces. `add_faces` takes MRST's optional `tags` argument for
the new half-faces (ordered like `neighbors` flattened column first, entries
for cell `-1` dropped), zero-fills `faces.tag` as MRST does, and keeps cell
tags aligned through `insert_in_packed_data_layout`.

Per-face geometry is the one thing `add_faces` cannot carry: no value exists
for an appended face, so `areas`, `normals`, `centroids`, `diameters` and the
edge fields are cleared rather than left short, and `compute_geometry` must be
rerun. Cell geometry is kept, as MRST keeps it, but is stale as soon as a cell
gains or loses a face - MRST's own note is that "cells may not be closed
polygons/polyhedra after this call".

`remove_pinch` and `remove_short_edges` now keep their metadata consistent
through a collapse. Both merge nodes, so `cells.cpnodes` - a corner-point node
map, attached dynamically by `process_grdecl_omo(preserve_cp_nodes=True)` - was
left addressing the pre-merge numbering: silently wrong, and out of range once
the node array shrank. Each merge map is now composed and applied through
`remap_cp_nodes`, including `remove_pinch`'s two-stage exact-then-tolerance
merge.

`cells.cpnodes` and `g.nnc` are attributes rather than declared dataclass
fields, so any rebuild of the grid drops them. `carry_custom_fields` copies a
grid's undeclared attributes onto a rebuilt one, and `add_faces`/`remove_faces`
now call it - both leave nodes and cells alone, so cpnodes and NNCs carry over
unchanged. NNCs that reference a cell the collapse removed are still filtered
out by `remove_cells`, which owns that mapping.

Geometry is cleared when a collapse changes the cell, face or node count.
Merging nodes moves them, so incoming `faces.areas`/`normals`/`centroids` and
`cells.volumes`/`centroids` describe a shape that no longer exists - and
because the arrays keep a valid length, nothing downstream errors on them. A
grid with nothing to collapse passes through with its geometry intact.

Both operations mutate the grid passed in, as their MRST originals do; the
returned grid is the one to use.

`compute_geometry(..., max_block_size=..., verbose=...)` implements MRST's
`MaxBlockSize` and `verbose` options. Both are operational: they change memory
use and reporting, never the numbers.

`max_block_size` caps how many cells the 3D cell pass handles at once. That
pass allocates one row per cell-face triangle - the largest temporary in the
routine - so bounding it bounds peak memory. `None` or a negative value
processes every cell in one pass; the default is MRST's 20000, which runs
ordinary grids in a single pass. `_partition_cells` ports MRST's
`partition_cells`: the requested size is first reduced to the largest that
divides the work evenly, then the remainder is spread one cell at a time over
the leading blocks, so no two blocks differ by more than one cell. The block
boundaries match MRST's for every case checked.

Measured on a 40x40x25 grid (40000 cells), peak allocation falls from 215 MB
in a single pass to 111 MB at `max_block_size=5000`, with no runtime penalty
and bit-identical results. Below that the saving plateaus, because the
face-level sub-triangle arrays are sized by the whole grid - as they are in
MRST, which preallocates them in full - and then dominate.

`verbose=True` prints per-block progress and rates for 3D grids and a total,
and the two phase timings for 2D, mirroring MRST's `dispif`/`ticif` output.
Cell ranges are reported zero-based, matching the rest of the Python API.

`examples/validation/stress_gridprocessing.py` exercises the stack at scale on
the four fronts the parity ledger left open: a faulted corner-point deck, a
deck that falls into many disconnected regions, a dense NNC network, and
repeated slicing. Every grid is put through one structural checker - closed
topology, no orphan faces, half-face incidence, positive finite volumes,
conserved total volume, and cells that close (the sum of outward face normals
must vanish). Failures name the offending entity, so a 200000-cell grid is
diagnosable. Sizes are `small` (what the test suite runs), `medium` and
`large`.

At `large` the suite handles a 200000-cell faulted deck in 1.9s and 633 MB, a
lattice of 289 disconnected components, and ~40000 requested NNCs, with cell
closure residuals at 1e-15. `tests/gridprocessing/test_stress_invariants.py`
runs the `small` configuration on every commit, and also feeds the checker
deliberately broken grids, since a checker that cannot fail proves nothing.
Timing and peak memory are measured in separate passes (`--memory` adds the
second): `tracemalloc` inflates allocation-heavy code several fold, so a run
that traced and timed at once reported neither honestly.

`validate_grid_large_deck.py` supplies the parity half: the stress harness only
shows a large grid is self-consistent, so a two-tier MRST reference pins it to
MRST's own output. Full sorted volumes and areas at 40x40x10, and counts plus a
21-point fingerprint of the sorted distributions at 60x60x20 (72000 cells) -
the fingerprint keeps the stored reference at 11 kB. Component counts and sizes
for the disconnected-lattice deck are pinned too. Everything agrees to 1e-14
relative or better.

The stress work drove the vectorization pass documented below. It also
showed why the harness must not time and trace at once: the first runs
attributed 93s to repeated slicing, nearly all of it `tracemalloc` overhead
rather than real work.

## Vectorization

Profiling the stress scenarios turned up one algorithmic defect and several
per-face Python loops. Every change below was checked to reproduce the previous
implementation exactly, not merely to pass the suite.

`compute_geometry(find_neighbors=True)` was quadratic in the face count. Its
reconstruction of `faces.neighbors` scanned the whole sorted half-face array
once per face to find that face's incident cells, so doubling the faces
quadrupled the work. Half-faces are now scattered into their face's free slot
using `bincount` and a within-group offset, which is linear. On a 72000-cell
faulted deck that is 20.3s to 0.92s; the 200000-cell deck went from an
extrapolated ~160s to 0.71s. This path runs automatically whenever
`faces.neighbors` is absent, so the cost was easy to hit without asking for it.

`triangulate_faces` built its triangle fan a node at a time; the fan is now
formed with a successor-index gather, 4x faster. `_face_edges` in
`remove_short_edges` used the same per-face ring loop and is now essentially
free. `remove_pinch`'s `_dedupe_face_nodes` compared each node with its cyclic
predecessor in Python and now does it as one masked comparison. In `slice_grid`,
`_rebuild_cell_faces` walked the neighbor table row by row, the per-face side
tests ran once per face per pass, and untouched faces had their node lists
converted to Python lists and back; faces the cut does not divide are now
handed through as array slices and concatenated once.

Measured on a 30x30x12 grid: `find_neighbors` 0.467s to 0.044s,
`triangulate_faces` 0.186s to 0.047s, `slice_grid` 0.236s to 0.127s,
`remove_short_edges` 0.081s to 0.059s, `remove_pinch` 0.074s to 0.058s.
`compute_geometry` itself was already vectorized and is unchanged at 0.23s for
72000 cells.

The main loop in `_slice_grid_single` remains, and is the largest Python loop
left in the module. It builds ragged per-face node lists for the faces a cut
divides, and restructuring it would put the degeneracy handling that
`validate_grid_slice_degeneracy.py` pins at risk for a modest gain now that
untouched faces bypass it.
