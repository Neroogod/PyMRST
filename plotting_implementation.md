# Finishing `pymrst/plotting`

What remains for `pymrst/plotting` to match MRST 2024a's top-level
`plotting/` folder end to end, while following the same layout.

This document is based on the local trees:

- MRST: `mrst-2024a-Geomech/plotting`
- PyMRST: `PyMRST/pymrst/plotting`

It audits behavior, not filenames. A Python function is not considered
complete merely because a similarly named renderer exists.

## Scope and current position

MRST's folder contains 23 files:

| Kind | Count |
|---|---:|
| Public executable `.m` files | 18 |
| Private executable `.m` files | 2 |
| Catalog/readme files | 3 |
| **Total** | **23** |

The original PyMRST plotting package contained one MRST-like grid plotter and
six PyMRST-specific renderer/workflow files. The latter now live under
`plotting/extensions/`, leaving the package root for the MRST-shaped core.

The toolbar, browser, saved-state, streamline, trapping-inventory, and VE
plotters are useful extensions. They are not counterparts of files in MRST's
top-level `plotting/` folder and therefore do not close MRST parity entries.

The honest executable parity snapshot is:

| Status | Count | Entries |
|---|---:|---|
| Implemented elsewhere | 1 | `boundaryFaces` |
| Partial | 3 | `plotGrid`, `plotCellData`, `plotWell` |
| Missing | 16 | Remaining 14 public functions and 2 private helpers |
| **Total executable entries** | **20** | |

`pymrst/plotting/plot_grid.py` says that it ports `plotFaces`, but it does not
export a `plot_faces` function or provide the face-selection, face-data,
outline, coarse-grid, and 2D line behavior of MRST's `plotFaces.m`.
`plotFaces` is consequently classified as missing, not partial.

## Required target layout

The finished core should mirror MRST's layout using Python naming conventions:

```text
pymrst/plotting/
├── __init__.py
├── README.md
├── contents.py
├── boundary_faces.py
├── colorbar_hist.py
├── interpolate_data.py
├── mrst_colorbar.py
├── outline_coarse_grid.py
├── plot_block_and_neighbors.py
├── plot_cell_data.py
├── plot_contours.py
├── plot_face_data.py
├── plot_faces.py
├── plot_faults.py
├── plot_grid.py
├── plot_grid_volumes.py
├── plot_node_data.py
├── plot_slice.py
├── plot_well.py
├── unstructured_contour.py
├── unstructured_surf.py
└── private/
    ├── __init__.py
    ├── plot_face_outline.py
    └── plot_patches.py
```

The existing PyMRST-only features should remain available, but should no
longer obscure the MRST-shaped core. The clean final organization is:

```text
pymrst/plotting/extensions/
├── browser.py
├── plot_inventory.py
├── plot_saved_states.py
├── plot_streamlines.py
├── plot_toolbar.py
└── plot_ve.py
```

Moving the extension files must be done with compatibility re-exports so
existing imports such as `from pymrst.plotting import plot_toolbar` continue
to work. The implementations are moved directly: no duplicate modules or
thin wrappers remain at the old paths. Code that imported an implementation
module directly must use `pymrst.plotting.extensions.<module>`.

## Backend contract

MRST plotting functions produce MATLAB patch, line, contour, axes, and
colorbar handles. A literal handle port is neither possible nor desirable.
PyMRST needs one consistent Python contract:

1. **Geometry first.** Private helpers build backend-neutral vertices,
   connectivity, entity indices, and scalar/color arrays.
2. **PyVista for grid geometry.** Polyhedral grids, faces, cells, wells,
   faults, volumes, and interactive 3D rendering use PyVista.
3. **Matplotlib for axes-native plots.** Histograms, histogram colorbars,
   contours, and structured 2D surfaces use Matplotlib where its API is the
   closest match.
4. **Explicit renderer injection.** Public functions accept `plotter=` or
   `axes=` as appropriate and add to the supplied target.
5. **Headless operation.** Every public renderer can be tested without a
   display and can save a deterministic PNG.
6. **Useful return objects.** Return a small result object containing the
   backend handle plus selected entity indices and generated geometry. Tests
   must not have to reverse-engineer VTK objects to learn which faces were
   plotted.
7. **Optional dependencies stay optional.** Importing `pymrst.plotting`
   must not fail when PyVista, Plotly, or Matplotlib is absent. Backend imports
   should be lazy and missing extras should produce an actionable error such
   as `pip install pymrst[plotting]`.

The existing `to_pyvista` converter is valuable and should be retained as a
public extension. Its cell-polyhedron logic should be reused by the new
MRST-shaped files rather than copied.

## Per-file parity ledger

### Foundation and private rendering

| MRST file | Current PyMRST state | Required implementation |
|---|---|---|
| `boundaryFaces.m` | Implemented at `utils/gridtools/boundary_faces.py` | Add `plotting/boundary_faces.py` as the canonical MRST-layout owner or a reviewed wrapper; preserve numeric subsets, logical masks, empty subsets, and optional owning-cell output |
| `private/plotPatches.m` | Missing | Build 2D/3D face topology, compact used vertices, accept scalar/indexed/RGB colors, preserve selected face IDs, and return backend-neutral patch geometry plus rendered actor |
| `private/plotFaceOutline.m` | Missing | Extract unique exterior edges for selected faces; support both 2D line segments and 3D face outlines |

`plot_patches` is the central dependency. It must be implemented before the
public face, grid, cell-data, face-data, and node-data functions. Building
each public function directly on `to_pyvista` would duplicate topology logic
and make subtle subset behavior diverge.

### Existing partial public functions

#### `plotGrid.m` → `plot_grid.py`

What exists:

- 2D and 3D cell conversion
- selected-cell rendering
- color, edges, opacity, screenshot, z scaling

What is missing:

- exact external-face extraction rather than plotting complete cell shells
- logical cell masks and strict subset validation
- coarse-grid component behavior
- empty subset behavior
- pass-through patch/actor properties
- returned selected cells/faces and geometry
- MATLAB-compatible face visibility semantics, especially
  `FaceColor='none'`
- multi-component-grid rejection or explicit component support

Completion criterion: the same cell subset must select the same external
faces as MATLAB `plotGrid`, for Cartesian, unstructured, faulted, pinched,
and coarse grids.

#### `plotCellData.m` → `plot_cell_data.py`

What exists:

- one scalar per active cell
- optional cell subset
- colormap, limits, opacity, edges, screenshots

What is missing:

- MRST's boundary-face coloring semantics
- scalar, indexed, and RGB/TrueColor inputs
- cell data supplied for all cells or only the selected cells
- logical masks
- coarse-grid parent/subcell expansion
- robust data-to-face mapping through owning cells
- patch/actor keyword pass-through
- subset and shape errors matching MRST's accepted forms
- reusable geometry/result object

Completion criterion: exterior face colors and selected face IDs match MATLAB
for full and subset plots; RGB data must bypass scalar colormapping.

#### `plotWell.m` → `plot_well.py`

What exists:

- a tube through completion-cell centroids
- color, radius, height, and z scaling

What is missing:

- 3D-only validation
- well head/heel markers
- labels and escaped well names
- injector/producer `color`/`color2` behavior
- default colors based on well sign
- wellbore cylinder resolution (`cylpts`)
- font size, ambient strength, line width
- explicit trajectory/ref-depth behavior
- graceful handling of one completion, horizontal wells, repeated
  completions, closed wells, and missing names
- separate returned handles for heads, text, paths, and label lines

Completion criterion: trajectory points, top location, color selection, and
labels agree with MATLAB fixtures for vertical, horizontal, injector, and
producer wells.

### Missing public geometry/data functions

| MRST file | Required behavior |
|---|---|
| `plotFaces.m` | Numeric/logical face selection; scalar/RGB face colors; 2D line and 3D patch paths; outline option; coarse-grid faces; compact packed topology |
| `plotFaceData.m` | Plot face-centered data on the exterior of all cells or a cell subset; accept both `(G, data)` and `(G, cells, data)` forms; 3D validation |
| `plotNodeData.m` | Interpolated nodal colors; selected-cell boundary in 3D; 2D cell polygons; optional displacement `uu`; compact node topology |
| `plotFaults.m` | Fault face patches, centroid labels, configurable label height/font/color, named fault structures |
| `outlineCoarseGrid.m` | Partition discontinuity detection; 2D edge outlines and 3D face outlines; outer boundary behavior |
| `plotBlockAndNeighbors.m` | Coarse-block neighbor discovery; separate block/neighbor colors and alpha; optional fault overlays; return all actors |
| `plotSlice.m` | Cartesian index slices in dimensions 1–3; `cartDims`/active mapping; correct tagged faces, values, and view orientation |
| `plotContours.m` | 3D cell-data contours using cell centroids, slice planes, and contour levels consistent with MRST |
| `plotGridVolumes.m` | 3D interpolation; histogram bins or supplied bins; min/max filtering; isosurfaces; extrusion; colormap/alpha progression; polygon reduction; reusable interpolant |

`plotGridVolumes` is not equivalent to PyVista's default volume rendering.
MRST builds a series of partially transparent isosurfaces. PyMRST must expose
that behavior, while optionally offering native volume rendering as an
extension.

### Missing interpolation and 2D functions

| MRST file | Required behavior |
|---|---|
| `interpolateData.m` | Infer node/face/cell source coordinates from row count; explicit source coordinates; node/face/cell or explicit query coordinates; multi-column data; interpolation/extrapolation methods |
| `unstructuredSurf.m` | 2D-only validation; cell subset; interpolate face/cell values to nodes; lift node z-coordinate; return surface geometry/actor |
| `unstructuredContour.m` | 2D-only validation; inferred or explicit Cartesian sampling dimensions; filled/unfilled contours; count or explicit levels; extrapolation; contour matrix/result |

`interpolate_data` should use `scipy.interpolate` and be independently tested
before either unstructured plotting function. Ambiguous source sizes—for
example, a pathological grid with equal node and cell counts—must require
explicit coordinates rather than silently choosing one.

### Missing colorbar functions

| MRST file | Required behavior |
|---|---|
| `colorbarHist.m` | Histogram plus colorbar at south/east/west; bin count or explicit bins; supplied limits; logarithmic labels; return both axes/artists |
| `mrstColorbar.m` | Accept explicit axes or current axes; infer plotted values or accept values; all outside locations supported by Matplotlib; log tick formatting; explicit limits |

These two should share one internal histogram/colorbar layout engine. They
remain separate public files because MRST exposes both.

## PyMRST-only extensions that must be preserved

The following are not parity gaps and should not be deleted:

| Extension file | Purpose |
|---|---|
| `browser.py` | Plotly/browser reservoir visualization |
| `plot_toolbar.py` | interactive field/time/threshold controls |
| `plot_saved_states.py` | saved simulation-state visualization |
| `plot_streamlines.py` | streamline conversion and rendering |
| `plot_inventory.py` | trapping inventory chart |
| `plot_ve.py` | caprock, plume, and trap rendering |

The extension move includes these cleanups:

- lazy imports so core geometry helpers work without browser/GUI extras;
- imports through public core APIs rather than private functions in
  `plot_grid.py`;
- lazy public exports from `pymrst.plotting.__init__` without old-path wrapper
  modules.

## Implementation order

### Phase 1 — Layout, result types, and topology primitives

Create the MRST-shaped files and the `private/` package. Move or wrap
`boundary_faces`. Define backend-neutral result dataclasses and implement:

1. `private/plot_patches.py`
2. `private/plot_face_outline.py`
3. `plot_faces.py`

Acceptance:

- exact face and vertex topology on 2D and 3D Cartesian grids;
- faulted/polyhedral face loops remain intact;
- numeric and logical subsets agree;
- scalar and RGB colors are both supported;
- no GUI is needed for geometry tests.

#### Phase 1 status: implemented and validated

The MRST-shaped `boundary_faces.py`, `plot_faces.py`, `results.py`, and
`private/` package now exist. `build_patch_geometry` produces compact
variable-length face topology; color normalization distinguishes constant,
per-face, per-node, scalar, and RGB data; `face_outline_geometry` removes
edges internal to a selected face collection; and the PyVista adapter is
tested through renderer injection without requiring a GUI.

Coarse faces retain their parent grid and expand into grouped fine faces.
Python regressions cover Cartesian, coarse, faulted corner-point, and
polyhedral grids. The MATLAB generator
`examples/validation/dump_plotting_phase1_reference.m` and Python validator
`validate_plotting_phase1.py` verify Cartesian boundary selection, compact
patch topology, outlines, and order-independent grouped coarse topology.

Faulted and polyhedral MATLAB fixtures remain additional cross-language
coverage, but their topology paths are covered by Python regressions.

### Phase 2 — Core grid, cell, face, and node plots

Refactor and finish:

1. `plot_grid.py`
2. `plot_cell_data.py`
3. `plot_face_data.py`
4. `plot_node_data.py`

Do not keep `plot_cell_data` and `plot_well` embedded inside `plot_grid.py`;
each MRST public function gets its corresponding file. Compatibility imports
from `plot_grid.py` may remain temporarily.

Acceptance:

- MATLAB face-selection fixtures match exactly;
- cell/face/node data maps to identical entity IDs;
- 2D/3D, subsets, RGB, deformation, and empty selections are covered;
- PyVista headless screenshots are nonempty.

#### Phase 2 status: in progress

`plot_face_data.py` is now a standalone MRST-layout implementation. It
supports `(G, data)` and `(G, cells, data)`, numeric or logical cell subsets,
full-grid scalar/RGB face data, empty selections, strict 3-D validation, lazy
rendering, and backend-neutral result geometry. Its subset-boundary
face/value mapping is covered by focused tests.

`plot_cell_data.py` now implements scalar and RGB values supplied for either
all cells or only the selected subset, numeric/logical selections, NaN-cell
exclusion before boundary extraction, exact owning-cell color mapping, 3-D
exterior faces, 2-D cell polygons, coarse-grid expansion, empty selections,
and backend-neutral result geometry.

The MRST `plotGrid` behavior is now routed through the validated topology
core: 3-D selections render only their exterior faces and 2-D selections
render cell polygons. This implementation now owns the canonical
`plot_grid.py`; historical whole-cell conversion and the older well adapter
live in `_pyvista_grid.py` and load only when requested. Existing imports of
`to_pyvista` and `plot_well` from `plot_grid` remain compatible.

`plot_node_data.py` now supports scalar/RGB nodal interpolation on 2-D cell
polygons and selected 3-D boundary faces, compact node topology, numeric or
logical cell selection, optional `uu` displacement without mutating the
simulation grid, and empty selections.

The MRST 1-D `plotGrid` and `plotCellData` paths are also implemented as
centroid/value polylines, including subsets and multi-series data. MATLAB
fixtures verify both line geometries exactly.

Phase 2 backend rendering QA is now defined and implemented. Common MATLAB
patch properties (`FaceColor`, `EdgeColor`, `FaceAlpha`, and `LineWidth`)
are translated case-insensitively by the shared rendering layer, while
native PyVista keywords retain precedence. `EdgeColor='none'` disables edge
rendering explicitly.

`zscale` now has one canonical meaning: it transforms the returned/rendered
3-D topology about the shallowest grid-node depth. It must be finite and
positive. Reversed-Z display is a camera convention rather than a geometry
reflection: the standard reservoir camera uses view-up `(0, 0, -1)` and a
reproducible `xz`/azimuth/elevation setup for plotters owned by the plotting
function. Caller-owned plotters retain their position but receive the
depth-down view-up convention.

The VTK screenshot regression runs rendering in an isolated subprocess.
This is necessary because an unavailable Cocoa/OpenGL backend can abort the
interpreter instead of raising a Python exception. A working renderer must
produce two identical, nonempty images; an unavailable renderer is reported
as a safe test skip without terminating the main test process.

The MATLAB fixture now also validates selected cell-data boundary faces,
owning-cell values, and compact node IDs for nodal data. These comparisons
run alongside the Phase 1 topology checks in
`validate_plotting_phase1.py`.

### Phase 3 — Wells, faults, coarse grids, and slices

Implement:

1. `plot_well.py`
2. `plot_faults.py`
3. `outline_coarse_grid.py`
4. `plot_block_and_neighbors.py`
5. `plot_slice.py`

Acceptance:

- well trajectory and label fixtures match MRST geometry;
- fault face and label positions match;
- coarse partition boundaries and neighbor sets match;
- all three slice dimensions and inactive-cell mappings are tested.

#### Phase 3 status: core behavior implemented and validated

`plot_well.py` is now the canonical MRST-layout well renderer. It validates
3-D grids and completion indices; handles individual or multiple wells,
single- and multi-completion trajectories, MRST radius scaling, configurable
height/radius/cylinder resolution/font/ambient strength/line width,
injector/producer `color`/`color2` selection, names and fallback names,
head/heel markers, labels, paths, label lines, and compatibility `zscale`.
It returns separate actor groups plus backend-neutral geometry.

MATLAB fixtures verify completion coordinates, head, heel, trajectory, and
radius. Exact MRST direction-dependent segment meshing is now implemented
for `x`, `y`, `z`, and slanted completion cylinders. The renderer builds the
same explicit quadrilateral surface connectivity as `plotWell.m`, including
MRST's anisotropic single-completion sphere and polygonal head/heel disks.
Closed wells produce inspectable inactive geometry without actors; closed
completions are filtered together with their direction entries; consecutive
repeated completion cells are retained in the reported completion list but
collapsed from the rendered path to avoid zero-length cylinders. Dedicated
MATLAB fixtures compare vertices, faces, and both cap polygons for all four
direction branches and the single-completion case.

`plot_faults.py` is now implemented for `GridFault` objects, mapping-style
fault records, or sequences of either. It validates and aggregates fault
faces, renders the shared black patch, retains per-fault face groups, places
names at mean face centroids with configurable height/font/text color, and
returns both patch and text handles with inspectable label coordinates.
MATLAB fixtures verify face IDs, groups, and label positions.

`outline_coarse_grid.py` is now implemented with MRST's dimension-specific
semantics: 2-D outlines contain partition interfaces and the outer domain
boundary, while 3-D outlines contain internal partition interfaces only.
It validates arbitrary numeric partition labels, exposes selected face IDs,
and renders through the common face topology path. Exact 2-D and 3-D face
selections are verified against MATLAB fixtures.

`plot_block_and_neighbors.py` is now implemented for retained-parent coarse
grids. It discovers face-adjacent coarse blocks, expands each block into its
fine-grid boundary, applies separate block/neighbor colors and alpha values,
optionally extracts tagged fault faces with MRST gray/red styling, and
returns every selected face set and actor. MATLAB fixtures verify the block,
neighbor IDs, and each fine boundary-face set.

`plot_slice.py` is now implemented for all three Cartesian dimensions using
zero-based Python slice indices. It maps global Cartesian indices through
`cells.indexMap`, skips inactive cells, selects the MRST negative-side
direction tag, maps active-cell scalar/RGB data onto the selected faces, and
returns cells, faces, values, and the view vector. MATLAB fixtures verify all
three directions exactly.

Phase 3 is complete at the documented scope: all public files exist and the
well, fault, coarse-grid, neighbor, and slice selection/data geometry is
validated against MATLAB/MRST fixtures.

### Phase 4 — Interpolation, surfaces, and contours

Implement:

1. `interpolate_data.py`
2. `unstructured_surf.py`
3. `unstructured_contour.py`
4. `plot_contours.py`

Acceptance:

- exact affine fields interpolate to machine precision;
- nonlinear scattered interpolation is compared with MATLAB using tolerances;
- node-, face-, and cell-sized inputs work;
- contour levels and structured query grids are recorded and testable.

#### Phase 4 status: implemented and validated

`interpolate_data.py` is now implemented as the shared numerical foundation.
It infers unambiguous node/face/cell source coordinates or accepts explicit
coordinates, supports named or explicit query locations, scalar and
multi-column data, 1-D/2-D/3-D scattered interpolation, linear/nearest and
2-D natural-style interpolation, and linear/nearest/no extrapolation.
Ambiguous entity counts require explicit coordinates instead of silently
choosing the wrong location. MATLAB fixtures verify affine interpolation,
linear extrapolation, and inferred cell-to-node interpolation.

`unstructured_surf.py` is now implemented for scalar node/face/cell data,
numeric or logical cell subsets, configurable boundary extrapolation,
compact 2-D cell topology, lifted z coordinates, interpolated nodal colors,
and backend-neutral results. MATLAB fixtures verify nodal values, selected
node IDs, connectivity, offsets, and lifted vertices.

`unstructured_contour.py` is now implemented with inferred or explicit
structured sampling dimensions, the MRST 1000-point cap, scalar
node/face/cell interpolation, configurable extrapolation, filled or line
contours, count or explicit levels, injected Matplotlib axes, and
backend-neutral query meshes/values. The plotting package defaults to the
headless Agg backend unless the caller explicitly selects another backend.
MATLAB fixtures verify inferred dimensions, x/y query meshes, and sampled
values.

`plot_contours.py` is now implemented with MRST's actual discrete 3-D
algorithm: equidistant value bands color exterior faces, while strict
below/above tests select internal threshold faces. It returns every level,
exterior/internal face set, color, and actor group with configurable opacity
and colormap. MATLAB fixtures verify levels and both face sets for every
band.

All Phase 4 public files now exist, their numerical geometry is covered by
Python tests, and their deterministic data paths are compared with MATLAB.

### Phase 5 — Histogram colorbars and volume plots

Implement:

1. `colorbar_hist.py`
2. `mrst_colorbar.py`
3. `plot_grid_volumes.py`

Acceptance:

- histogram counts, bins, limits, log labels, and orientation are unit tested;
- isosurface values, alpha levels, and geometry are deterministic;
- reused interpolants produce the same output;
- large meshes respect the polygon/patch limit.

#### Phase 5 status: implemented and validated

`colorbar_hist.py` and `mrst_colorbar.py` are now implemented with
Matplotlib-native, headless-safe rendering. They expose MATLAB-compatible
center-bin histogram counts, explicit or default bin counts, linear/log10
limits and labels, distinct south/north/east/west layouts, injected axes,
explicit values, and automatic scalar-data
collection from plotted artists. MATLAB fixtures verify linear and log
counts, centers, and transformed limits; Python tests render every supported
orientation through Agg. The north layout places the histogram above the
main axes and its horizontal colorbar farther outward, reverses the histogram
count direction like MRST's `view(0, -90)`, and moves colorbar ticks to the
top; it no longer reuses south placement.

Advanced MATLAB fixtures now cover exact boundary-patch loops for a faulted
corner-point grid and a deterministic triangular-prism polyhedral grid,
inactive-cell slice selection in all three Cartesian directions, and
isosurface bounds and area with tolerant face-count diagnostics for MATLAB/
VTK triangulation differences. Directional `plotWell` fixtures separately
compare exact vertices, quadrilateral connectivity, and head/heel polygons
for `x`, `y`, `z`, slanted, and single-completion meshes.

The PyMRST browser extension is organized as an engineering state explorer
rather than a bare Plotly animation. Its standalone HTML workspace separates
property and timestep selection, reports robust P2–P98 color ranges and
current-state statistics, provides playback and opacity controls, toggles
grid/well overlays, exposes reservoir camera presets, supports interactive
X/Y/depth cutaways, and reports clicked cell coordinates. The responsive dark
technical layout remains server-free and preserves both mesh and point modes.

`plot_grid_volumes.py` is now implemented with MRST histogram-center
isovalues, `min`/`max` filtering, explicit bins, default or explicit sampling
meshes, callable or named colormaps, progressive `basealpha`, a total patch
budget, boundary-face extrusion, reusable sampled fields, and PyVista
isosurface extraction. The result exposes sources, axes, structured values,
isovalues, meshes, face counts, colors, opacities, and actors.

MATLAB fixtures verify extruded source coordinates/values, structured axes,
the shared interpolation domain, full sampled-field reuse, and isovalues.
Boundary-corner extrapolation is deliberately not compared point-for-point
because it depends on the Delaunay boundary triangulation chosen by MATLAB
and SciPy; both implementations use linear scattered extrapolation.

All Phase 5 public files now exist and are validated.

### Phase 6 — Extension relocation and public API

Move PyMRST-only features under `plotting/extensions/`, add compatibility
wrappers, and make `pymrst.plotting.__init__` export:

- all 18 MRST public counterparts in snake_case;
- optional MATLAB-style aliases such as `plotGrid` and `plotCellData`;
- `to_pyvista` and all existing PyMRST extensions;
- no eager optional-backend imports.

Acceptance:

- existing examples import unchanged;
- importing `pymrst.plotting` succeeds with only core dependencies;
- missing plotting extras produce targeted installation instructions.

## Validation strategy

Pixel equality is not a reliable MATLAB/Python parity criterion. Different
renderers rasterize the same geometry differently. Validation should compare
the data that determine the plot.

### MATLAB reference fixtures

Add a MATLAB generator under `examples/validation/matlab/` that records:

- selected cell and face indices;
- owning cells for boundary faces;
- compact vertex arrays and polygon/edge connectivity;
- scalar or RGB color arrays after entity mapping;
- well trajectory, head, heel, and label coordinates;
- coarse-boundary face/edge IDs and neighbor blocks;
- slice cell/face IDs and values;
- interpolation query coordinates and outputs;
- contour query meshes and levels;
- histogram bins/counts and color limits;
- volume isovalues and sampled scalar fields.

Fixtures should cover:

1. 2D Cartesian grid;
2. 3D Cartesian grid;
3. inactive Cartesian cells;
4. a faulted corner-point grid;
5. a general polyhedral grid;
6. a coarse partition;
7. vertical and horizontal wells;
8. scalar and RGB fields.

### Python tests

Use three levels:

- **geometry unit tests:** no renderer, fast, exact topology/index checks;
- **MATLAB parity tests:** load `.mat` fixtures and compare arrays;
- **render smoke tests:** headless PyVista/Matplotlib output, nonempty PNG,
  valid actor/axes objects.

Image-regression tests may be added for PyMRST stability, but they must not
replace geometry and data parity.

## Completion definition

The folder matches MRST end to end only when all of the following are true:

- all 18 public MRST functions have MRST-shaped Python files;
- both private helpers exist and are used by the public core;
- documented input forms and important options are implemented;
- 2D, 3D, Cartesian, unstructured, faulted, coarse, subset, and logical-mask
  cases are supported where MRST supports them;
- scalar, indexed, and RGB data paths work;
- MATLAB geometry/data fixtures pass;
- headless render smoke tests pass;
- existing PyMRST plotting extensions still work;
- optional plotting dependencies are lazy;
- `__init__.py`, `README.md`, and `contents.py` document and export the full
  surface;
- no entry is marked complete solely because a filename or approximate
  visualization exists.

Until those gates pass, the plotting folder should be described as a useful
PyVista visualization layer with partial MRST plotting compatibility—not as
an end-to-end port of MRST `plotting/`.
