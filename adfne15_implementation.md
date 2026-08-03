# ADFNE 1.5 implementation plan

## Scope and licensing

The reference is `mrst-2024a-Geomech/modules/ADFNE15`: 86 MATLAB owners,
7,399 source lines, four PDFs, four images and one compatibility archive.
PyMRST retains the upstream three-clause ADFNE 1.5 license and attribution.
The port is native Python; it does not execute MATLAB at runtime.

## Implemented foundation

- explicit configuration replacing mutable MATLAB globals;
- `Angle`, numeric `Scale`, `Bbox`, `Center`, `Length`, `Plane`, `Project`,
  `Polynorm`, `Polyline`, `Pointline`, `Unique` and line/polygon `Clip`;
- line and polygon `Orientation`;
- bounded uniform/exponential and circular Fisher/von-Mises sampling;
- `Cluster`, `Occurrence`, `Percolate` and IDW;
- runnable seeded 2-D and 3-D `DFN` generation with rectangular-domain
  clipping and circle, ellipse and square fracture shapes;
- direct ADFNE MATLAB fixture for deterministic foundation owners.
- MRST-owner modules for `Intersect`, `Snap`, `Group`, `Sort`, `Split` and
  `Isolated`, including 2-D and convex 3-D intersection geometry and preserved
  zero-based source lineage.
- MRST-owner modules for `Pipe`, `Backbone`, `Graph`, `Boundary` and `Solve`:
  2-D fracture/boundary networks are split into 3-D pipes, dead-end branches
  are pruned, typed sparse graphs are built and the ADFNE cubic-law pressure
  and flow system is solved with SciPy sparse linear algebra.
- deterministic MRST owners `Distance`, `Field`, `Label`, `Ray`, `Relabel`,
  `Rotate`, `Translate` and `Wrap`, plus `Net`, `Stack`, `Reshape`, `Flatri`,
  `Filter`, `Not` and `Naned`;
- explicit parity failures for `Mesh` and `Tensor`, whose distributed MATLAB
  owners themselves only raise the ADFNE "not included" error.
- polygon-domain owners `Inpoly`, `PolyClip` and `GetShapePoly`, including
  plane-tolerance checks, ADFNE-ordered half-space vertex splicing and
  oriented cylinder boundary generation;
- computational `Geometry` modes for ellipse and sphere generation. The
  ellipsoid branch in MATLAB returns a graphics handle through `Draw` and is
  intentionally tracked with the visualization owners rather than represented
  as a numerical mesh with a different contract.
- conversion and option infrastructure in MRST-equivalent `Convert`, `Option`,
  `setstructfields` and `Globals` owners, using immutable Python configuration
  rather than mutable process-wide MATLAB globals;
- the numerical `CData` utility surface and native fluent `CPipe`, `CBackbone`
  and `CGraph` owners. The fluent classes invoke the same ported numerical
  kernels and support the full pipe-to-solve chain without MATLAB runtime
  delegation.
- the MRST-layout statistics/geostatistics owners `Info`, `Stats`,
  `Variocloud`, `Variogram`, `Variomodel`, `Kriging`, `Markov` and
  `Upscaling`. Ordinary kriging uses SciPy distance algebra but retains the
  ADFNE augmented Lagrange system, mesh ordering, estimate and error formulas.
- the remaining 19 MRST owners: `CExt` geometry utilities, legacy `DFN_orig`
  and `DFN_bin`, grid generation, contour and stereonet data, Matplotlib
  rendering/axes synchronization, VTK/Gmsh/Abaqus/XYZ/TXT/SVG/HTML/MAT and
  figure export, reporting/timing, and callable versions of the four upstream
  script workflows plus `Demo`.

## Layout rule

The Python package mirrors the flat owner layout of MRST `modules/ADFNE15`:
`Angle.m` maps to `angle.py`, `Rand.m` to `rand.py`, `DFN.m` to `dfn.py`, and
likewise for every implemented owner. The temporary `geometry.py` aggregate
has been removed: `Bbox`, `Center`, `Clip`, `Length`, `Plane`, `Pointline`,
`Polynorm`, `Polyline`, `Project` and `Unique` now live in their individual
MRST-equivalent modules. The former `topology.py` aggregate was likewise
removed; `Cluster`, `IDW`, `Occurrence` and `Percolate` have distinct owners.

## Completion and continuing certification

All 86 distributed MATLAB owner filenames now have native Python owners; the
only additional Python owner is immutable `config.py`. Implementation work is
therefore owner-complete for the distributed source surface. Continuing work
is certification rather than missing-file porting: broader randomized 3-D
geometry cases, golden-file checks for every export dialect, image comparisons
for presentation details, and full-sized runs of the callable example scripts.

`Mesh.m` and `Tensor.m` cannot supply numerical algorithms because the
distributed reference explicitly marks both unavailable. PyMRST preserves
those failures with typed exceptions.

## End-to-end acceptance

The acceptance suite now exercises complete workflows rather than isolated
owners:

- reference-scale 150-fracture directional and random 2-D generation;
- pipe splitting, backbone extraction, graph assembly and cubic-law solve;
- off-screen rendering plus SVG and PNG output;
- the reference-scale 99-fracture cylindrical 3-D core, clipping,
  intersections, VTK/Gmsh export and rendering;
- variogram cloud, experimental binning, ordinary kriging, contouring and
  stereonet density;
- direct deterministic MATLAB fixtures and all callable examples.

Run the reproducible gate from the repository root with:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/pymrst-mpl \
python3 scripts/audit_adfne15_parity.py --run-tests
```

No remaining owner may be represented by a forwarding wrapper or an empty
file. Unsupported behavior must remain absent or fail explicitly until ported.
