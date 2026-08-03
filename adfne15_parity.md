# ADFNE 1.5 parity

## Current conclusion

The distributed owner surface is **fully represented by native Python owner
modules: 86 of 86 MATLAB filenames**. Core deterministic numerics are directly
MATLAB-certified, and the complete Python qualification suite passes. Strict
behavioral certification remains deeper than filename/layout completion,
with reference-scale cross-owner acceptance. Two limitations are inherited
from the distributed upstream installation: `Mesh`/`Tensor` contain no
implementation, and three MATLAB comparisons require an unavailable
Statistics Toolbox. These are explicit exclusions rather than PyMRST gaps.

| Reference area | Status | Evidence |
|---|---|---|
| Configuration and options | Implemented/validating | immutable configuration, complete predefined polygon set, line set and option aliases |
| Foundational geometry | Implemented/validating | focused Python tests and direct MATLAB fixture |
| 2-D DFN generation | Implemented | seeded bounded networks and orientation checks |
| 3-D DFN generation | Implemented | planar clipped circle/ellipse/square polygons |
| Intersections and preprocessing | Implemented/validating | 2-D and convex 3-D intersections, snap, group, sort, split and isolation; direct fixtures except toolbox-blocked `Split` |
| Polygon-domain processing | Implemented/validating | direct MATLAB fixtures for 2-D/3-D `Inpoly`, oriented-cylinder `GetShapePoly`, and exact ordered `PolyClip` output |
| Geometry generation | Implemented/validating | ellipse and sphere directly MATLAB-certified; ellipsoid rendered natively through Matplotlib |
| Conversion and options | Implemented/validating | direct MATLAB fixtures cover clock/angle/normal/triangle/2-D/3-D conversion, option aliases and `CData` utilities |
| Fluent helper classes | Implemented | native `CPipe → CBackbone → CGraph → Solve` execution test; numerical stages are directly MATLAB-certified by their owning fixtures |
| Graph, backbone, pipe and solve | Implemented/validating | complete 2-D chain, direct MATLAB-certified 3-D polygon-center pipes, sparse cubic-law solve and direct MATLAB `Boundary`/`Solve`; `Mesh` triangulation is explicitly unavailable in the distributed ADFNE owner and PyMRST preserves that failure |
| Deterministic utility owners | Implemented/validating | direct MATLAB fixtures cover transforms, rays, distances, labels, array layout, filtering and NaN-separated coordinates |
| Geostatistics | Implemented/validating | `Info`, variogram cloud/binning, spherical/exponential/linear models, ordinary kriging, Markov chains and recursive upscaling implemented; direct MATLAB fixtures cover all deterministic owners except toolbox-blocked `Stats` and `Kriging` |
| Export and conversion | Implemented/validating | VTK, Gmsh, Abaqus, XYZ, TXT, SVG, HTML, MAT and PNG/PDF paths; format smoke/round-trip evidence |
| Plotting and stereonet | Implemented/validating | native line/point/polygon/cube/sphere/cylinder/ellipsoid rendering, contours, axes and stereonet density; pixel-level MATLAB equivalence remains certification work |
| Examples | Implemented/qualified | reference-scale directional/random 2-D and 99-fracture cylindrical core 3-D acceptance, OMO orientation and deterministic demo workflows |

## Acceptance result

- MRST-layout owners: **86/86**
- Python qualification: **33 passed**
- direct MATLAB foundation fixture: **generated and passing**
- reference-scale 2-D and 3-D cross-owner workflows: **passing**
- headless rendering and export acceptance: **passing**

The upstream random generator and NumPy do not share a bit-identical stream.
Deterministic geometric owners use direct value fixtures; stochastic owners
will use shared realizations plus distributional and seeded-reproducibility
tests.

`Intersect`, `Snap` and `Isolated` have direct MATLAB fixtures. `Split` is
composition-tested from those owners, but the installed MATLAB cannot execute
the reference `Split.m` because ADFNE calls Statistics Toolbox `pdist`; this is
an explicit validation dependency rather than a numerical acceptance claim.

Direct MATLAB validation also exposed and preserves ADFNE `Boundary.m`'s exact
hit behavior: a point lying exactly on an element produces `NaN` from its
`Inf/Inf` weighted quotient. PyMRST does not silently repair this in the parity
owner; a safer policy may be added later as an explicitly named extension.

`Tensor.m`, like `Mesh.m`, contains no numerical implementation in the
distributed ADFNE 1.5 source and raises an error unconditionally. PyMRST
therefore exposes a typed explicit failure rather than claiming a missing
upstream algorithm as a parity gap.

The installed MATLAB lacks the functions required by two upstream owners:
`Stats.m` stops at `nanmin`, while `Kriging.m` requires
`pdist`/`squareform`/`pdist2`. Their Python owners are qualified through
analytical identities and observation-reproduction tests. The fixture records
both missing-toolbox conditions explicitly; it does not label them direct
MATLAB matches.
