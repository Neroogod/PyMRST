# Finishing the `vem` module

What it would take for `pymrst/modules/vem` to match MRST's `modules/vem`
end to end, and in what order.

`VEM_PARITY.md` is the ledger this plan retires; `scripts/audit_vem_parity.py`
regenerates it. The CSV under `docs/vem/parity.csv` is authoritative.

## Where the module actually stands

MRST's `vem` is 19 files, four of which are `Contents.m` catalogs. The
remaining **15 production files are about 1,950 lines** of code excluding
comments. PyMRST has six Python files, and they are not small: 1,087 code
lines in `compute_virtual_ip.py` alone against MRST's 756.

So this is not a module that has barely been started. It is a module whose
**numerical core is largely done and largely validated**, and whose remaining
gap is mostly one of *layout* plus a short tail of genuinely missing
functionality.

| | count |
|---|---:|
| At an MRST-shaped path | 1 |
| Implemented but at a different path | 8 |
| Not implemented | 6 |

Nine of fifteen exist and work. That is the number that matters for planning,
and a filename-only scan would have reported it as one.

### What is already validated

Twenty MATLAB-backed validators (`validate_phase400`–`419`) cover this module
and **all of them pass**. They pin the quadrature rules, `polygonInt`,
`polyhedronInt`, the VEM geometry, the k=1 and k=2 inner products in 2-D and
3-D, and the solver on patch tests. The numerical substance of
`computeVirtualIP` and `incompVEM` is therefore not in question; what follows
is not a re-derivation of any of it.

### The one thing the ledger cannot tell you

`addBCVEM` shows as unvalidated because it is unported - but
`validate_phase406` compares PyMRST against an MRST run that *used*
`addBCVEM`. PyMRST reaches the same answer through `incomp_vem`'s
`dirichlet_nodes` argument. The physics is pinned; the function is not ported.
Both statements are true and the plan below treats them separately.

## Phase 1: put the existing code at MRST's paths

**Eight files, no new numerics.**

Seven MRST functions live inside two Python files:

| MRST file | currently in | should be |
|---|---|---|
| `utils/polyDim.m` | `utils/quadrature.py` | `utils/poly_dim.py` |
| `utils/triangleQuadRule.m` | `utils/quadrature.py` | `utils/triangle_quad_rule.py` |
| `utils/tetrahedronQuadRule.m` | `utils/quadrature.py` | `utils/tetrahedron_quad_rule.py` |
| `utils/polygonInt.m` | `utils/quadrature.py` | `utils/polygon_int.py` |
| `utils/polygonInt3D.m` | `utils/quadrature.py` | `utils/polygon_int_3d.py` |
| `utils/polyhedronInt.m` | `utils/quadrature.py` | `utils/polyhedron_int.py` |
| `utils/computeVEMGeometry.m` | `utils/vem_geometry.py` | `utils/compute_vem_geometry.py` |

Plus `utils/squeezeBlockDiag.m`, which is 21 lines and genuinely absent.

**Why this is worth doing rather than declaring the layout a matter of taste.**
The same argument as `ad-props` Phase 2: someone looking for MRST's
`polygonInt3D` should find `polygon_int_3d.py`, not have to know that six
unrelated functions were bundled under a name MRST never uses. A consolidated
file also hides which parts are ported - the auditor had to grow a `misplaced`
classification to see through it, and no auditor should need that.

**The constraint that makes this safe.** `utils/__init__.py` already re-exports
every symbol, so moving the definitions and leaving the imports alone changes
no call site. Twenty validators pass now and must pass unchanged afterwards;
that is the acceptance test for the whole phase.

**Do not leave `quadrature.py` behind as a re-export shim.** A file whose only
content is `from ... import *` is exactly the thing the `ad-props` ledger
counts as a shim and refuses to call ported. Delete it once the definitions
have moved.

**Size.** Half a day. Mechanical, and fully covered by existing tests.

### Phase 1 status: complete

`utils/` is now one module per MRST function:

`poly_dim.py`, `triangle_quad_rule.py`, `tetrahedron_quad_rule.py`,
`polygon_int.py`, `polygon_int_3d.py`, `polyhedron_int.py`,
`compute_vem_geometry.py`, and `squeeze_block_diag.py` (new). Each carries its
own tests. `quadrature.py` and `vem_geometry.py` are **deleted**, not left as
shims; `utils/__init__.py` re-exports every symbol, so no call site changed.

The ledger moved from 1 to **9 of 15** at MRST-shaped paths, and the misplaced
count from 8 to 1 - `incompVEM`, which is Phase 2.

One shared helper is not an MRST counterpart: `_cell_geometry.py`, holding the
corner-node gather that MRST inlines separately in `polygonInt` and
`polyhedronInt`. It is private and marked as PyMRST's own rather than a port.

**The split exposed a measurement error, which is the more interesting
result.** Before it, `polyDim`, `triangleQuadRule` and `tetrahedronQuadRule`
were recorded as validated. They were not. They shared `quadrature.py` with
`polygon_int`, which *is* named by `validate_phase403`, and the auditor credits
by file - so three functions inherited coverage they never had. Splitting them
made the ledger report 4 newly unvalidated entries, which looked like a
regression and was actually the measurement becoming honest.

That gap is now closed for real: `dump_vem_quadrature_reference.m` and
`validate_vem_quadrature.py` pin the raw points, weights, reference vertices
and volumes of all six rules, the full `polyDim` table over k = -2..4 and
dim = 1..3, and `squeezeBlockDiag` in both directions - **29 comparisons, all
matching MRST**. `validate_phase403` could not have caught a rule whose points
and weights were wrong in compensating ways; this can.

**Every ported entry is now validated.** The only unvalidated rows left are the
five that are not implemented, which is the correct state.

**Acceptance test met**: all 30 `validate_phase4xx` validators pass unchanged,
along with gridprocessing (250), ad_props (72) and deckformat (39).

One thing to know about `tetrahedron_quad_rule`, because it is a trap: its
weights do **not** sum to 1, and the returned `vol` is **not** the reference
volume - it is the square root of it. Only the product `vol * sum(w)` is the
volume. A caller that assumes either separately gets an answer that a
constant-function test cannot catch, because both errors cancel there. The
file's tests assert the product, and assert the two false invariants are false.

## Phase 2: `incompVEM` at its own path

**One file, ~390 MRST lines.**

`incomp_vem` is defined inside `compute_virtual_ip.py`. It is the module's
main entry point and MRST gives it a file at the module root; it should be
`pymrst/modules/vem/incomp_vem.py`.

This is more than a move, because the two functions have diverged in scope:

| MRST `incompVEM` option | PyMRST `incomp_vem` |
|---|---|
| `bc` | per-face constant Dirichlet/Neumann only |
| `src` | yes |
| `facePressure` | **missing** - no k=1 face-pressure reconstruction |
| `cellPressure` | **missing** - no k=1 cell-pressure reconstruction |
| `calculateFlux` | yes (`reconstruct_flux`) |
| `linSolve` | **missing** - solver is not injectable |
| `matrixOutput` | **missing** |

`facePressure` and `cellPressure` are the substantial ones. At k=1 the
unknowns are node pressures, so face and cell pressures have to be
*reconstructed* - MRST does it by applying the projection operator
`S.PiNstar` to the nodal values. Without them a k=1 solution cannot be handed
to anything that expects cell-centred pressures, which is most of MRST.

`cellPressure` needs `squeezeBlockDiag` from Phase 1, which is why that
ordering is not arbitrary.

**Also in this phase.** PyMRST's docstring records a limitation: *"Prescribed
nonzero Neumann flux is supported for 2-D only; in 3-D, boundary faces without
a pressure BC are treated as no-flow."*

**That docstring is stale, and the claim above it was wrong.** An earlier draft
of this plan called 3-D Neumann the module's highest-severity gap on the
strength of it. `_neumann_rhs` in fact handles 3-D at both orders - the flux
goes to the face-moment dof at k=2 and is distributed to the face nodes by
that face's P0 projection at k=1. A column with a prescribed influx and a
pressure outlet conserves mass to 6e-15. Nothing needed fixing; the docstring
needed deleting, and a test needed writing so the claim cannot rot again.

**Size.** 2 days.

### Phase 2 status: complete

`incomp_vem.py` exists at the module root, holding the solver and the ten
tests that exercise it; `compute_virtual_ip.py` keeps the inner product and
its three. The misplaced count is now **0** and the ledger reads **10 of 15**
at MRST-shaped paths. Nine validators imported `incomp_vem` from its old home
and were repointed.

All four missing options are in: `face_pressure`, `cell_pressure`, `lin_solve`
and `matrix_output`.

**The one real defect fixed here** was `_face_pressures` in 3-D at k=1, which
raised `NotImplementedError`. MRST projects with `PiNFstar(1,:)`, and PyMRST
already computed exactly that quantity in `_face_p0_weights` for the cell B
matrix - it was simply never reused. Face and cell pressures now reproduce a
linear field on a 3x3x3 grid to 2e-15.

**What was not a defect**: 3-D Neumann, as above. Checking it cost one test
and removed a wrong item from this plan.

`squeeze_block_diag` from Phase 1 turned out not to be needed by
`_cell_pressures`, which reaches `PiNstar` per cell directly rather than
through a block-diagonal squeeze. The dependency this plan asserted between
Phases 1 and 2 was therefore real for MRST's implementation and not for this
one. `squeeze_block_diag` remains ported and validated on its own account.

**One file, 35 MRST lines - but the interesting part is what consumes it.**

`addBCVEM` builds a boundary-condition struct whose entries are *function
handles*, not constants: `bc.func{i}` is evaluated at boundary coordinates.
That is how MRST imposes a spatially varying exact solution, which is what
makes a patch test a patch test.

PyMRST substitutes `dirichlet_nodes`, a dict of dof index to value, which the
caller has to populate by evaluating the function itself and knowing the dof
numbering - including, at k=2, the face-moment and edge dofs. It works, and
`validate_phase406` proves it produces MRST's answer, but it pushes the dof
layout onto every caller.

Porting `addBCVEM` and teaching `incomp_vem` to accept `bc.func` moves that
knowledge back inside the module. `dirichlet_nodes` should stay - it is a
legitimate lower-level entry point - but it should stop being the only way in.

**Size.** 1 day.

### Phase 3 status: complete

`utils/add_bc_vem.py` holds `add_bc_vem` and the `VemBC` container; `incomp_vem`
gained `_impose_dirichlet_func` and `_neumann_rhs_func` and routes to them when
the bc carries functions. The ledger reads **11 of 15**.

Six tests cover it, and the one that matters most is
`test_it_agrees_with_the_hand_built_dof_map`: `add_bc_vem` and
`dirichlet_nodes` must be two routes to the same answer, or the convenience
layer is quietly solving a different problem. Patch tests pass through the new
path at k=1 and k=2, in 2-D and 3-D, to ~1e-14 - the k=2 3-D case being the
one that exercises edge moments, face moments and nodes together, which is
exactly the dof bookkeeping a caller should not have to do.

`validate_phase406`, which compares against an MRST run that used MRST's own
`addBCVEM`, still passes - so this is cross-validated against the reference
implementation, not only against PyMRST's other entry point.

**Function-valued flux is 2-D only, and that is MRST's limit too.** MRST
asserts `griddim == 2` before the loop because its own 3-D branch is an empty
`else` - it would contribute nothing and report nothing. This port raises
`NotImplementedError` naming the restriction and pointing at `add_bc` for the
constant-flux case, which does work in 3-D.

**A note on a mistake worth recording.** The first k=2 check through this path
failed by 1.2, and the cause was the test field, not the code: `0.5x^2` has
Laplacian 1, so with `K = I` it is not harmonic and the solve needs a source
term. A VEM patch test only reproduces fields the PDE actually admits. `x^2 -
y^2` works; the tests now say so in a comment, because this will catch the
next person too.

**One file, 94 MRST lines.**

VEM fluxes are not locally conservative as computed; `conserveFlux`
post-processes them so the divergence matches the sources cell by cell. It
solves a small constrained least-squares problem with permeability- or
area-weighted face weights.

Nothing in PyMRST does this, and nothing substitutes for it. Any transport
step taking VEM fluxes needs it - a non-conservative flux field produces
saturation that appears from nowhere.

Its return value is `[state, r]`, where `r` is the residual before correction.
Keep it: it is how a caller knows whether the correction was needed and by how
much, and MRST warns rather than failing when the flux was already
conservative.

**Dependencies - and this is the part to check first.** `conserveFlux` uses
`boundaryFaces` and `blockDiagIndex`. **Neither exists in PyMRST**: there is
no `boundary_faces` and no `block_diag_index` anywhere in the tree. Both are
small - a boundary face is one with a single neighbour, and `blockDiagIndex`
returns the row and column indices of a block-diagonal layout - but they are
used well beyond this module in MRST, so they belong in `pymrst/utils/`
rather than under `vem/`, and porting them is part of this phase's cost.

**Size.** 2 days: 1–2 for `conserveFlux` itself and the weighting options,
plus the two missing utilities.

### Phase 4 status: complete

`utils/conserve_flux.py`, plus `pymrst/utils/gridtools/block_diag_index.py`
and `boundary_faces.py`. The ledger reads **12 of 15**; only the three
examples remain.

**Which PyMRST flux fields actually need this - measured, not assumed.** On a
twisted 5x5 grid, `incomp_vem`'s default reconstruction and its MPFA variant
are conservative to ~1e-15 already: they recover a constant `K grad p` per
cell, whose flux through a closed surface cancels by construction. The
**TPFA** reconstruction is off by ~1e-1 on the same problem, because a
two-point transmissibility cannot represent a full tensor on a non-orthogonal
grid. `conserve_flux` takes it from 1.0e-01 to 9.1e-17.

So this plan's claim that "any transport step taking VEM fluxes needs it" was
wrong for two of the three paths. The function is still needed - for the TPFA
path and for fields arriving from elsewhere - but the reason is narrower than
stated, and the tests now open by establishing the premise rather than
assuming it.

**A trap in the weighting options.** The weights must be equal on both sides
of a face or the correction does not telescope and the result is not
conservative. `permWeighted` and `ones` are face-symmetric by construction;
`tpf` is a *half*-face transmissibility and differs across a face by 15% on
the twisted grid, leaving a residual of 5.9e-03. MRST has the same property,
which is why its last act is a "could not construct conservative flux field"
warning rather than an assertion. That is reproduced, and tested, with the
asymmetry measured directly so the mechanism is on record.

**One deliberate layout deviation.** MRST files `boundaryFaces.m` under
`plotting/`. Placing the counterpart in `pymrst/plotting/` made
`import conserve_flux` pull in PyVista and matplotlib - a rendering toolchain
dragged into a numerical routine. It sits in `pymrst/utils/gridtools/` beside
`rldecode` and `mcolon` instead. This is the only place in the `vem` port
where MRST's layout is not followed, and the reason is in the module's
docstring.

**A fixture that would have flattered the code.** `permWeighted` reduces to
`ones` when the rock is isotropic with unit permeability, since `n' K n` is
then 1 on every face. A homogeneous test would have reported a weighting that
does nothing as working; the test uses an anisotropic tensor, and a second
test records the coincidence so the fixture is not mistaken for arbitrary.

## Phase 5: the examples

**Three files, ~210 MRST lines.**

`vemExample1` (2-D, k=1 vs k=2 on a PEBI grid against `incompTPFA`),
`vemExample2` (3-D), and `producerInjectorGrid`, the helper both use.

These are not decoration. They are the only place the module is exercised as a
user would use it - build a grid, set wells or sources, solve, compare against
TPFA - and they will exercise Phase 2's `cellPressure` and Phase 4's
`conserveFlux` in combination, which no unit test does.

Each should run end to end and be validated against a MATLAB dump of the same
example, in the pattern the rest of this repo uses.

**Dependency check - done.** Both examples need `incompTPFA`, `addSource`,
`addBC`, `initResSol` and `initSingleFluid`. All five are present:

| MRST | PyMRST |
|---|---|
| `incompTPFA` | `pymrst/modules/incomp/incomp_tpfa.py` |
| `addSource` | `pymrst/params/wells_and_bc/add_source.py` |
| `addBC` | `pymrst/params/wells_and_bc/add_bc.py` |
| `initResSol` | `pymrst/solvers/init_res_sol.py` |
| `initSingleFluid` | `pymrst/modules/incomp/fluid/__init__.py` |

So this phase does not grow.

**Size.** 2 days.

### Phase 5 status: two of three, and the third is blocked

`examples/vem_example_1.py` (3-D, both orders, `cell_pressure` and
`matrix_output`) and `examples/vem_example_2.py` (2-D, four rotated anisotropy
regions, injector and producer) both run and self-test. The ledger reads
**14 of 15**.

**`producerInjectorGrid` is blocked, and the dependency check above missed
it.** It needs `createWellGridPoints`, `createFaultGridPoints`, `mlqt`,
`removeConflictPoints2` and `clippedPebi2D` - all from MRST's **`upr`**
module, which PyMRST does not have at all. Porting it means porting
well-and-fault-constrained PEBI generation, which is a module of its own and
plainly outside "finish `vem`". It is left pending with that reason recorded.

The check in this plan verified the five dependencies I thought of and
declared the phase safe. It should have enumerated what the files actually
call. Two further things it got wrong: `producerInjectorGrid` is used by
**neither** example, not "the helper both use"; and both examples need `upr`
grid generators of their own - `mirroredPebi3D` and `pebiGrid2D`.

**So the examples are substitutions, not ports**, and both say so at the top.
The grid *generators* differ; everything downstream - `compute_vem_geometry`,
`compute_virtual_ip` at k=1 and k=2, `incomp_vem` with sources, boundary
conditions, `cell_pressure` and `matrix_output` - is the code MRST's examples
exercise.

**Three things the examples taught, which is what examples are for:**

- **A lattice makes `tetrahedral_grid` degenerate.** Coplanar points give a
  tetrahedralisation whose `faces.neighbors` reference cells that do not
  exist, and zero-volume cells. Random points, as MRST uses, are sound across
  seeds. `build_grid` now checks rather than assumes.
- **Jittering PEBI generators makes slivers.** At any jitter from 0.02 to
  0.35, cell volumes span five orders of magnitude and the local VEM blocks
  acquire negative eigenvalues, leaving the system indefinite and the solve
  accurate only to ~1e-6. An unjittered lattice gives 4- to 7-sided cells with
  a volume ratio of 4. This is a property of PEBI construction from
  near-cocircular points, not of the VEM code - and it is documented because a
  reader's first instinct is to turn the jitter up.
- **The two orders agree in example 1 and differ by 40% in example 2**, and
  the reason is the solution, not the method: a uniform drop across a
  homogeneous box is linear, which both orders reproduce exactly. Rotated
  tensors refract the flow and produce a solution neither order captures
  exactly. An earlier version of example 1 called the agreement
  "discretisation error", which was the wrong explanation of a right number.

## Phase 6: tests and the ledger

There is **no `tests/vem/` directory**. The module's entire safety net is the
20 `validate_phase4xx` scripts, which pin saved values against MATLAB but do
not test invariants, error paths, or option combinations.

`tests/vem/` should cover:

- **patch tests as tests**: a linear field reproduced exactly at k=1, a
  quadratic at k=2, in 2-D and 3-D, as assertions rather than printed numbers;
- **convergence**: refining a grid reduces the error at the expected rate -
  the check that catches a discretisation that is consistent but not accurate;
- **conservation**: after `conserveFlux`, the per-cell flux divergence equals
  the source to machine precision;
- **option coverage**: `facePressure`, `cellPressure` and `calculateFlux` each
  on and off, since these are the paths Phase 2 adds;
- **the k=1/k=2 boundary**: a k=2 solve on a grid where k=1 is inexact, so the
  two orders are distinguished rather than assumed.

**Naming.** `tests/ad_core/` and `tests/ad_blackoil/` already collide on
`test_parity_ledger.py`, which blocks whole-suite collection. Any parity test
here must use a distinct basename - `test_vem_parity_ledger.py` - so this does
not deepen an existing problem.

**Size.** 2 days.

### Phase 6 status: complete

`tests/vem/` holds **61 tests** in five files, all with distinct basenames:

| file | covers |
|---|---|
| `test_vem_convergence.py` | measured convergence rates at both orders |
| `test_vem_patch_and_conservation.py` | exactness in 2-D and 3-D, conservation, `conserve_flux` |
| `test_vem_solver_options.py` | every `incomp_vem` option on and off, both orders |
| `test_vem_examples.py` | both examples still run and still demonstrate their point |
| `test_vem_parity_ledger.py` | the ledger matches the tree |

**Convergence was the gap worth closing.** Everything else in the module
tested exactness on polynomials the method reproduces exactly, which a
non-convergent method can also pass. On `exp(x) sin(y)` - harmonic, so no
source, and not a polynomial, so genuinely hard - the measured rates are
**2.0 at k=1 and 3.8 at k=2** in the nodal maximum norm, steady across four
refinements.

**A trap the tests found and now pin.** `bc=None` is *not* a sealed domain.
Without a boundary condition the flux reconstruction leaves each boundary face
at whatever value it computed - about 0.23 against a source of 1.0 on the test
grid - so a source-and-sink problem silently leaks and the totals do not
cancel. Sealing it takes an explicit zero-flux condition. This also
invalidated a conservation assertion in `vem_example_2`, which had been
passing by accident on its particular grid; the example now seals the boundary.

**The ledger test asserts what the ledger is for**: nothing may be
`misplaced`, every ported entry must be validated, a pending entry may never
be marked validated, and regenerating must not change the committed files. It
skips rather than passes when the MRST tree is absent.

| Phase | Work | Size |
|---|---|---|
| 1 | eight files to MRST paths, plus `squeezeBlockDiag` | 0.5 d |
| 2 | `incompVEM` at its path, face/cell pressure, 3-D Neumann | 2–3 d |
| 3 | `addBCVEM` and function-handle BCs | 1 d |
| 4 | `conserveFlux`, plus `boundaryFaces` and `blockDiagIndex` | 2 d |
| 5 | three examples with MATLAB references | 2 d |
| 6 | `tests/vem/` | 2 d |

**Roughly 9.5–11.5 days.**

Phases 1 and 2 are ordered by dependency: `cellPressure` needs
`squeezeBlockDiag`. Phases 3 and 4 are independent of each other and could run
in either order. Phase 5 depends on 2 and 4. Phase 6 can start as soon as
Phase 2 lands and should grow with each later phase rather than waiting for
all of them.

## Two things this plan deliberately does not do

**It does not rewrite `computeVirtualIP`.** At 1,087 code lines it is the
largest thing here and it is already validated against MRST at k=1 and k=2 in
both dimensions. Re-deriving it would be the most expensive way to change
nothing.

**It does not delete `vem_poisson.py`.** It has no MRST counterpart - MRST
solves through `incompVEM` - so it is PyMRST-only. But it is a self-contained,
tested, first-order Poisson solver that is far easier to read than
`computeVirtualIP`, and it is the natural place to send someone who wants to
understand the method before reading the general implementation. The ledger
records it as having no counterpart, which is the honest description; that is
different from calling it surplus.
