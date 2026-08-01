# vemmech end-to-end implementation plan

## Objective

Complete `pymrst/modules/vemmech` against the MRST 2024a `modules/vemmech/`
tree to the same standard already applied to `gridprocessing` and `ad-core`:

- every relevant MRST production file is inventoried;
- every numerical API has a direct or clearly documented consolidated Python
  counterpart;
- MATLAB-only, MEX, and compatibility-only files are explicitly classified
  rather than silently omitted;
- Python behavior is covered by focused unit and invariant tests;
- numerical behavior is validated against independently generated MATLAB/MRST
  reference data;
- MRST's own vemmech examples run end to end and agree numerically;
- the parity audit reports zero pending and zero ported-but-unvalidated items.

The target MRST source tree is:

```text
mrst-2024a-Geomech/modules/vemmech
```

The target PyMRST package is:

```text
PyMRST/pymrst/modules/vemmech
```

## Current implementation status

Measured against the MRST tree, not estimated:

| | MRST | PyMRST |
|---|---:|---:|
| files | 72 `.m` (7 are `Contents.m`, so 65 production) | 55 `.py` |
| lines | 10,701 | 11,791 |

- **49 of 65** MRST production files are accounted for: 42 have direct
  filename counterparts, 4 are consolidated into another documented Python
  module (`lincompTPFA`, `ssparsemul`, `tbinaryop`, `tcontract`), and 3 MEX
  helpers are explicitly excluded.
- **16 are genuinely absent**, classified in the phases below. The parity
  ledger (`VEMMECH_PARITY.md`) is the authoritative count.
- **37 validators pass** against saved MATLAB references
  (`validate_phase` 7, 27, 61, 68-79, 84-99, 102-104 and 106-108),
  covering the 2D and 3D operators, `VEM_assemble`, `VEM_linElast`, the TEDFM
  variants, stress recovery, `grid_utils`, `plotting`, the AD version and the
  calibration stack.
- **The solver works end to end today.** A 4x4x4 uniaxial compaction case
  solves to the correct field: prescribed top displacement honoured, clamped
  base held, mid-plane displacement -0.004894 against -0.005 for the ideal
  1-D case (the difference is the Poisson response to the clamped base), and
  `calculate_stress_vem` returns the expected `(ncells, 6)` tensors.

The directory layout already mirrors MRST, including empty `examples/` and
`examples/utils/` scaffolding awaiting Phase 4. Only `ad_version/mex/` is
absent, which is correct - see Phase 1.

What this means: **the physics core is not the gap.** The gap is the fixture,
example and audit layer that turns a working solver into a module that can be
shown to match MRST end to end.

## Definition of done

vemmech is complete only when all the following are true:

1. A reproducible parity scanner inventories every MRST `.m` file under
   `modules/vemmech/`.
2. Every file is classified as one of `ported`, `ported-consolidated`,
   `excluded-mex`, `excluded-catalog`, or `excluded-documentation`, with an
   explicit reviewed reason recorded for every exclusion.
3. No entry is `pending-review`, and no ported entry lacks a naming test or
   reference validator.
4. Each of MRST's six vemmech examples has a Python counterpart that runs to
   completion and agrees with MATLAB on displacement, stress and, where
   applicable, pressure.
5. A `tests/vemmech/` pytest suite covers the module's invariants, not only
   its saved reference values.
6. The poroelastic path is validated with both the interior and the boundary
   fluid terms active.

## Compatibility conventions

These follow the conventions already fixed by `gridprocessing` and apply
throughout the phases below.

### Indexing

MRST is 1-based; PyMRST is 0-based everywhere, including node, face and cell
indices held in boundary-condition structures. Any MATLAB reference dump must
convert indices on the MATLAB side (`face0 = face - 1`) so the Python
validator compares like with like, as the existing vemmech dumps already do.

### Boundary-condition structures

`ElasticBC` carries `disp_nodes`, `disp_mask`, `disp_vals`, `force_faces` and
`force`. MRST's `el_bc` splits the same information across `disp_bc` and
`force_bc` structs. The fixtures in Phase 2 are the main consumers of this
mapping, and it must be written down once in the fixture module rather than
re-derived per example.

### Numerical tolerances

Displacement and stress fields compare at `rtol=1e-10`, matching the existing
vemmech validators. Assembled operators compare at `rtol=1e-10, atol=1e-12`.
Where a comparison must be looser, the validator states why inline.

## Phase 1: parity ledger and audit tooling

**Why first.** vemmech is the only substantial ported module without a parity
ledger. `ad_core`, `ad_blackoil` and `gridprocessing` each have one; without
it, "finished" is not measurable and the MEX files cannot be formally
excluded.

**Deliverables**

- `scripts/audit_vemmech_parity.py`, mirroring
  `scripts/audit_gridprocessing_parity.py`;
- `docs/vemmech/parity.csv` as the authoritative ledger;
- `VEMMECH_PARITY.md` as the generated summary.

**Classification decisions to record**

| MRST file | status | reason |
|---|---|---|
| `ad_version/mex/getjac.m` | `excluded-mex` | MATLAB MEX accessor for AD tensor internals |
| `ad_version/mex/getval.m` | `excluded-mex` | same |
| `ad_version/mex/makeadi.m` | `excluded-mex` | MATLAB ADI construction helper |
| `ad_version/mex/tbinaryop.m` | `ported-consolidated` | absorbed into `SparseMultiArray` |
| `ad_version/mex/tcontract.m` | `ported-consolidated` | absorbed into `SparseMultiArray` |
| `ad_version/private/ssparsemul.m` | `ported-consolidated` | resolved: contraction is native to `SparseMultiArray.__mul__` |
| `utils/lincompTPFA.m` | `ported-consolidated` | `lincomp_tpfa` in `models/linear_compressibility_model.py` |

**Status: done.** `scripts/audit_vemmech_parity.py`, `docs/vemmech/parity.csv`
and `VEMMECH_PARITY.md` are in place. The scan reports 72 files: 42 ported, 4
consolidated, 3 excluded MEX, 7 catalogs, 16 pending.

Two things the ledger settled that this plan had left open:

- **`ssparsemul` is consolidated, not missing.** It is called only from
  `SparseMultiArray.m`, at its two contraction sites, and PyMRST's
  `SparseMultiArray.__mul__` performs that contraction natively. The
  1,004-line implementation lives in `ad_version/__init__.py`;
  `ad_version/sparse_multi_array.py` is a namespace re-export.
- **Pending is 16, not 20.** `ssparsemul`, `tbinaryop` and `tcontract` are
  consolidated rather than absent, and `simple_matching_example` is counted
  under Phase 4.

**What the ledger found that this plan did not.** Eight ported entries have no
test or validator naming them: `C2D`, `VEM_div`, `VEM3D_div`, `calculateQC`,
`lincompTPFA`, `Sigma2SmaxSminTheta`, `optimize_mech` and `compute_Ax_derivs`.
Each is exercised indirectly - `c2d` runs inside `vem_assemble`, which is
validated - but none is pinned on its own, so a regression confined to one of
them would not be caught. Phase 5 must close this; it is the concrete form of
the "completeness of the ported files is unaudited" risk below.

**Size.** Small. Half a day. Spent.

## Phase 2: test-grid fixtures

**Why this is the real blocker.** Every MRST vemmech example is built on these
fixtures. Until they exist, no example can be run, and no example can be
compared against MATLAB.

**Files** (line counts exclude comments and blanks)

| MRST file | lines | destination |
|---|---:|---|
| `utils/squareTest.m` | 264 | `pymrst/modules/vemmech/utils/square_test.py` |
| `utils/squareLayersTest.m` | 263 | `utils/square_layers_test.py` |
| `utils/squareGrid.m` | 156 | `utils/square_grid.py` |
| `examples/utils/makeCompactionTest.m` | 129 | `examples/utils/make_compaction_test.py` |
| `utils/complex3DGrid.m` | 81 | `utils/complex_3d_grid.py` |
| `utils/linearDisplacement.m` | 52 | `utils/linear_displacement.py` |
| `utils/exploreSquareGrid.m` | 25 | `utils/explore_square_grid.py` |
| `utils/ENu2LMu_3D.m` | 16 | `utils/enu2lmu_3d.py` |

Total ~986 lines of MATLAB.

**Where the risk is.** `squareTest` and `squareLayersTest` are 527 lines
between them and encode boundary-condition setup together with analytic
reference solutions. The line count understates them: getting the BC
conventions and the analytic field exactly right is where the time goes, not
the transcription. `linearDisplacement` is the analytic linear displacement
field these lean on and should be ported first so the others can be checked
against it.

**Validation.** One MATLAB dump per fixture recording the generated grid
(cell/face/node counts, coordinates) and the assembled `el_bc` arrays, so a
fixture is pinned before anything is built on it.

**Size.** Medium. Two to three days.

## Phase 3: poroelastic boundary coupling

**The gap.** `vem_lin_elast` currently applies only the *interior* Biot term:

```python
if pressure is not None:
    rhs += div.T @ pressure          # vem_lin_elast.py
```

MRST's `utils/addFluidContribMechVEM.m` (54 lines) supplies a different term:
the *boundary* fluid traction. For each boundary face carrying a pressure
condition it adds `p * n` at that face's nodes, weighted by the face normal
and orientation. Without it the mechanics sees no fluid load on the boundary,
and `example_poroelasticity` cannot be reproduced.

**Deliverables**

- `pymrst/modules/vemmech/utils/add_fluid_contrib_mech_vem.py`;
- wiring into `vem_lin_elast` alongside the existing interior term, so both
  can be active at once;
- a MATLAB reference on a boundary-pressure case that exercises the boundary
  term with the interior term zeroed, then both together.

**Care required.** The boundary term must respect the same Dirichlet-DOF
elimination the interior assembly uses (`isdirdofs` in the MRST signature),
otherwise it will load constrained nodes. Test that a fully clamped boundary
produces no spurious displacement.

**Size.** Small to medium. One day.

## Phase 4: examples end to end

This phase is what "matching MRST end to end" actually means.

| MRST example | lines | needs |
|---|---:|---|
| `examples/example_2D.m` | 35 | `squareTest` |
| `examples/example_2D_layers.m` | 35 | `squareLayersTest` |
| `examples/example_2D_complex_grid.m` | 97 | `squareTest`, `exploreSquareGrid` |
| `examples/compactionTest2D.m` | 69 | `makeCompactionTest`, `squareGrid` |
| `examples/compactionTest3D.m` | 93 | `makeCompactionTest`, `complex3DGrid` |
| `examples/example_poroelasticity.m` | 168 | `squareTest`, `lincomp_tpfa`, Phase 3 |
| `ad_version/calibration/simple_matching_example.m` | 145 | calibration stack (already ported) |

**Deliverables per example**

- a Python port under `pymrst/modules/vemmech/examples/`;
- `examples/validation/dump_vemmech_<name>_reference.m`;
- `examples/validation/validate_vemmech_<name>.py`.

**Where the risk is.** `example_poroelasticity` is the hard one. It couples
`lincomp_tpfa` with the VEM solve over a time loop, so matching MRST means
matching the coupling and the stepping, not a single assembly. Port it last,
after the other five have pinned the mechanics.

**Reference-size discipline.** Follow the two-tier pattern used for the
gridprocessing large-deck reference: full fields at a small size, counts plus
a fixed-index fingerprint of the sorted distributions at the larger size, so
the stored `.mat` files stay in the tens of kilobytes.

**Size.** Medium to large. Two to three days.

## Phase 5: pytest suite

vemmech has **no `tests/` directory**. `tests/` currently holds `ad_blackoil`,
`ad_core` and `gridprocessing`. All 37 vemmech tests are
`examples/validation/validate_phase*.py` scripts, which pin saved values but
do not test invariants, error paths, or option handling.

**Deliverables**

- `tests/vemmech/` mirroring `tests/gridprocessing/`, covering:
  - assembly invariants: symmetry of the stiffness matrix, rigid-body modes in
    its null space, positive-definiteness on the constrained system;
  - patch tests: a linear displacement field must be reproduced exactly, which
    is what `linearDisplacement` exists for;
  - BC handling: clamped nodes stay clamped, force BCs sum to the applied
    load;
  - the fixtures from Phase 2 produce grids satisfying the standard
    `check_grid` topology invariants;
  - poroelastic coupling: zero pressure reproduces the dry solve.
- direct coverage for the eight ported entries the ledger reports as
  unvalidated (`C2D`, `VEM_div`, `VEM3D_div`, `calculateQC`, `lincompTPFA`,
  `Sigma2SmaxSminTheta`, `optimize_mech`, `compute_Ax_derivs`), each of which
  is currently only exercised through a caller;
- registration of the new tests in the Phase 1 ledger, so every ported entry
  is named by a test or a validator, taking the ledger's
  "ported but unvalidated" table to zero.

**Note on naming.** `tests/ad_core/` and `tests/gridprocessing/` already
collide on `test_parity_audit.py`, `test_parity_ledger.py` and
`test_compute_cp_geometry.py`, which blocks whole-suite collection. Any new
`tests/vemmech/` parity tests must use distinct basenames, or the directories
need `__init__.py`, so this phase does not deepen an existing problem.

**Size.** Medium. One to two days.

## Risks and open questions

- **Completeness of the ported files** has not been audited symbol by symbol.
  Import, the 37 validators and one 3D solve all pass, but a partially ported
  file with no validator covering the missing part would not show up. Phase 1
  has narrowed this to a named list: the eight entries the ledger reports as
  ported but unvalidated are exactly where such a gap could hide.
- **`LinearCompressibilityModel`'s thermal `linearTransport` branch** is
  unported, as recorded in `GAPS.md`. It is out of scope for this plan unless
  the coupled thermal path is wanted.
- **Coupled time stepping** in `example_poroelasticity` is the single largest
  unknown. If MRST's splitting scheme differs from the natural Python
  formulation, matching it may need a dedicated reference at the level of
  individual time steps rather than the final state.

## Effort summary

| phase | work | size |
|---|---|---|
| 1 | parity ledger and audit tooling | **done** |
| 2 | eight test-grid fixtures | 2-3 days |
| 3 | boundary fluid traction, wired and validated | 1 day |
| 4 | six examples plus dump/validate pairs | 2-3 days |
| 5 | `tests/vemmech/` invariant suite | 1-2 days |

**Status: 57 of 65 production files ported, 1 pending.** The parity ledger (`VEMMECH_PARITY.md`) is the live count.

MATLAB R2021b and the MRST 2024a tree are both available locally, so every
phase above can generate its own reference data - the same workflow that took
`gridprocessing` to 1e-14 agreement on a 72,000-cell deck.

## Defects found while implementing

Three bugs surfaced that had nothing to do with the new code, and all three
produced plausible-looking output rather than errors.

**Cell-face tags losing their alignment.** `cells.tags` runs parallel to
`cells.faces`, so anything that reorders or rebuilds the latter has to carry
the former. `_sort_cell_faces` permuted faces and left the tags behind, and
`triangulate_faces` dropped them entirely. The effect was that `pside` and
`boundary_face_indices` returned faces from the wrong side of the grid, so
every vemmech boundary condition built on them was silently wrong.

It showed up as `compactionTest2D` returning 2.24 times the analytic solution
on a triangulated grid while the untriangulated grid stayed exact to 1e-15.
The diagnosis went: not the twist; does not converge under refinement, so a
bug not discretization; patch test passes, so assembly is fine; body force
exact, so the load is fine; MRST gives ratio 1.0 on both grids, so it is ours;
grids match node for node, so triangulation is fine; and finally the boundary
conditions pinned 70 entries where MRST pinned 43.
`tests/gridprocessing/test_cell_face_tag_alignment.py` guards both defects.

**`remove_cells` required an `indexMap`.** A glued grid has none - `squareGrid`
drops it exactly as MRST does - and MRST's `removeCells` guards with `isfield`.
The port did not, so cutting cells out of a glued grid raised a `TypeError`.

**`lincomp_tpfa` built a mis-sized accumulation matrix.** With wells present
the system carries one row per control on top of the cells, but `ct` was built
from the cell-only pore volumes, so `matrix_output=True` with a well raised a
shape error. The well rows hold no accumulation and are now padded with zeros.
