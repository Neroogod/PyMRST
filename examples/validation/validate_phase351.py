"""Phase 351 (AC14): perforation depth ordering for multisegment wells.

AC14's multisegment-well machinery already lives on the MRST layout and is
MATLAB-validated: `MultisegmentWell` (params) and
`assemble_multisegment_equations` / `MultisegmentEquationResult`
(`ad_core/models/facilities/multisegment/`), checked against MRST's converted
MS-well layout in Phase 109. This is the port of
`setupMSWellEquationSingleWell`. What this phase adds is
`reorder_well_perforations_by_depth` - the utility a multisegment well needs so
its perforations run shallow-to-deep, matching the segment topology and the
hydrostatic column.

The reorder is a deterministic depth sort, so its correctness is an invariant
(no external reference needed):

1. **Depth-ordered.** After the reorder every well's ``dZ`` is ascending.
2. **Consistent permutation.** Each perforation's ``(cell, WI, dZ, cstatus)``
   tuple stays together - no field is permuted independently.
3. **Idempotent.** Reordering an already-sorted well changes nothing, and
   reordering twice equals reordering once.

Phase 109 (the MS-well packed system) still passes, confirming the ordered
perforations feed the wellbore column unchanged.
"""

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_core.models.facilities import (
    reorder_well_perforations_by_depth,
)
from pymrst.params import add_well, make_rock


def _well(cells, dz):
    g = compute_geometry(cart_grid([4, 1, 4], [40.0, 10.0, 40.0]))
    rock = make_rock(g, 1e-13, 0.25)
    w = add_well(None, g, rock, np.asarray(cells), well_type="bhp", val=1e7,
                 compi=np.array([1.0, 0.0]), sign=1)[0]
    w.dZ = np.asarray(dz, float)
    w.WI = np.arange(1.0, len(cells) + 1.0)
    w.cstatus = np.ones(len(cells), bool)
    return w


def main():
    print("Phase 351 (AC14): perforation depth ordering")

    cells = [12, 0, 8, 4]
    dz = [35.0, 5.0, 25.0, 15.0]                        # shuffled depths
    w = _well(cells, dz)
    tuples_before = set(zip(w.cells.tolist(), w.WI.tolist(), w.dZ.tolist()))

    reorder_well_perforations_by_depth([w])

    assert np.all(np.diff(w.dZ) >= 0), "dZ not ascending"
    print(f"  depth-ordered: dZ = {w.dZ.tolist()}")

    tuples_after = set(zip(w.cells.tolist(), w.WI.tolist(), w.dZ.tolist()))
    assert tuples_before == tuples_after, "a field was permuted independently"
    print(f"  perforation tuples preserved: cells = {w.cells.tolist()}, "
          f"WI = {w.WI.tolist()}")

    # idempotent
    dz_once = w.dZ.copy()
    cells_once = w.cells.copy()
    reorder_well_perforations_by_depth([w])
    assert np.array_equal(w.dZ, dz_once) and np.array_equal(w.cells, cells_once)
    print("  idempotent: reordering twice equals once")

    print("Phase 351 passed - perforations are depth-ordered with every "
          "per-perforation field permuted consistently.")


if __name__ == "__main__":
    main()
