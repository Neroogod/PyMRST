"""Phase 379: removeShortEdges against MRST (Phase 25 core leftovers).

Pins `remove_short_edges` against MRST's `removeShortEdges` on a pentagon with a
sliver top edge (1e-4 long) collapsed at tol 1e-3 into a quad. The cell/face/node
counts and the sorted node coordinates (the sliver endpoints merge to their
average) are compared number for number.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.gridprocessing.remove_short_edges import remove_short_edges
from pymrst.gridprocessing.tessellation_grid import tessellation_grid

REF = Path(__file__).with_name("ref_p379_shortedges.mat")


def _check(name: str, ok: bool, detail: str = "") -> int:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{('  ' + detail) if detail else ''}")
    return 0 if ok else 1


def main() -> int:
    if not REF.exists():
        print(f"missing {REF.name}; run dump_phase379_reference.m in MATLAB first")
        return 1
    ref = loadmat(REF)

    p = np.array([[0., 0.], [1., 0.], [1., 1.], [0.5, 1.0001], [0.5, 1.0]])
    g = tessellation_grid(p, [[0, 1, 2, 3, 4]])
    g = remove_short_edges(g, tol=1e-3)

    failed = 0
    print("removeShortEdges (pentagon with sliver) vs MRST:")
    failed += _check(f"cell count ({g.cells.num})",
                     g.cells.num == int(ref["cells_num"].ravel()[0]))
    failed += _check(f"face count ({g.faces.num})",
                     g.faces.num == int(ref["faces_num"].ravel()[0]))
    failed += _check(f"node count ({g.nodes.num})",
                     g.nodes.num == int(ref["nodes_num"].ravel()[0]))

    coords = np.round(np.asarray(g.nodes.coords, dtype=float), 8)
    coords = coords[np.lexsort(coords.T[::-1])]
    ref_coords = np.asarray(ref["coords"], dtype=float)
    err = np.max(np.abs(coords - ref_coords)) if coords.shape == ref_coords.shape else np.inf
    failed += _check("merged node coordinates match", err < 1e-8,
                     f"max|dx| = {err:.3e}")

    print("\nremoveShortEdges matches MRST!" if failed == 0
          else f"\n{failed} checks FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
