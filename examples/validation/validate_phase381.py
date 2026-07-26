"""Phase 381: removePinch against MRST (Phase 25 core leftovers).

Pins `remove_pinch` against MRST's `removePinch`. Two unit squares are
concatenated with their shared-edge nodes and faces duplicated (as a pinched
layer produces), then removePinch runs. The cell/face/node counts and sorted
neighbour-pair multiset are compared - both codes must merge the duplicates and
reconnect the two cells.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.gridprocessing.glue_2d_grid import _concatenate
from pymrst.gridprocessing.grid_structure import Cells, Faces, Grid, Nodes
from pymrst.gridprocessing.remove_pinch import remove_pinch
from pymrst.gridprocessing.tessellation_grid import tessellation_grid

REF = Path(__file__).with_name("ref_p381_pinch.mat")


def _check(name: str, ok: bool, detail: str = "") -> int:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{('  ' + detail) if detail else ''}")
    return 0 if ok else 1


def _sorted_pairs(neigh: np.ndarray) -> np.ndarray:
    nb = np.sort(np.asarray(neigh, dtype=np.int64), axis=1)
    return nb[np.lexsort((nb[:, 1], nb[:, 0]))]


def main() -> int:
    if not REF.exists():
        print(f"missing {REF.name}; run dump_phase381_reference.m in MATLAB first")
        return 1
    ref = loadmat(REF)

    g1 = tessellation_grid(np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]]),
                           [[0, 1, 2, 3]])
    g2 = tessellation_grid(np.array([[1., 0.], [2., 0.], [2., 1.], [1., 1.]]),
                           [[0, 1, 2, 3]])
    c = _concatenate(g1, g2)
    g = Grid(cells=Cells(num=c.cells.num, facePos=c.cells.facePos,
                         faces=c.cells.faces, indexMap=np.arange(c.cells.num)),
             faces=Faces(num=c.faces.num, nodePos=c.faces.nodePos,
                         nodes=c.faces.nodes, neighbors=c.faces.neighbors),
             nodes=Nodes(num=c.nodes.num, coords=c.nodes.coords),
             griddim=2, type=["concat"])
    g = remove_pinch(g)

    failed = 0
    print("removePinch (duplicated squares) vs MRST:")
    failed += _check(f"cell count ({g.cells.num})",
                     g.cells.num == int(ref["cells_num"].ravel()[0]))
    failed += _check(f"face count ({g.faces.num})",
                     g.faces.num == int(ref["faces_num"].ravel()[0]))
    failed += _check(f"node count ({g.nodes.num})",
                     g.nodes.num == int(ref["nodes_num"].ravel()[0]))
    pairs = _sorted_pairs(g.faces.neighbors)
    ref_pairs = np.asarray(ref["pairs"], dtype=np.int64).copy()
    ref_pairs[ref_pairs != -1] -= 1
    failed += _check("neighbour-pair multiset matches",
                     pairs.shape == ref_pairs.shape and
                     np.array_equal(pairs, _sorted_pairs(ref_pairs)))

    print("\nremovePinch matches MRST!" if failed == 0
          else f"\n{failed} checks FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
