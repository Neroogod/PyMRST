"""Phase 377: glue2DGrid against MRST (Phase 25 core leftovers).

Pins `glue_2d_grid` against MRST's `glue2DGrid` on two cases: (A) two matching
unit squares sharing an edge, and (B) a coarse square glued to two stacked
squares (needs the shared edge to be split to conform). Face renumbering differs,
so the check uses the cell/face/node counts and the sorted neighbour-pair
multiset (outside as -1, MRST's 1-based cells shifted to 0-based).
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.gridprocessing.glue_2d_grid import glue_2d_grid
from pymrst.gridprocessing.tessellation_grid import tessellation_grid

REF = Path(__file__).with_name("ref_p377_glue.mat")


def _check(name: str, ok: bool, detail: str = "") -> int:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{('  ' + detail) if detail else ''}")
    return 0 if ok else 1


def _sorted_pairs(neigh: np.ndarray) -> np.ndarray:
    nb = np.sort(np.asarray(neigh, dtype=np.int64), axis=1)
    return nb[np.lexsort((nb[:, 1], nb[:, 0]))]


def _cmp(tag, g, ref, key, failed):
    print(f"\nglue2DGrid case {tag} vs MRST:")
    failed[0] += _check(f"cell count ({g.cells.num})",
                        g.cells.num == int(ref[f"{key}_c"].ravel()[0]))
    failed[0] += _check(f"face count ({g.faces.num})",
                        g.faces.num == int(ref[f"{key}_f"].ravel()[0]))
    failed[0] += _check(f"node count ({g.nodes.num})",
                        g.nodes.num == int(ref[f"{key}_n"].ravel()[0]))
    pairs = _sorted_pairs(g.faces.neighbors)
    ref_pairs = np.asarray(ref[f"{key}_pairs"], dtype=np.int64).copy()
    ref_pairs[ref_pairs != -1] -= 1
    failed[0] += _check("neighbour-pair multiset matches",
                        pairs.shape == ref_pairs.shape and
                        np.array_equal(pairs, _sorted_pairs(ref_pairs)))


def main() -> int:
    if not REF.exists():
        print(f"missing {REF.name}; run dump_phase377_reference.m in MATLAB first")
        return 1
    ref = loadmat(REF)
    failed = [0]

    gA = glue_2d_grid(
        tessellation_grid(np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]]),
                          [[0, 1, 2, 3]]),
        tessellation_grid(np.array([[1., 0.], [2., 0.], [2., 1.], [1., 1.]]),
                          [[0, 1, 2, 3]]))
    _cmp("A (matching squares)", gA, ref, "A", failed)

    gB = glue_2d_grid(
        tessellation_grid(np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]]),
                          [[0, 1, 2, 3]]),
        tessellation_grid(
            np.array([[1., 0.], [2., 0.], [2., .5], [1., .5], [2., 1.], [1., 1.]]),
            [[0, 1, 2, 3], [3, 2, 4, 5]]))
    _cmp("B (non-conforming)", gB, ref, "B", failed)

    print("\nglue2DGrid matches MRST!" if failed[0] == 0
          else f"\n{failed[0]} checks FAILED")
    return 1 if failed[0] else 0


if __name__ == "__main__":
    raise SystemExit(main())
