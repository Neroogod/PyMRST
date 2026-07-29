"""Phase 419: DG element cubatures against MRST.

Pins PyMRST's Triangle/Line/Tetrahedron cubatures against MRST's classes on a
2-D grid (TriangleCubature over cells, LineCubature over faces) and a 3-D grid
(TetrahedronCubature over cells, TriangleCubature over faces). For each element
the cubature points and weights are compared as sets (sorted), so the match is
independent of point ordering.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.dg.cubatures.element_cubature import (
    LineCubature,
    TetrahedronCubature,
    TriangleCubature,
)
from pymrst.modules.vem.utils.vem_geometry import compute_vem_geometry

REF = Path(__file__).with_name("ref_p419_dg_element_cubature.mat")


def _check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{('  ' + detail) if detail else ''}")
    return 0 if ok else 1


def _compare(cub, x_ref, w_ref, pos_ref):
    """Max per-element point/weight error after sorting each element's points."""
    pos_ref = pos_ref.astype(np.int64) - 1             # MATLAB 1-based
    if not np.array_equal(cub.pos, pos_ref):
        return np.inf, np.inf
    xe = we = 0.0
    for e in range(cub.pos.size - 1):
        i0, i1 = cub.pos[e], cub.pos[e + 1]
        P, W = cub.points[i0:i1], cub.weights[i0:i1]
        Pr, Wr = x_ref[i0:i1], w_ref[i0:i1]
        oi = np.lexsort(P.T[::-1])
        oj = np.lexsort(Pr.T[::-1])
        xe = max(xe, np.max(np.abs(P[oi] - Pr[oj])))
        we = max(we, np.max(np.abs(W[oi] - Wr[oj])))
    return xe, we


def main() -> int:
    if not REF.exists():
        print(f"missing {REF.name}; run dump_phase419_reference.m in MATLAB first")
        return 1
    ref = loadmat(REF)

    g2 = compute_geometry(cart_grid([3, 2], [3.0, 2.0]))
    g3 = compute_vem_geometry(compute_geometry(cart_grid([2, 2, 2], [2.0, 2.0, 2.0])))

    cases = [
        ("TriangleCubature (2-D cells)", TriangleCubature(g2, 3), "tri2"),
        ("LineCubature (2-D faces)", LineCubature(g2, 3), "line2"),
        ("TetrahedronCubature (3-D cells)", TetrahedronCubature(g3, 3), "tet3"),
        ("TriangleCubature (3-D faces)", TriangleCubature(g3, 3), "trif3"),
    ]

    failed = 0
    print("DG element cubatures vs MRST:")
    for name, cub, key in cases:
        xe, we = _compare(cub, np.asarray(ref[key + "_x"], float),
                          np.asarray(ref[key + "_w"], float).ravel(),
                          np.asarray(ref[key + "_pos"]).ravel())
        failed += _check(f"{name}: points & weights match",
                         xe < 1e-11 and we < 1e-11,
                         f"max|dx| {xe:.1e}, max|dw| {we:.1e}")

    print("\nDG element cubatures match MRST!" if failed == 0
          else f"\n{failed} checks FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
