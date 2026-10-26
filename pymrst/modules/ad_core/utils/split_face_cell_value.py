"""Split a face/cell-valued vector into its face and cell parts (MRST
`splitFaceCellValue`).

Some discretizations hand a single vector `x` that packs both face values and
cell values, and its length tells you which layout it is in: cell-only (needs
upstream weighting to get face values), face-then-cell, or half-face-then-cell.
This routine reads that length against the grid sizes `(nf, nc)` and returns the
two pieces `(fx, cx)` - the face value and the cell value - so a flux term can
use each where it belongs. It is the companion to `faceUpstr`.
"""

import numpy as np

from pymrst.utils.adi import value


def _numel(x) -> int:
    return int(np.asarray(value(x)).shape[0])


def split_face_cell_value(operators, flag: np.ndarray, x, sz):
    """Return `(face_value, cell_value)` from a packed vector `x`.

    `sz = (nf, nc)` is the interior-face count and the cell count. The layout is
    inferred from `len(x)`: `nc` (cell values, upstream-weighted to faces),
    `nf + nc` (face then cell), or `2*nf + nc` (both half-faces then cell).
    """
    nf, nc = int(sz[0]), int(sz[1])
    n = _numel(x)
    flag = np.asarray(flag)

    if n == nc:
        fx = operators.upw(flag, x)
        cx = x
    elif n == nf + nc:
        fx = x[:nf]
        cx = x[nf:]
    elif n == 2 * nf + nc:
        subs = np.arange(nf) + (~flag) * nf
        fx = x[subs]
        cx = x[2 * nf:]
    elif n == nf:
        raise NotImplementedError("face-only split is not implemented in MRST")
    elif n == 2 * nf:
        raise NotImplementedError("half-face-only split is not implemented in MRST")
    else:
        raise ValueError(f"unexpected input length {n} for sz=({nf}, {nc})")
    return fx, cx


#: MRST's own spelling. This port names functions in snake_case; the
#: alias keeps the MATLAB name callable, which is the convention the
#: rest of the package already follows.
splitFaceCellValue = split_face_cell_value

if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_core.utils.operators import setup_operators_tpfa
    from pymrst.params.rock import make_rock
    from pymrst.utils.units import DARCY, MILLI

    g = compute_geometry(cart_grid([4, 1, 1], [4.0, 1.0, 1.0]))
    rock = make_rock(g, 100 * MILLI * DARCY, 0.25)
    ops = setup_operators_tpfa(g, rock)
    nf = ops.N.shape[0]           # 3 interior faces
    nc = g.cells.num              # 4 cells
    flag = np.array([True, False, True])

    def test_cell_only_uses_upstream_weighting():
        """A cell-length vector is upstream-weighted to the faces, and the cell
        part is returned untouched."""
        x = np.array([10.0, 20.0, 30.0, 40.0])
        fx, cx = split_face_cell_value(ops, flag, x, (nf, nc))
        assert np.allclose(cx, x)
        assert np.allclose(fx, ops.upw(flag, x))
        print("ok test_cell_only_uses_upstream_weighting")

    def test_face_then_cell_layout_is_sliced():
        x = np.array([1.0, 2.0, 3.0, 100.0, 200.0, 300.0, 400.0])   # nf + nc
        fx, cx = split_face_cell_value(ops, flag, x, (nf, nc))
        assert np.allclose(fx, [1.0, 2.0, 3.0])
        assert np.allclose(cx, [100.0, 200.0, 300.0, 400.0])
        print("ok test_face_then_cell_layout_is_sliced")

    def test_half_face_picks_the_upwind_half():
        """2*nf half-face values: face i takes the first half where flag is
        set, the second half otherwise."""
        half = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])   # 2*nf
        x = np.concatenate([half, np.zeros(nc)])
        fx, cx = split_face_cell_value(ops, flag, x, (nf, nc))
        # flag = [T, F, T] -> subs = [0, 1+nf, 2] = [0, 4, 2] -> [1, 5, 3]
        assert np.allclose(fx, [1.0, 5.0, 3.0]), fx
        print("ok test_half_face_picks_the_upwind_half")

    test_cell_only_uses_upstream_weighting()
    test_face_then_cell_layout_is_sliced()
    test_half_face_picks_the_upwind_half()
    print("\nAll tests passed!")
