"""Compare PyMRST's 2D EDFM against MATLAB hfm.

Checks the geometric heart of EDFM (conductivity index and embedded
fracture length per matrix cell), then the NNC list, the NNC
transmissibilities, and finally the coupled pressure solution.

PyMRST integrates the average fracture distance <d> exactly (the matrix
cell polygon is clipped by the fracture line and the linear distance
function integrated on each side); MRST evaluates the same integral with
a Delaunay quadrature whose points straddle the fracture, so the two
agree to round-off.

The fracture cell count is taken from the MRST reference, because
assembleFracNodes2D snaps it through a float-fragile rule; and the NNC
transmissibility is weighted by cell volume, because the hfm examples
never set rock.poro and computeEffectiveTrans then falls back to volumes.
"""

from pathlib import Path

import numpy as np
import scipy.io

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.hfm import FractureLine, build_edfm, solve_incomp_edfm
from pymrst.utils.units import BARSA, CENTI, DARCY, POISE

HERE = Path(__file__).parent


def run_python(ref):
    g = compute_geometry(cart_grid([30, 30], [9.0, 9.0]))
    perm_m = np.full(g.cells.num, 1.0 * DARCY)
    poro_m = np.full(g.cells.num, 0.2)
    ln = FractureLine(np.asarray(ref["fl"], float).ravel())
    # Use MRST's fracture cell count (from its saved fracture nodes) so
    # both sides discretize the fracture identically.
    n_frac = np.atleast_2d(np.asarray(ref["frac_nodes"], float)).shape[0] - 1
    e = build_edfm(g, [ln], aperture=1.0 / 25.0, perm_matrix=perm_m,
                   perm_frac=1e4 * DARCY, poro_matrix=poro_m,
                   poro_frac=1.0, n_cells=[n_frac],
                   trans_weight="volume")

    fc = g.faces.centroids
    bd = np.any(g.faces.neighbors == -1, axis=1)
    cells, tb, pb = [], [], []
    for sel, pval in (((fc[:, 0] < 1e-9), 10 * BARSA),
                      ((fc[:, 0] > 9.0 - 1e-9), 1 * BARSA)):
        faces = np.flatnonzero(bd & sel)
        c = g.faces.neighbors[faces].sum(axis=1) + 1
        dist = np.abs(fc[faces, 0] - g.cells.centroids[c, 0])
        cells.append(c)
        tb.append(perm_m[c] * g.faces.areas[faces] / dist)
        pb.append(np.full(c.size, pval))
    p = solve_incomp_edfm(e, 1 * CENTI * POISE, np.concatenate(cells),
                          np.concatenate(tb), np.concatenate(pb))
    return e, p


def main() -> int:
    path = HERE / "ref_p7c_edfm.mat"
    if not path.exists():
        print("ref_p7c_edfm.mat missing (run dump_phase7c_reference.m)")
        return 1
    ref = scipy.io.loadmat(path, squeeze_me=True)
    e, p = run_python(ref)

    failed = 0

    def chk(name, ok, detail=""):
        nonlocal failed
        print(f"  {'ok  ' if ok else 'FAIL'} {name}{detail}")
        failed += 0 if ok else 1

    # --- geometry: cells crossed, embedded length, conductivity index ---
    nm = e.n_matrix
    fa_ml = np.asarray(ref["fA_cell"], float).ravel()
    ci_ml = np.asarray(ref["CI_cell"], float).ravel()

    fa_py = np.zeros(nm)
    ci_py = np.zeros(nm)
    fm = np.array(e.conn_type) == "frac_matrix"
    for (m, _), ci in zip(e.conn[fm], e.ci[fm]):
        ci_py[m] += ci
    # embedded length per matrix cell, recomputed from the clipper
    from pymrst.modules.hfm.edfm2d import _seg_in_cell
    ln = FractureLine(np.asarray(ref["fl"], float).ravel())
    for m in range(nm):
        s = _seg_in_cell(e.g, m, ln)
        if s is not None:
            fa_py[m] = float(np.linalg.norm(s[1] - s[0]))

    same_cells = np.array_equal(fa_py > 1e-9, fa_ml > 1e-9)
    chk("matrix cells crossed by fracture", same_cells,
        f" ({int((fa_py > 1e-9).sum())} vs {int((fa_ml > 1e-9).sum())})")

    sel = fa_ml > 1e-9
    d_len = np.max(np.abs(fa_py[sel] - fa_ml[sel])) / fa_ml[sel].max()
    chk("embedded fracture length", d_len < 1e-10, f" (rel {d_len:.2e})")
    assert np.isclose(fa_py.sum(), ln.length, rtol=1e-10)

    d_ci = np.max(np.abs(ci_py[sel] - ci_ml[sel])) / ci_ml[sel].max()
    chk("conductivity index CI", d_ci < 1e-10, f" (rel {d_ci:.2e})")

    # --- NNC list and transmissibilities ---
    nnc_ml = np.atleast_2d(np.asarray(ref["nnc_cells"], np.int64)) - 1
    t_ml = np.asarray(ref["nnc_T"], float).ravel()
    order_ml = np.lexsort((nnc_ml[:, 1], nnc_ml[:, 0]))
    py_pairs = e.conn[fm]
    order_py = np.lexsort((py_pairs[:, 1], py_pairs[:, 0]))
    chk("NNC count", py_pairs.shape[0] == nnc_ml.shape[0],
        f" ({py_pairs.shape[0]} vs {nnc_ml.shape[0]})")
    if py_pairs.shape[0] == nnc_ml.shape[0]:
        same = np.array_equal(py_pairs[order_py], nnc_ml[order_ml])
        chk("NNC cell pairs", same)
        if same:
            t_py = e.trans[fm][order_py]
            d_t = np.max(np.abs(t_py - t_ml[order_ml])) / t_ml.max()
            chk("NNC transmissibility", d_t < 1e-10, f" (rel {d_t:.2e})")

    # --- coupled pressure solution ---
    p_ml = np.asarray(ref["p"], float).ravel()
    chk("total cells", p.size == p_ml.size,
        f" ({p.size} vs {p_ml.size})")
    if p.size == p_ml.size:
        scale = p_ml.max() - p_ml.min()
        d_p = np.max(np.abs(p - p_ml)) / scale
        chk("pressure (matrix + fracture)", d_p < 1e-10, f" (rel {d_p:.2e})")

    print("\nEDFM matches MATLAB hfm!" if failed == 0
          else f"\n{failed} comparisons FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
