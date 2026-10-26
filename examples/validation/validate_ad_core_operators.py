"""Validate MRST ``setupOperatorsTPFA`` and ``faceUpstr``: the ports in
`ad_core/utils/setup_operators_tpfa.py` and `ad_core/utils/face_upstr.py`.

The reference is MRST 2024a (`dump_ad_core_operators_reference.m`) on a
4x3x2 grid with heterogeneous anisotropic permeability, porosity and
net-to-gross: every field (`T`, `T_all`, `N`, `internalConn`, `pv`, `C`, `M`)
and every operator (`Grad`, `Div`, `AccDiv`, `faceAvg`, `faceUpstr`) on test
vectors; grid NNCs (one new pair, one duplicating an existing connection,
one touching the outside, and one more), transmissibilities given per
internal interface, per face and with explicit neighbours, the grid's own
`TRANS`, and the `porv` option; `faceUpstr` with scalar flags and several
columns. Indices compare as MRST's less one.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

HERE = Path(__file__).resolve().parent
RTOL = 1e-14


def report(ok: bool, text: str) -> bool:
    print(f"{'ok  ' if ok else 'FAIL'} {text}")
    return ok


def _close(name: str, a, b) -> bool:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return report(False, f"{name}: shape {a.shape} vs MRST {b.shape}")
    err = float(np.abs(a - b).max()/max(np.abs(b).max(), 1e-300)) if b.size else 0.0
    return report(err < RTOL, f"{name} rel. diff {err:.1e}")


def _same(name: str, a, b) -> bool:
    return report(np.array_equal(np.asarray(a), np.asarray(b)), f"{name} identical")


def main() -> int:
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_core.utils.face_upstr import face_upstr
    from pymrst.modules.ad_core.utils.setup_operators_tpfa import setup_operators_tpfa
    from pymrst.params.rock import make_rock

    ref = loadmat(HERE/"ref_ad_core_operators.mat", squeeze_me=True, struct_as_record=False)
    base, nnc, given, upstr = ref["base"], ref["nnc"], ref["given"], ref["upstr"]
    x, f, flag = ref["x"], ref["f"], np.asarray(ref["flag"], dtype=bool)

    G = compute_geometry(cart_grid([4, 3, 2], [40.0, 30.0, 20.0]))
    nc = G.cells.num
    perm = (1 + np.arange(1, nc + 1))*1e-13
    rock = make_rock(G, np.column_stack([perm, 2*perm, 0.5*perm]), 0.1 + 0.01*np.arange(1, nc + 1),
                     ntg=0.5 + 0.02*np.arange(1, nc + 1))
    op = setup_operators_tpfa(G, rock)
    ok = True
    for name in ("T", "T_all", "pv"):
        ok &= _close(f"{name}", getattr(op, name), getattr(base, name))
    ok &= _same("N", op.N + 1, base.N)
    ok &= _same("internalConn", op.internalConn, np.asarray(base.internalConn, dtype=bool))
    ok &= _same("C", op.C.toarray(), base.C)
    ok &= _same("M", op.M.toarray(), base.M)
    for name, actual in (("Grad", op.Grad(x)), ("Div", op.Div(f)), ("AccDiv", op.AccDiv(x, f)),
                         ("faceAvg", op.faceAvg(x)), ("faceUpstr", op.faceUpstr(flag, x)),
                         ("faceUpstr, scalar flag", op.faceUpstr(True, x)),
                         ("faceUpstr, two columns", op.faceUpstr(flag, np.column_stack([x, 2*x])))):
        key = {"Grad": "grad", "Div": "div", "AccDiv": "accdiv", "faceAvg": "avg",
               "faceUpstr": "upw", "faceUpstr, scalar flag": "upw_true",
               "faceUpstr, two columns": "upw_matrix"}[name]
        ok &= _close(name, actual, getattr(base, key))

    G.nnc = {"cells": np.array([[1, 24], [2, 1], [0, 5], [7, 18]]) - 1,
             "trans": np.array([5.0, 9.0, 7.0, 3.0])}
    opn = setup_operators_tpfa(G, rock)
    ok &= _same("grid NNCs: N", opn.N + 1, nnc.N)
    ok &= _close("grid NNCs: T", opn.T, nnc.T)
    ok &= _close("grid NNCs: T_all", opn.T_all, nnc.T_all)
    ok &= _same("grid NNCs: internalConn", opn.internalConn, np.asarray(nnc.internalConn, dtype=bool))
    del G.nnc

    n_int = int(np.all(G.faces.neighbors >= 0, axis=1).sum())
    opi = setup_operators_tpfa(G, rock, trans=np.arange(1, n_int + 1, dtype=float))
    ok &= _close("trans per internal interface: T", opi.T, given.T_int)
    ok &= _close("trans per internal interface: T_all", opi.T_all, given.T_all_int)
    opa = setup_operators_tpfa(G, rock, trans=np.arange(1, G.faces.num + 1)/10)
    ok &= _close("trans per face: T", opa.T, given.T_full)
    ok &= _close("trans per face: T_all", opa.T_all, given.T_all_full)
    ope = setup_operators_tpfa(G, rock, neighbors=np.array([[1, 2], [2, 3], [1, 5]]) - 1,
                               trans=np.array([4.0, 5.0, 6.0]))
    ok &= _same("explicit neighbours: N", ope.N + 1, given.N_pairs)
    ok &= _close("explicit neighbours: T", ope.T, given.T_pairs)
    ok &= _close("explicit neighbours: T_all", ope.T_all, given.T_all_pairs)
    ok &= _same("explicit neighbours: internalConn", ope.internalConn,
                np.asarray(given.conn_pairs, dtype=bool))
    ok &= _close("explicit neighbours: Grad", ope.Grad(x), given.grad_pairs)
    G.faces.TRANS = np.arange(1, G.faces.num + 1)*2.0
    ok &= _close("the grid's TRANS: T", setup_operators_tpfa(G, rock).T, given.T_grid)
    del G.faces.TRANS
    ok &= _close("porv option", setup_operators_tpfa(G, rock, porv=np.arange(1.0, nc + 1)).pv,
                 given.pv_porv)

    N, sz = op.N, (op.N.shape[0], nc)
    ok &= _close("faceUpstr, two columns", face_upstr(flag, np.column_stack([x, x**2]), N, sz),
                 upstr.xu)
    xu, xc = face_upstr(False, x, N, sz, nargout=2)
    ok &= _close("faceUpstr, scalar false", xu, upstr.xu_scalar)
    ok &= _close("faceUpstr, second output", xc, upstr.xc)
    print("\nsetupOperatorsTPFA and faceUpstr match MRST" if ok
          else "\nsetupOperatorsTPFA and faceUpstr do NOT match MRST")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
