"""Compare PyMRST process_grdecl against MATLAB processGRDECL.

Each reference file carries the raw grdecl arrays (so both sides process
identical input) plus order-independent invariants: cell volumes and
centroids keyed by indexMap, and per-cell-pair aggregated connection
areas and TPFA transmissibilities.
"""

from pathlib import Path

import numpy as np
import scipy.io

from pymrst.gridprocessing import compute_geometry
from pymrst.gridprocessing.process_grdecl import process_grdecl
from pymrst.params import make_rock
from pymrst.solvers import compute_trans

HERE = Path(__file__).parent
CASES = ["ref_p6_spe1.mat", "ref_p6_fault.mat", "ref_p6_johansen.mat"]


def invariants(g):
    rock = make_rock(g, 1.0, 0.3)
    ht = compute_trans(g, rock)
    t = 1.0 / np.bincount(g.cells.faces, weights=1.0 / ht,
                          minlength=g.faces.num)
    n = g.faces.neighbors
    internal = np.all(n >= 0, axis=1)
    imap = g.cells.indexMap
    pair = np.sort(np.column_stack([imap[n[internal, 0]],
                                    imap[n[internal, 1]]]), axis=1)
    upair, ic = np.unique(pair, axis=0, return_inverse=True)
    return {
        "cells_num": g.cells.num, "faces_num": g.faces.num,
        "nodes_num": g.nodes.num, "indexMap": np.sort(imap),
        "volumes": g.cells.volumes[np.argsort(imap)],
        "centroids": g.cells.centroids[np.argsort(imap)],
        "upair": upair,
        "conn_area": np.bincount(ic, weights=g.faces.areas[internal]),
        "conn_trans": np.bincount(ic, weights=t[internal]),
        "conn_count": np.bincount(ic),
        "total_bnd_area": float(g.faces.areas[~internal].sum()),
    }


def compare_case(name: str) -> int:
    ref = scipy.io.loadmat(HERE / name, squeeze_me=True)
    grdecl = {
        "cartDims": tuple(int(v) for v in np.atleast_1d(ref["cartDims"])),
        "COORD": np.asarray(ref["COORD"], float).ravel(),
        "ZCORN": np.asarray(ref["ZCORN"], float).ravel(),
        # MATLAB uses actnum(:), i.e. column-major flattening (the
        # Johansen ActiveCells file loads as a 25-column matrix).
        "ACTNUM": np.asarray(ref["ACTNUM"], float).ravel(order="F"),
    }
    g = compute_geometry(process_grdecl(grdecl))
    py = invariants(g)

    failed = 0

    def chk(label, ok, detail=""):
        nonlocal failed
        print(f"  {'ok  ' if ok else 'FAIL'} {label}{detail}")
        failed += 0 if ok else 1

    chk("cells.num", py["cells_num"] == int(ref["cells_num"]),
        f" ({py['cells_num']} vs {int(ref['cells_num'])})")
    chk("faces.num", py["faces_num"] == int(ref["faces_num"]),
        f" ({py['faces_num']} vs {int(ref['faces_num'])})")
    chk("nodes.num", py["nodes_num"] == int(ref["nodes_num"]),
        f" ({py['nodes_num']} vs {int(ref['nodes_num'])})")

    ml_imap = np.sort(np.asarray(ref["indexMap"]).ravel()) - 1
    chk("indexMap", np.array_equal(py["indexMap"], ml_imap))

    order = np.argsort(np.asarray(ref["indexMap"]).ravel())
    ml_vol = np.asarray(ref["volumes"]).ravel()[order]
    ml_cen = np.atleast_2d(ref["centroids"])[order]
    dv = np.max(np.abs(py["volumes"] - ml_vol)) / ml_vol.max()
    chk("volumes", dv < 1e-11, f" (rel {dv:.2e})")
    dc = np.max(np.abs(py["centroids"] - ml_cen))
    chk("centroids", dc < 1e-7, f" (abs {dc:.2e})")

    ml_pair = np.atleast_2d(ref["upair"]).astype(np.int64) - 1
    ml_order = np.lexsort((ml_pair[:, 1], ml_pair[:, 0]))
    same_conns = (py["upair"].shape == ml_pair.shape
                  and np.array_equal(py["upair"], ml_pair[ml_order]))
    chk("connection pairs", same_conns,
        f" ({py['upair'].shape[0]} vs {ml_pair.shape[0]})")
    if same_conns:
        for key, tol in (("conn_area", 1e-9), ("conn_trans", 1e-9),
                         ("conn_count", 0.0)):
            ml = np.asarray(ref[key]).ravel()[ml_order]
            d = np.max(np.abs(py[key] - ml)) / max(np.abs(ml).max(), 1e-300)
            chk(key, d <= tol + 1e-15, f" (rel {d:.2e})")
    db = abs(py["total_bnd_area"] - float(ref["total_bnd_area"])) \
        / float(ref["total_bnd_area"])
    chk("boundary area", db < 1e-11, f" (rel {db:.2e})")
    return failed


def main() -> int:
    total, missing = 0, 0
    for name in CASES:
        if not (HERE / name).exists():
            print(f"{name}: missing (run dump_phase6_reference.m)")
            missing += 1
            continue
        print(f"{name}:")
        total += compare_case(name)
    if missing == len(CASES):
        return 1
    print("\nAll corner-point grids match!" if total == 0
          else f"\n{total} comparisons FAILED")
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
