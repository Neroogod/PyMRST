"""Phase 21: node-based trap analysis against MRST co2lab-spillpoint.

Compares `computeNodeTraps` at the node level - the summits, the spill
depths, the trapped nodes - and its projection onto cells, on a two-dome
caprock where the updip trap spills into the downdip one.
"""

import sys
from pathlib import Path

import numpy as np
import scipy.io

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pymrst.gridprocessing import cart_grid, compute_geometry       # noqa: E402
from pymrst.modules.co2lab.node_trapping import (                   # noqa: E402
    node_trap_analysis,
    nodes_to_cells,
)
from pymrst.modules.co2lab.ve import top_surface_grid               # noqa: E402

REF = Path(__file__).with_name("ref_p21_node_trapping.mat")


def build():
    nx, ny = 60, 44
    g = cart_grid([nx, ny, 1], [800.0, 600.0, 20.0])
    x, y = g.nodes.coords[:, 0], g.nodes.coords[:, 1]
    g.nodes.coords[:, 2] += (
        1000.0 - 0.02 * x
        - 10.0 * np.exp(-(((x - 250.0) / 80.0) ** 2
                          + ((y - 300.0) / 90.0) ** 2))
        - 14.0 * np.exp(-(((x - 600.0) / 85.0) ** 2
                          + ((y - 300.0) / 90.0) ** 2)))
    return top_surface_grid(compute_geometry(g))


def match(mine: np.ndarray, theirs: np.ndarray, n: int) -> dict:
    """Trap labels are arbitrary; match them by overlap."""
    m = {}
    for i in range(1, n + 1):
        sel = theirs == i
        if not np.any(sel):
            continue
        cnt = np.bincount(mine[sel], minlength=int(mine.max()) + 1)
        cnt[0] = 0
        m[i] = int(np.argmax(cnt))
    return m


def main() -> int:
    if not REF.exists():
        print(f"{REF.name} missing (run dump_phase21_reference.m in MATLAB)")
        return 0

    r = scipy.io.loadmat(REF, squeeze_me=True)
    ts = build()
    res = node_trap_analysis(ts)
    cell = nodes_to_cells(ts, res)

    # MRST's node ordering is not PyMRST's, and node indices are internal
    # bookkeeping - so everything node-level is compared by *position*. Build
    # the permutation from MRST's nodes to ours once, and apply it throughout.
    ml_xy = np.atleast_2d(np.asarray(r["node_xy"], float))[:, :2]
    my_xy = ts.g.nodes.coords[:, :2]
    key = lambda a: np.round(a, 6)                        # noqa: E731
    lut = {tuple(k): i for i, k in enumerate(map(tuple, key(my_xy)))}
    perm = np.array([lut[tuple(k)] for k in map(tuple, key(ml_xy))])
    assert perm.size == my_xy.shape[0] and len(set(perm.tolist())) == perm.size, (
        "MRST's nodes do not map one-to-one onto PyMRST's")

    ok = True

    def chk(name, good, detail=""):
        nonlocal ok
        ok &= bool(good)
        print(f"  {'ok  ' if good else 'FAIL'} {name:34s} {detail}")

    # ---- the surface both codes see ---------------------------------------
    dz = np.abs(ts.node_z[perm]
                - np.asarray(r["node_z"], float).ravel()).max()
    chk("node depths", dz < 1e-9, f"max diff = {dz:.2e} m")

    # ---- node level --------------------------------------------------------
    ml_traps = np.asarray(r["node_traps"], int).ravel()
    ml_n = int(r["n_traps"])
    chk("number of node traps", res.n_traps == ml_n,
        f"{res.n_traps} vs {ml_n}")

    if res.n_traps == ml_n:
        my_traps = res.traps[perm]              # into MRST's node order
        chk("trapped nodes (as a set)",
            np.array_equal(my_traps > 0, ml_traps > 0),
            f"{int((my_traps > 0).sum())} vs {int((ml_traps > 0).sum())}")

        mapping = match(my_traps, ml_traps, ml_n)
        same = all(np.array_equal(ml_traps == i, my_traps == j)
                   for i, j in mapping.items())
        chk("each node trap, node for node", same)

        ml_z = np.atleast_1d(np.asarray(r["node_trap_z"], float))
        mine_z = np.array([res.trap_z[mapping[i] - 1] for i in sorted(mapping)])
        want_z = np.array([ml_z[i - 1] for i in sorted(mapping)])
        d = np.abs(mine_z - want_z).max()
        chk("node spill depths", d < 1e-9, f"max diff = {d:.2e} m")

    # ---- projected onto cells ----------------------------------------------
    ml_ct = np.asarray(r["cell_traps"], int).ravel()
    ml_cn = int(ml_ct.max())
    chk("number of cell traps", cell.n_traps == ml_cn,
        f"{cell.n_traps} vs {ml_cn}")

    if cell.n_traps == ml_cn:
        chk("trapped cells (as a set)",
            np.array_equal(cell.traps > 0, ml_ct > 0),
            f"{int((cell.traps > 0).sum())} vs {int((ml_ct > 0).sum())}")

        cmap = match(cell.traps, ml_ct, ml_cn)
        same = all(np.array_equal(ml_ct == i, cell.traps == j)
                   for i, j in cmap.items())
        chk("each cell trap, cell for cell", same)

        ml_cz = np.atleast_1d(np.asarray(r["cell_trap_z"], float))
        mine_cz = np.array([cell.trap_z[cmap[i] - 1] for i in sorted(cmap)])
        want_cz = np.array([ml_cz[i - 1] for i in sorted(cmap)])
        d = np.abs(mine_cz - want_cz).max()
        chk("cell spill depths", d < 1e-9, f"max diff = {d:.2e} m")

    print(f"\n  (MRST: {ml_n} node traps over "
          f"{int((ml_traps > 0).sum())} nodes, projecting to "
          f"{int((ml_ct > 0).sum())} cells)")
    print("\nNode-based trap analysis matches MATLAB!" if ok
          else "\ncomparisons FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
