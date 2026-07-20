"""Compare PyMRST's structural trap analysis against MATLAB co2lab-spillpoint.

MRST's cell-based trapAnalysis floods the caprock iteratively with a
graph search; PyMRST computes the same result with a priority flood.
Both answer the same question - to what depth does buoyant CO2 rise in
each cell, which cells are consequently trapped, and how much CO2 does
each trap hold - so the results must agree cell for cell.

Trap *labels* are arbitrary in both codes, so traps are matched by their
cell sets rather than by index.

Note: running MRST's cell-based algorithm on this machine required
replacing MatlabBGL's dfs.m and bfs.m, whose MEX binaries are not built
for Apple Silicon (originals kept as *.bgl-backup).
"""

from pathlib import Path

import numpy as np
import scipy.io

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.co2lab.trapping import trap_analysis, volumes_of_traps
from pymrst.modules.co2lab.ve import top_surface_grid

HERE = Path(__file__).parent

SURFACES = {
    "two_domes": lambda x, y: (
        1000.0 - 0.02 * x
        - 12.0 * np.exp(-(((x - 250.0) / 90.0) ** 2
                          + ((y - 500.0) / 200.0) ** 2))
        - 12.0 * np.exp(-(((x - 700.0) / 90.0) ** 2
                          + ((y - 500.0) / 200.0) ** 2))
    ),
    "multi_dome": lambda x, y: (
        1000.0 - 0.015 * x - 0.01 * y
        - 15.0 * np.exp(-(((x - 250.0) / 80.0) ** 2
                          + ((y - 250.0) / 80.0) ** 2))
        - 10.0 * np.exp(-(((x - 700.0) / 90.0) ** 2
                          + ((y - 300.0) / 90.0) ** 2))
        - 18.0 * np.exp(-(((x - 500.0) / 70.0) ** 2
                          + ((y - 750.0) / 70.0) ** 2))
    ),
}


def match_traps(py_traps: np.ndarray, ml_traps: np.ndarray) -> dict:
    """Map MATLAB trap labels to PyMRST ones by comparing cell sets."""
    mapping = {}
    for i in range(1, int(ml_traps.max()) + 1):
        sel = ml_traps == i
        overlap = np.bincount(py_traps[sel], minlength=py_traps.max() + 1)
        overlap[0] = 0                      # ignore untrapped
        mapping[i] = int(overlap.argmax())
    return mapping


def check_case(name: str) -> int:
    path = HERE / f"ref_p10_{name}.mat"
    if not path.exists():
        print(f"{path.name}: missing (run dump_phase10_reference.m)")
        return 1
    ref = scipy.io.loadmat(path, squeeze_me=True)
    nx, ny = int(ref["nx"]), int(ref["ny"])
    Lx, Ly = float(ref["Lx"]), float(ref["Ly"])

    g3 = cart_grid([nx, ny, 1], [Lx, Ly, 20.0])
    x, y = g3.nodes.coords[:, 0], g3.nodes.coords[:, 1]
    g3.nodes.coords[:, 2] += SURFACES[name](x, y)
    ts = top_surface_grid(compute_geometry(g3))

    failed = 0

    def chk(label, ok, detail=""):
        nonlocal failed
        print(f"  {'ok  ' if ok else 'FAIL'} {label}{detail}")
        failed += 0 if ok else 1

    d = float(np.abs(ts.z - np.asarray(ref["z_ve"], float).ravel()).max())
    chk("caprock depth z", d < 1e-9, f" (max diff {d:.2e} m)")

    res = trap_analysis(ts)
    ml_traps = np.asarray(ref["traps"], np.int64).ravel()
    ml_trap_z = np.atleast_1d(np.asarray(ref["trap_z"], float))
    ml_vols = np.atleast_1d(np.asarray(ref["vols"], float))
    n_ml = int(ref["ntraps"])

    chk("number of traps", res.n_traps == n_ml,
        f" ({res.n_traps} vs {n_ml})")
    if res.n_traps != n_ml:
        return failed

    # same set of trapped cells, regardless of labelling
    chk("trapped cells", np.array_equal(res.traps > 0, ml_traps > 0),
        f" ({int((res.traps > 0).sum())} vs {int((ml_traps > 0).sum())})")

    mapping = match_traps(res.traps, ml_traps)
    exact = all(
        np.array_equal(ml_traps == i, res.traps == mapping[i])
        for i in range(1, n_ml + 1)
    )
    chk("trap regions (cell sets)", exact)

    poro = np.full(ts.g.cells.num, 0.2)
    py_vols = volumes_of_traps(ts, res, poro=poro)

    dz = max(abs(ml_trap_z[i - 1] - res.trap_z[mapping[i] - 1])
             for i in range(1, n_ml + 1))
    chk("spill depths", dz < 1e-9, f" (max diff {dz:.2e} m)")

    dv = max(abs(ml_vols[i - 1] - py_vols[mapping[i] - 1])
             for i in range(1, n_ml + 1))
    scale = float(ml_vols.max())
    chk("trap capacities", dv < 1e-9 * scale,
        f" (max rel diff {dv / scale:.2e})")
    return failed


def main() -> int:
    total = 0
    for name in SURFACES:
        print(f"{name}:")
        total += check_case(name)
    print("\nTrap analysis matches MATLAB co2lab-spillpoint!" if total == 0
          else f"\n{total} comparisons FAILED")
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
