"""Phase 415: incompVEM with TPFA flux reconstruction against MRST.

Pins PyMRST's `trans='tpfa'` option against MRST's `incompVEM` with
`computeVirtualIP(G, rock, k, 'trans', 'tpfa')` on a twisted 2-D grid: fluxes are
reconstructed by a TPFA scheme on the VEM cell/face pressures rather than by the
direct VEM gradient. Node pressures (unaffected by the flux choice) and the TPFA
fluxes are compared for both k = 1 and k = 2.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.vem.compute_virtual_ip import compute_virtual_ip
from pymrst.modules.vem.incomp_vem import incomp_vem
from pymrst.params.rock import make_rock
from pymrst.params.wells_and_bc.types import BoundaryCondition
from pymrst.utils.units import DARCY

REF = Path(__file__).with_name("ref_p415_incompvem_tpfa.mat")


def _twisted_grid(nx=6):
    g = cart_grid([nx, nx], [float(nx), float(nx)])
    c = np.asarray(g.nodes.coords, dtype=float)
    x, y, L = c[:, 0].copy(), c[:, 1].copy(), float(nx)
    c[:, 0] += 0.3 * np.sin(np.pi * x / L) * np.sin(np.pi * y / L)
    c[:, 1] += 0.6 * 0.3 * np.sin(2 * np.pi * x / L) * np.sin(np.pi * y / L)
    g.nodes.coords = c
    return compute_geometry(g)


def _check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{('  ' + detail) if detail else ''}")
    return 0 if ok else 1


def _rel(a, b):
    return np.max(np.abs(a - b)), max(np.max(np.abs(b)), 1e-300)


def main() -> int:
    if not REF.exists():
        print(f"missing {REF.name}; run dump_phase415_reference.m in MATLAB first")
        return 1
    ref = loadmat(REF)

    g = _twisted_grid()
    rock = make_rock(g, np.array([[2.0, 0.6, 1.0]]).repeat(g.cells.num, 0) * DARCY, 0.2)
    fc = np.asarray(g.faces.centroids)
    tol = 1e-6
    left = np.flatnonzero(np.abs(fc[:, 0] - fc[:, 0].min()) < tol)
    right = np.flatnonzero(np.abs(fc[:, 0] - fc[:, 0].max()) < tol)
    bc = BoundaryCondition()
    bc.face = np.concatenate([left, right])
    bc.type = ["pressure"] * (left.size + right.size)
    bc.value = np.concatenate([np.ones(left.size), np.zeros(right.size)])

    failed = 0
    print("incompVEM TPFA flux reconstruction vs MRST:")
    for k in (1, 2):
        S = compute_virtual_ip(g, rock, k, trans="tpfa")
        state = incomp_vem(g, S, bc=bc)
        pe, ps = _rel(state["node_pressure"], ref[f"nodePressure{k}"].ravel())
        failed += _check(f"k={k} node pressures match", pe < 1e-9 * ps,
                         f"max|dp| = {pe:.3e} (scale {ps:.2e})")
        fe, fs = _rel(state["flux"], ref[f"flux{k}"].ravel())
        failed += _check(f"k={k} TPFA fluxes match", fe < 1e-9 * fs,
                         f"max|dq| = {fe:.3e} (scale {fs:.2e})")

    print("\nTPFA flux reconstruction matches MRST!" if failed == 0
          else f"\n{failed} checks FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
