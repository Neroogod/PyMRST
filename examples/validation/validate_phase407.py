"""Phase 407: incompVEM standard-BC + Neumann (k=1, 2D) against MRST.

Complements Phase 406 by pinning the standard MRST boundary-condition path:
per-face-constant Dirichlet on left/right, a prescribed Neumann flux on the
bottom, natural no-flow on top. This validates the Neumann RHS assembly and the
boundary-flux handling that the function-handle path bypasses.
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

REF = Path(__file__).with_name("ref_p407_incompvem_bc.mat")


def _twisted_grid(nx=6):
    g = cart_grid([nx, nx], [float(nx), float(nx)])
    c = np.asarray(g.nodes.coords, dtype=float)
    x, y, L = c[:, 0].copy(), c[:, 1].copy(), float(nx)
    c[:, 0] += 0.3 * np.sin(np.pi * x / L) * np.sin(np.pi * y / L)
    c[:, 1] += 0.6 * 0.3 * np.sin(2 * np.pi * x / L) * np.sin(np.pi * y / L)
    g.nodes.coords = c
    return compute_geometry(g)


def _check(name: str, ok: bool, detail: str = "") -> int:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{('  ' + detail) if detail else ''}")
    return 0 if ok else 1


def main() -> int:
    if not REF.exists():
        print(f"missing {REF.name}; run dump_phase407_reference.m in MATLAB first")
        return 1
    ref = loadmat(REF)
    p_ref = np.asarray(ref["nodePressure"], dtype=float).ravel()
    flux_ref = np.asarray(ref["flux"], dtype=float).ravel()

    g = _twisted_grid()
    rock = make_rock(g, np.array([[2.0, 0.6, 1.0]]).repeat(g.cells.num, 0) * DARCY, 0.2)
    S = compute_virtual_ip(g, rock, 1)

    fc = np.asarray(g.faces.centroids)
    tol = 1e-6
    left = np.flatnonzero(np.abs(fc[:, 0] - fc[:, 0].min()) < tol)
    right = np.flatnonzero(np.abs(fc[:, 0] - fc[:, 0].max()) < tol)
    bottom = np.flatnonzero(np.abs(fc[:, 1] - fc[:, 1].min()) < tol)

    bc = BoundaryCondition()
    bc.face = np.concatenate([left, right, bottom])
    bc.type = ["pressure"] * left.size + ["pressure"] * right.size \
        + ["flux"] * bottom.size
    bc.value = np.concatenate([np.ones(left.size), np.zeros(right.size),
                               np.full(bottom.size, 0.3 * DARCY)])

    state = incomp_vem(g, S, bc=bc)
    p = state["node_pressure"]
    flux = state["flux"]

    failed = 0
    print("incompVEM (k=1, 2D, standard BC: Dirichlet + Neumann) vs MRST:")
    pscale = np.max(np.abs(p_ref))
    perr = np.max(np.abs(p - p_ref))
    failed += _check("node pressures match", perr < 1e-9 * pscale,
                     f"max|dp| = {perr:.3e} (scale {pscale:.2e})")
    fscale = np.max(np.abs(flux_ref))
    ferr = np.max(np.abs(flux - flux_ref))
    failed += _check("all face fluxes match (incl. boundary)", ferr < 1e-9 * fscale,
                     f"max|dq| = {ferr:.3e} (scale {fscale:.2e})")

    print("\nStandard-BC incompVEM matches MRST!" if failed == 0
          else f"\n{failed} checks FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
