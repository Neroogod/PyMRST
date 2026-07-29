"""Phase 412: 3-D first-order incompVEM against MRST (incompVEM).

Pins the full PyMRST 3-D k=1 `incomp_vem` - assembly, source distribution,
exact-nodal Dirichlet, and flux reconstruction - against MRST's `incompVEM` on a
twisted 3-D grid with a full permeability tensor. Both solve the identical
discrete system (exact-nodal Dirichlet + balanced source), so the comparison is
number-for-number, independent of the 3-D scheme's own approximate linear
reproduction. As in Phase 406, MRST zeroes all boundary fluxes on the func-BC
path, so only interior fluxes are compared.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.vem.compute_virtual_ip import compute_virtual_ip, incomp_vem
from pymrst.modules.vem.utils.vem_geometry import compute_vem_geometry
from pymrst.params.rock import make_rock
from pymrst.params.wells_and_bc.types import Source
from pymrst.utils.units import DARCY

REF = Path(__file__).with_name("ref_p412_incompvem_3d.mat")


def _twisted_grid(nx=3, amp=0.2):
    g = cart_grid([nx, nx, nx], [float(nx)] * 3)
    c = np.asarray(g.nodes.coords, dtype=float)
    x, y, z, L = c[:, 0].copy(), c[:, 1].copy(), c[:, 2].copy(), float(nx)
    c[:, 0] += amp * np.sin(np.pi * x / L) * np.sin(np.pi * y / L)
    c[:, 1] += 0.75 * amp * np.sin(np.pi * x / L) * np.sin(np.pi * z / L)
    c[:, 2] += 0.5 * amp * np.sin(np.pi * y / L) * np.sin(np.pi * z / L)
    g.nodes.coords = c
    return compute_vem_geometry(compute_geometry(g))


def _boundary_nodes(g):
    node_pos = np.asarray(g.faces.nodePos)
    fnodes = np.asarray(g.faces.nodes)
    bf = np.flatnonzero(np.any(np.asarray(g.faces.neighbors) < 0, axis=1))
    return np.unique(np.concatenate(
        [fnodes[node_pos[f]:node_pos[f + 1]] for f in bf]))


def _check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{('  ' + detail) if detail else ''}")
    return 0 if ok else 1


def main() -> int:
    if not REF.exists():
        print(f"missing {REF.name}; run dump_phase412_reference.m in MATLAB first")
        return 1
    ref = loadmat(REF)
    p_ref = np.asarray(ref["nodePressure"], dtype=float).ravel()
    flux_ref = np.asarray(ref["flux"], dtype=float).ravel()

    g = _twisted_grid()
    perm = np.array([[2.0, 0.5, 0.3, 1.5, 0.2, 1.0]]).repeat(g.cells.num, 0) * DARCY
    rock = make_rock(g, perm, 0.2)
    S = compute_virtual_ip(g, rock, 1)

    coords = np.asarray(g.nodes.coords)
    gfun = lambda X: 1.5 * X[:, 0] - 2.0 * X[:, 1] + 0.8 * X[:, 2] + 3.0
    d = {int(nd): float(gfun(coords[nd:nd + 1])[0]) for nd in _boundary_nodes(g)}
    src = Source(cell=np.array([4, 21]), rate=np.array([1.0, -1.0]))

    state = incomp_vem(g, S, src=src, dirichlet_nodes=d)
    internal = np.flatnonzero(np.all(np.asarray(g.faces.neighbors) >= 0, axis=1))

    failed = 0
    print("incompVEM (k=1, 3-D, full tensor, source + nodal Dirichlet) vs MRST:")
    pscale = np.max(np.abs(p_ref))
    perr = np.max(np.abs(state["node_pressure"] - p_ref))
    failed += _check("node pressures match", perr < 1e-9 * pscale,
                     f"max|dp| = {perr:.3e} (scale {pscale:.2e})")
    fscale = np.max(np.abs(flux_ref[internal]))
    ferr = np.max(np.abs(state["flux"][internal] - flux_ref[internal]))
    failed += _check("reconstructed interior fluxes match", ferr < 1e-9 * fscale,
                     f"max|dq| = {ferr:.3e} (scale {fscale:.2e})")

    print("\n3-D first-order incompVEM matches MRST!" if failed == 0
          else f"\n{failed} checks FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
