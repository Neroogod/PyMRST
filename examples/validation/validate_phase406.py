"""Phase 406: first-order incompVEM (k=1, 2D) against MRST (incompVEM).

Pins the full PyMRST `incomp_vem` solver - global assembly, source distribution,
exact-nodal Dirichlet, and direct `(p' A r)/|E|` flux reconstruction - against
MRST's `incompVEM` on a twisted grid with a full permeability tensor. Boundary
data is a bilinear function handle (not reproduced by k=1, so the interior solve
is a genuine test) and a balanced source pair drives the RHS path.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.vem.compute_virtual_ip import compute_virtual_ip, incomp_vem
from pymrst.params.rock import make_rock
from pymrst.params.wells_and_bc.types import Source
from pymrst.utils.units import DARCY

REF = Path(__file__).with_name("ref_p406_incompvem.mat")


def _twisted_grid(nx=6):
    g = cart_grid([nx, nx], [float(nx), float(nx)])
    c = np.asarray(g.nodes.coords, dtype=float)
    x, y, L = c[:, 0].copy(), c[:, 1].copy(), float(nx)
    c[:, 0] += 0.3 * np.sin(np.pi * x / L) * np.sin(np.pi * y / L)
    c[:, 1] += 0.6 * 0.3 * np.sin(2 * np.pi * x / L) * np.sin(np.pi * y / L)
    g.nodes.coords = c
    return compute_geometry(g)


def _boundary_nodes(g):
    node_pos = np.asarray(g.faces.nodePos)
    fnodes = np.asarray(g.faces.nodes)
    bf = np.flatnonzero(np.any(np.asarray(g.faces.neighbors) < 0, axis=1))
    return np.unique(np.concatenate(
        [fnodes[node_pos[f]:node_pos[f + 1]] for f in bf]))


def _check(name: str, ok: bool, detail: str = "") -> int:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{('  ' + detail) if detail else ''}")
    return 0 if ok else 1


def main() -> int:
    if not REF.exists():
        print(f"missing {REF.name}; run dump_phase406_reference.m in MATLAB first")
        return 1
    ref = loadmat(REF)
    p_ref = np.asarray(ref["nodePressure"], dtype=float).ravel()
    flux_raw = ref["flux"]
    if hasattr(flux_raw, "toarray"):
        flux_raw = flux_raw.toarray()
    flux_ref = np.asarray(flux_raw, dtype=float).ravel()

    g = _twisted_grid()
    rock = make_rock(g, np.array([[2.0, 0.6, 1.0]]).repeat(g.cells.num, 0) * DARCY, 0.2)
    S = compute_virtual_ip(g, rock, 1)

    coords = np.asarray(g.nodes.coords)
    gfun = lambda X: 2.0 + 3.0 * X[:, 0] - 1.5 * X[:, 1] + 0.5 * X[:, 0] * X[:, 1]
    nodes = {int(nd): float(gfun(coords[nd:nd + 1])[0]) for nd in _boundary_nodes(g)}
    src = Source(cell=np.array([11, 22]), rate=np.array([1.0, -1.0]))

    state = incomp_vem(g, S, src=src, dirichlet_nodes=nodes)
    p = state["node_pressure"]
    flux = state["flux"]
    internal = np.flatnonzero(np.all(np.asarray(g.faces.neighbors) >= 0, axis=1))

    failed = 0
    print("incompVEM (k=1, 2D, full tensor, source + nodal Dirichlet) vs MRST:")
    pscale = np.max(np.abs(p_ref))
    perr = np.max(np.abs(p - p_ref))
    failed += _check("node pressures match", perr < 1e-9 * pscale,
                     f"max|dp| = {perr:.3e} (scale {pscale:.2e})")
    fscale = np.max(np.abs(flux_ref[internal]))
    ferr = np.max(np.abs(flux[internal] - flux_ref[internal]))
    failed += _check("reconstructed interior fluxes match", ferr < 1e-9 * fscale,
                     f"max|dq| = {ferr:.3e} (scale {fscale:.2e})")

    # MRST quirk: with the addBCVEM function-handle BC path, incompVEM cannot
    # tell Dirichlet from Neumann faces in the func struct, so it overrides ALL
    # boundary fluxes to zero (the `~isfield(bc,'func')` guard skips the
    # Dirichlet-unflagging). PyMRST keeps the physically correct reconstructed
    # boundary flux, so those faces are intentionally not compared.
    bf = np.flatnonzero(np.any(np.asarray(g.faces.neighbors) < 0, axis=1))
    print(f"  note MRST zeroes all {bf.size} boundary fluxes on the func-BC "
          f"path; PyMRST keeps the reconstructed value (max |q_bdry| "
          f"{np.max(np.abs(flux[bf])):.3e})")

    print("\nFirst-order incompVEM matches MRST!" if failed == 0
          else f"\n{failed} checks FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
