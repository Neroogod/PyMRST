"""Phase 409: second-order incompVEM (k=2, 2D) against MRST (incompVEM).

Pins the full PyMRST k=2 `incomp_vem` - assembly over node/face/cell dofs,
source at the cell moment dof, exact-nodal Dirichlet at nodes and face
centroids, and flux reconstruction - against MRST's `incompVEM`. Boundary data
is a quadratic function handle and a balanced source pair drives the cell-dof
RHS path. As in Phase 406, MRST zeroes all boundary fluxes on the func-BC path,
so only interior fluxes are compared.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.vem.compute_virtual_ip import compute_virtual_ip, incomp_vem
from pymrst.params.rock import make_rock
from pymrst.params.wells_and_bc.types import Source
from pymrst.utils.units import DARCY

REF = Path(__file__).with_name("ref_p409_incompvem_k2.mat")


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


def _rel(a, b):
    scale = max(np.max(np.abs(b)), 1e-300)
    return np.max(np.abs(a - b)), scale


def main() -> int:
    if not REF.exists():
        print(f"missing {REF.name}; run dump_phase409_reference.m in MATLAB first")
        return 1
    ref = loadmat(REF)
    pn_ref = np.asarray(ref["nodePressure"], dtype=float).ravel()
    pf_ref = np.asarray(ref["facePressure"], dtype=float).ravel()
    pc_ref = np.asarray(ref["cellPressure"], dtype=float).ravel()
    flux_ref = np.asarray(ref["flux"], dtype=float).ravel()

    g = _twisted_grid()
    rock = make_rock(g, np.array([[2.0, 0.6, 1.0]]).repeat(g.cells.num, 0) * DARCY, 0.2)
    S = compute_virtual_ip(g, rock, 2)

    coords = np.asarray(g.nodes.coords)
    fc = np.asarray(g.faces.centroids)
    gfun = lambda X: X[:, 0] ** 2 - 0.5 * X[:, 0] * X[:, 1] + 2.0 * X[:, 1] ** 2 \
        + 3.0 * X[:, 0] - X[:, 1] + 4.0
    nn = g.nodes.num
    d = {int(nd): float(gfun(coords[nd:nd + 1])[0]) for nd in _boundary_nodes(g)}
    for f in np.flatnonzero(np.any(np.asarray(g.faces.neighbors) < 0, axis=1)):
        d[nn + int(f)] = float(gfun(fc[f:f + 1])[0])
    src = Source(cell=np.array([11, 22]), rate=np.array([1.0, -1.0]))

    state = incomp_vem(g, S, src=src, dirichlet_nodes=d)
    internal = np.flatnonzero(np.all(np.asarray(g.faces.neighbors) >= 0, axis=1))

    failed = 0
    print("incompVEM (k=2, 2D, full tensor, source + nodal Dirichlet) vs MRST:")
    for name, a, b in (("node pressures", state["node_pressure"], pn_ref),
                       ("face pressures", state["face_pressure"], pf_ref),
                       ("cell pressures", state["pressure"], pc_ref)):
        err, scale = _rel(a, b)
        failed += _check(f"{name} match", err < 1e-9 * scale,
                         f"max|d| = {err:.3e} (scale {scale:.2e})")
    err, scale = _rel(state["flux"][internal], flux_ref[internal])
    failed += _check("reconstructed interior fluxes match", err < 1e-9 * scale,
                     f"max|dq| = {err:.3e} (scale {scale:.2e})")

    print("\nSecond-order incompVEM matches MRST!" if failed == 0
          else f"\n{failed} checks FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
