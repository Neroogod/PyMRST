"""Phase 414: 3-D second-order incompVEM (planar) against MRST (incompVEM).

Pins the full PyMRST 3-D k=2 `incomp_vem` - assembly over node/edge/face/cell
dofs, source at the cell moment dof, exact-nodal Dirichlet at nodes, edges and
face moments, and flux reconstruction - against MRST's `incompVEM` on a
planar-faced grid (where the k=2 3-D method is exact). Both solve the identical
discrete system, so the match is machine precision. As in Phase 406/412, MRST
zeroes all boundary fluxes on the func-BC path, so only interior fluxes compare.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.vem.compute_virtual_ip import compute_virtual_ip
from pymrst.modules.vem.incomp_vem import incomp_vem
from pymrst.modules.vem.utils import polygon_int_3d
from pymrst.modules.vem.utils.compute_vem_geometry import (
    compute_vem_geometry,
)
from pymrst.params.rock import make_rock
from pymrst.params.wells_and_bc.types import Source
from pymrst.utils.units import DARCY

REF = Path(__file__).with_name("ref_p414_incompvem_3d_k2.mat")


def _check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{('  ' + detail) if detail else ''}")
    return 0 if ok else 1


def _rel(a, b):
    return np.max(np.abs(a - b)), max(np.max(np.abs(b)), 1e-300)


def main() -> int:
    if not REF.exists():
        print(f"missing {REF.name}; run dump_phase414_reference.m in MATLAB first")
        return 1
    ref = loadmat(REF)
    pn = np.asarray(ref["nodePressure"], float).ravel()
    pe = np.asarray(ref["edgePressure"], float).ravel()
    pf = np.asarray(ref["facePressure"], float).ravel()
    pc = np.asarray(ref["cellPressure"], float).ravel()
    flux_ref = np.asarray(ref["flux"], float).ravel()

    g = compute_vem_geometry(compute_geometry(cart_grid([3, 3, 3], [3.0, 3.0, 3.0])))
    perm = np.array([[2.0, 0.5, 0.3, 1.5, 0.2, 1.0]]).repeat(g.cells.num, 0) * DARCY
    S = compute_virtual_ip(g, make_rock(g, perm, 0.2), 2)

    coords = np.asarray(g.nodes.coords)
    ecent = np.asarray(g.edges.centroids)
    areas = np.asarray(g.faces.areas)
    gfun = lambda X: X[:, 0] ** 2 - 0.5 * X[:, 0] * X[:, 1] + 2.0 * X[:, 2] ** 2 \
        + 3.0 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2] + 4.0
    nn, ne = g.nodes.num, g.edges.num
    npos = np.asarray(g.faces.nodePos)
    fnodes = np.asarray(g.faces.nodes)
    epos = np.asarray(g.faces.edgePos)
    fedges = np.asarray(g.faces.edges)
    bf = np.flatnonzero(np.any(np.asarray(g.faces.neighbors) < 0, axis=1))
    d = {}
    for f in bf:
        for nd in fnodes[npos[f]:npos[f + 1]]:
            d[int(nd)] = float(gfun(coords[nd:nd + 1])[0])
        for e in fedges[epos[f]:epos[f + 1]]:
            d[nn + int(e)] = float(gfun(ecent[e:e + 1])[0])
        d[nn + ne + int(f)] = polygon_int_3d(g, [f], lambda P: gfun(P)[:, None], 3)[0, 0] / areas[f]
    src = Source(cell=np.array([4, 21]), rate=np.array([1.0, -1.0]))

    state = incomp_vem(g, S, src=src, dirichlet_nodes=d)
    internal = np.flatnonzero(np.all(np.asarray(g.faces.neighbors) >= 0, axis=1))

    failed = 0
    print("incompVEM (k=2, 3-D planar, full tensor, source + nodal Dirichlet) vs MRST:")
    for name, a, b in (("node pressures", state["node_pressure"], pn),
                       ("edge pressures", state["edge_pressure"], pe),
                       ("face pressures", state["face_pressure"], pf),
                       ("cell pressures", state["pressure"], pc)):
        err, scale = _rel(a, b)
        failed += _check(f"{name} match", err < 1e-9 * scale,
                         f"max|d| = {err:.3e} (scale {scale:.2e})")
    err, scale = _rel(state["flux"][internal], flux_ref[internal])
    failed += _check("reconstructed interior fluxes match", err < 1e-9 * scale,
                     f"max|dq| = {err:.3e} (scale {scale:.2e})")

    print("\n3-D second-order incompVEM matches MRST!" if failed == 0
          else f"\n{failed} checks FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
