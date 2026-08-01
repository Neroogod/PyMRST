"""Phase 423: VEM MPFA flux reconstruction (invariant, MRST path is broken).

MRST's own `incompVEM` MPFA reconstruction is broken in 2024a: the option
`computeVirtualIP(G, rock, k, 'trans', 'mpfa')` stores whatever
`computeMultiPointTrans` returns, but that no longer produces the `S.T.T`
transmissibility struct `incompVEM`'s `computeFlux` expects (the old-style
struct is commented out in `computeMultiPointTransLegacy`), so `incompVEM`
errors with "Unrecognized field name T". A direct MRST comparison is therefore
impossible.

PyMRST's `trans='mpfa'` option is validated instead by the defining MPFA
invariant: on a twisted, non-K-orthogonal grid with a full permeability tensor,
the MPFA reconstruction reproduces a linear pressure's flux exactly, whereas
TPFA does not. (The underlying `compute_multipoint_trans` operator is itself
pinned to MRST elsewhere in the mpfa module.)
"""

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.vem.compute_virtual_ip import compute_virtual_ip
from pymrst.modules.vem.incomp_vem import incomp_vem
from pymrst.params.rock import make_rock


def _twisted_grid(nx=5):
    g = cart_grid([nx, nx], [float(nx), float(nx)])
    c = np.asarray(g.nodes.coords, dtype=float)
    x, y, L = c[:, 0].copy(), c[:, 1].copy(), float(nx)
    c[:, 0] += 0.25 * np.sin(np.pi * x / L) * np.sin(np.pi * y / L)
    c[:, 1] += 0.15 * np.sin(2 * np.pi * x / L) * np.sin(np.pi * y / L)
    g.nodes.coords = c
    return compute_geometry(g)


def _boundary_nodes(g):
    npos = np.asarray(g.faces.nodePos)
    fn = np.asarray(g.faces.nodes)
    bf = np.flatnonzero(np.any(np.asarray(g.faces.neighbors) < 0, axis=1))
    return np.unique(np.concatenate([fn[npos[f]:npos[f + 1]] for f in bf]))


def _check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{('  ' + detail) if detail else ''}")
    return 0 if ok else 1


def main() -> int:
    g = _twisted_grid()
    Kmat = np.array([[2.0, 0.6], [0.6, 1.0]])
    rock = make_rock(g, np.array([[2.0, 0.6, 1.0]]).repeat(g.cells.num, 0), 0.2)
    coords = np.asarray(g.nodes.coords)
    fc = np.asarray(g.faces.centroids)
    grad = np.array([1.5, -0.7])
    lin = lambda P: P[:, :2] @ grad + 2.0
    internal = np.flatnonzero(np.all(np.asarray(g.faces.neighbors) >= 0, axis=1))
    expected = -(np.asarray(g.faces.normals)[internal, :2] @ (Kmat @ grad))
    scale = np.max(np.abs(expected))

    failed = 0
    print("VEM MPFA flux reconstruction (twisted grid, MPFA invariant):")
    print("  note MRST's incompVEM MPFA path errors in 2024a (S.T.T removed); "
          "validated by the MPFA linear-exactness invariant instead")
    for k in (1, 2):
        Sm = compute_virtual_ip(g, rock, k, trans="mpfa")
        St = compute_virtual_ip(g, rock, k, trans="tpfa")
        d = {int(nd): float(lin(coords[nd:nd + 1])[0]) for nd in _boundary_nodes(g)}
        if k == 2:
            bf = np.flatnonzero(np.any(np.asarray(g.faces.neighbors) < 0, axis=1))
            for f in bf:
                d[g.nodes.num + int(f)] = float(lin(fc[f:f + 1])[0])
        qm = incomp_vem(g, Sm, dirichlet_nodes=d)["flux"][internal]
        qt = incomp_vem(g, St, dirichlet_nodes=d)["flux"][internal]
        me = np.max(np.abs(qm - expected))
        te = np.max(np.abs(qt - expected))
        failed += _check(f"k={k}: MPFA exact for linear field, TPFA is not",
                         me < 1e-9 * scale and te > 0.1 * scale,
                         f"MPFA err {me:.2e}, TPFA err {te:.2e} (scale {scale:.2e})")

    print("\nMPFA flux reconstruction validated (invariant)!" if failed == 0
          else f"\n{failed} checks FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
