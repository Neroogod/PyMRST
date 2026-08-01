"""Phase 413: 3-D second-order VEM inner product against MRST.

Pins the 3-D k=2 `compute_virtual_ip` against MRST's `computeVirtualIP` (k=2),
comparing per-cell local-stiffness eigenvalues. This exercises the whole 3-D k=2
path: the per-face k=2 2-D VEM projections, the node/edge/face/cell dof moments,
the boundary flux assembly via face moments, and the ip_simple stability.

On a PLANAR-faced grid the method is exact and the match is machine precision -
this is the rigorous pin. On a TWISTED (non-planar-faced) grid MRST and PyMRST
differ by O(twist): MRST integrates each warped face with an edge Gauss-Lobatto
rule using the projected 3-D edge normals, while PyMRST integrates the flattened
polygon exactly. Both are convergent; the twisted case is reported for
characterisation, not asserted to machine precision.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.vem.compute_virtual_ip import compute_virtual_ip
from pymrst.modules.vem.utils.compute_vem_geometry import (
    compute_vem_geometry,
)
from pymrst.params.rock import make_rock
from pymrst.utils.units import DARCY

REF = Path(__file__).with_name("ref_p413_vem_ip_3d_k2.mat")


def _grid(nx=3, amp=0.0):
    g = cart_grid([nx, nx, nx], [float(nx)] * 3)
    if amp:
        c = np.asarray(g.nodes.coords, dtype=float)
        x, y, z, L = c[:, 0].copy(), c[:, 1].copy(), c[:, 2].copy(), float(nx)
        c[:, 0] += amp * np.sin(np.pi * x / L) * np.sin(np.pi * y / L)
        c[:, 1] += 0.75 * amp * np.sin(np.pi * x / L) * np.sin(np.pi * z / L)
        c[:, 2] += 0.5 * amp * np.sin(np.pi * y / L) * np.sin(np.pi * z / L)
        g.nodes.coords = c
    return compute_vem_geometry(compute_geometry(g))


def _check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{('  ' + detail) if detail else ''}")
    return 0 if ok else 1


def _eigs(g, ref_shape):
    perm = np.array([[2.0, 0.5, 0.3, 1.5, 0.2, 1.0]]).repeat(g.cells.num, 0) * DARCY
    S = compute_virtual_ip(g, make_rock(g, perm, 0.2), 2)
    eig = np.zeros(ref_shape)
    for c, Ae in enumerate(S["A"]):
        e = np.sort(np.linalg.eigvalsh((Ae + Ae.T) / 2))
        eig[c, :e.size] = e
    return eig


def main() -> int:
    if not REF.exists():
        print(f"missing {REF.name}; run dump_phase413_reference.m in MATLAB first")
        return 1
    ref = loadmat(REF)
    ep_ref = np.asarray(ref["eigblk_planar"], dtype=float)
    et_ref = np.asarray(ref["eigblk_twisted"], dtype=float)

    failed = 0
    print("computeVirtualIP (k=2, 3-D, full tensor) vs MRST:")
    ep = _eigs(_grid(amp=0.0), ep_ref.shape)
    ps = np.max(np.abs(ep_ref))
    pe = np.max(np.abs(ep - ep_ref))
    failed += _check("planar-faced grid: eigenvalues match (exact)",
                     pe < 1e-9 * ps, f"max|dlambda| = {pe:.3e} (scale {ps:.2e})")

    et = _eigs(_grid(amp=0.2), et_ref.shape)
    ts = np.max(np.abs(et_ref))
    te = np.max(np.abs(et - et_ref))
    print(f"  note twisted grid: max|dlambda| = {te:.3e} (scale {ts:.2e}, "
          f"rel {te / ts:.1e}) - MRST vs PyMRST face-quadrature recipe on "
          f"non-planar faces; both convergent, not asserted")

    print("\n3-D second-order VEM inner product matches MRST (planar-exact)!"
          if failed == 0 else f"\n{failed} checks FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
