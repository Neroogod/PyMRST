"""Phase 411: 3-D first-order VEM inner product against MRST.

Pins the 3-D k=1 `compute_virtual_ip` against MRST's `computeVirtualIP` (k=1) on
a twisted 3-D grid with a full permeability tensor, comparing per-cell
local-stiffness eigenvalues. This exercises the whole 3-D path: the augmented
edge geometry, the per-face 2-D VEM P0 projection (PiNFstar), the vertex-average
constant row, and the ip_simple stability.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.vem.compute_virtual_ip import compute_virtual_ip
from pymrst.modules.vem.utils.vem_geometry import compute_vem_geometry
from pymrst.params.rock import make_rock
from pymrst.utils.units import DARCY

REF = Path(__file__).with_name("ref_p411_vem_ip_3d.mat")


def _twisted_grid(nx=3, amp=0.2):
    g = cart_grid([nx, nx, nx], [float(nx)] * 3)
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


def main() -> int:
    if not REF.exists():
        print(f"missing {REF.name}; run dump_phase411_reference.m in MATLAB first")
        return 1
    ref = loadmat(REF)
    eig_ref = np.asarray(ref["eigblk"], dtype=float)

    g = _twisted_grid()
    perm = np.array([[2.0, 0.5, 0.3, 1.5, 0.2, 1.0]]).repeat(g.cells.num, 0) * DARCY
    rock = make_rock(g, perm, 0.2)
    S = compute_virtual_ip(g, rock, 1)

    eig = np.zeros_like(eig_ref)
    for c, Ae in enumerate(S["A"]):
        e = np.sort(np.linalg.eigvalsh((Ae + Ae.T) / 2))
        eig[c, :e.size] = e

    failed = 0
    print("computeVirtualIP (k=1, 3-D, full tensor, twisted grid) vs MRST:")
    failed += _check(f"same block shape ({eig.shape[1]} nodes/cell)",
                     eig.shape == eig_ref.shape)
    scale = np.max(np.abs(eig_ref))
    err = np.max(np.abs(eig - eig_ref))
    failed += _check("per-cell local-stiffness eigenvalues match",
                     err < 1e-9 * scale, f"max|dlambda| = {err:.3e} (scale {scale:.2e})")

    print("\n3-D first-order VEM inner product matches MRST!" if failed == 0
          else f"\n{failed} checks FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
