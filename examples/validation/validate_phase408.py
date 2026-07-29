"""Phase 408: second-order (k=2) VEM inner product against MRST.

Pins the k=2 `compute_virtual_ip` against MRST's `computeVirtualIP` (k=2) on a
twisted grid with a full permeability tensor, comparing per-cell local-stiffness
eigenvalues (an ordering-independent fingerprint that pins the quadratic
consistency term and the ip_simple stability).
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.vem.compute_virtual_ip import compute_virtual_ip
from pymrst.params.rock import make_rock
from pymrst.utils.units import DARCY

REF = Path(__file__).with_name("ref_p408_vem_ip_k2.mat")


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
        print(f"missing {REF.name}; run dump_phase408_reference.m in MATLAB first")
        return 1
    ref = loadmat(REF)
    eig_ref = np.asarray(ref["eigblk"], dtype=float)

    g = _twisted_grid()
    rock = make_rock(g, np.array([[2.0, 0.6, 1.0]]).repeat(g.cells.num, 0) * DARCY, 0.2)
    S = compute_virtual_ip(g, rock, 2)

    eig = np.zeros_like(eig_ref)
    for c, Ae in enumerate(S["A"]):
        e = np.sort(np.linalg.eigvalsh((Ae + Ae.T) / 2))
        eig[c, :e.size] = e

    failed = 0
    print("computeVirtualIP (k=2, full tensor, twisted grid) vs MRST:")
    failed += _check(f"same block shape ({eig.shape[1]} dofs/cell)",
                     eig.shape == eig_ref.shape)
    scale = np.max(np.abs(eig_ref))
    err = np.max(np.abs(eig - eig_ref))
    failed += _check("per-cell local-stiffness eigenvalues match",
                     err < 1e-9 * scale, f"max|dlambda| = {err:.3e} (scale {scale:.2e})")

    print("\nSecond-order VEM inner product matches MRST!" if failed == 0
          else f"\n{failed} checks FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
