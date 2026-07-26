"""Phase 403: VEM polygon integration against MRST (vem/utils).

Pins `polygon_int` / the triangle quadrature against MRST's `polygonInt` on a 3x3
grid: the per-cell integrals of the monomial basis [1, x, y, x^2, xy, y^2] with a
precision-3 rule. Both integrate the same polynomials exactly, so the comparison
is triangulation-independent and machine-precision.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.vem.utils.quadrature import polygon_int

REF = Path(__file__).with_name("ref_p403_polyint.mat")


def _monomials(P: np.ndarray) -> np.ndarray:
    x, y = P[:, 0], P[:, 1]
    return np.column_stack([np.ones_like(x), x, y, x ** 2, x * y, y ** 2])


def _check(name: str, ok: bool, detail: str = "") -> int:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{('  ' + detail) if detail else ''}")
    return 0 if ok else 1


def main() -> int:
    if not REF.exists():
        print(f"missing {REF.name}; run dump_phase403_reference.m in MATLAB first")
        return 1
    ref = loadmat(REF)

    g = compute_geometry(cart_grid([3, 3], [3.0, 3.0]))
    I = polygon_int(g, range(g.cells.num), _monomials, 3)

    failed = 0
    print("polygonInt (monomial basis, precision 3) vs MRST:")
    failed += _check("same shape", I.shape == ref["I"].shape,
                     f"{I.shape} vs {ref['I'].shape}")
    err = np.max(np.abs(I - ref["I"]))
    failed += _check("per-cell monomial integrals match", err < 1e-10,
                     f"max|dI| = {err:.3e}")

    print("\nVEM polygon integration matches MRST!" if failed == 0
          else f"\n{failed} checks FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
