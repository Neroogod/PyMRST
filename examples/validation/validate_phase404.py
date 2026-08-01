"""Phase 404: VEM polyhedron integration against MRST (vem/utils).

Pins `polyhedron_int` / the tetrahedron quadrature against MRST's `polyhedronInt`
on a 2x2x2 grid: the per-cell integrals of the 3-D monomial basis
[1, x, y, z, x^2, y^2, z^2, xy] with a precision-3 rule. Exact for these
polynomials, so triangulation-independent and machine-precision.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.vem.utils import polyhedron_int

REF = Path(__file__).with_name("ref_p404_polyhedronint.mat")


def _monomials(P: np.ndarray) -> np.ndarray:
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    return np.column_stack([np.ones_like(x), x, y, z,
                            x ** 2, y ** 2, z ** 2, x * y])


def _check(name: str, ok: bool, detail: str = "") -> int:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{('  ' + detail) if detail else ''}")
    return 0 if ok else 1


def main() -> int:
    if not REF.exists():
        print(f"missing {REF.name}; run dump_phase404_reference.m in MATLAB first")
        return 1
    ref = loadmat(REF)

    g = compute_geometry(cart_grid([2, 2, 2], [2.0, 2.0, 2.0]))
    I = polyhedron_int(g, range(g.cells.num), _monomials, 3)

    failed = 0
    print("polyhedronInt (3-D monomial basis, precision 3) vs MRST:")
    failed += _check("same shape", I.shape == ref["I"].shape,
                     f"{I.shape} vs {ref['I'].shape}")
    err = np.max(np.abs(I - ref["I"]))
    failed += _check("per-cell monomial integrals match", err < 1e-10,
                     f"max|dI| = {err:.3e}")

    print("\nVEM polyhedron integration matches MRST!" if failed == 0
          else f"\n{failed} checks FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
