"""Phase 79: material AD derivatives for VEM ``Ax_derivs``."""

from types import SimpleNamespace

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.vemmech.ad_version import SparseMultiArray, vem_assemble_ad, vem_lin_elast_ad
from pymrst.modules.vemmech.vem import create_augmented_grid
from pymrst.utils.adi import init_variables_adi


def compare(name, py, ref, rtol=1e-5, atol=1e-8) -> int:
    py = np.asarray(py, dtype=float).ravel()
    ref = np.asarray(ref, dtype=float).ravel()
    scale = max(float(np.max(np.abs(ref))) if ref.size else 0.0, 1.0)
    diff = 0.0 if py.size == ref.size == 0 else float(np.max(np.abs(py - ref)))
    ok = py.shape == ref.shape and diff <= atol + rtol * scale
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: max abs diff = {diff:.3e} "
          f"(rel {diff / scale:.2e}, rtol {rtol:.0e})")
    return 0 if ok else 1


def main() -> int:
    g = create_augmented_grid(
        compute_geometry(cart_grid([1, 1, 1], [1.0, 1.0, 1.0]))
    )
    E0 = np.array([1.13e9])
    nu0 = np.array([0.24])
    E, nu = init_variables_adi(E0, nu0)

    bc = SimpleNamespace(
        disp_bc=SparseMultiArray(
            [0.0, 0.0, 0.0, 0.015],
            [[1, 1], [1, 2], [1, 3], [8, 3]],
            ["n", "d"],
            keep_zeros=True,
        ),
        force_bc=None,
    )
    load = SparseMultiArray(np.zeros((g.cells.nodes.size, g.griddim)),
                            ["cn", "d"], keep_zeros=True)
    uu, extra = vem_lin_elast_ad(g, E, nu, bc, load)
    full_u = uu.reshape(-1)
    free = ~extra.disc["isdirdofs"]

    hE = 1e-5 * E0[0]
    hnu = 1e-6
    Sp, _ = vem_assemble_ad(g, E0 + hE, nu0)
    Sm, _ = vem_assemble_ad(g, E0 - hE, nu0)
    fd_E = ((Sp - Sm) @ full_u)[free] / (2.0 * hE)

    Sp, _ = vem_assemble_ad(g, E0, nu0 + hnu)
    Sm, _ = vem_assemble_ad(g, E0, nu0 - hnu)
    fd_nu = ((Sp - Sm) @ full_u)[free] / (2.0 * hnu)

    got = extra.Ax_derivs.toarray()
    failures = 0
    print("Phase 79: VEM material AD Ax_derivs")
    failures += compare("shape rows", [got.shape[0]], [np.count_nonzero(free)],
                        rtol=0, atol=0)
    failures += compare("shape cols", [got.shape[1]], [2], rtol=0, atol=0)
    failures += compare("dAx/dE", got[:, 0], fd_E, rtol=1e-5, atol=1e-9)
    failures += compare("dAx/dnu", got[:, 1], fd_nu, rtol=1e-5, atol=1e-3)

    if failures:
        print(f"\nVEM material AD derivative validation failed ({failures} checks).")
    else:
        print("\nPyMRST VEM Ax_derivs match central finite differences.")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
