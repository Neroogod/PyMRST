"""Phase 85: calibration boundary-condition derivative columns."""

from types import SimpleNamespace

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.vemmech.ad_version import vem_lin_elast_ad, vem_reduced_system_residual
from pymrst.modules.vemmech.calibration import identify_boundary_elements, set_bcfun, set_material_fun
from pymrst.modules.vemmech.vem import create_augmented_grid


def compare(name, py, ref, rtol=1e-6, atol=1e-9) -> int:
    py = np.asarray(py, dtype=float).ravel()
    ref = np.asarray(ref, dtype=float).ravel()
    scale = max(float(np.max(np.abs(ref))) if ref.size else 0.0, 1.0)
    diff = 0.0 if py.size == ref.size == 0 else float(np.max(np.abs(py - ref)))
    ok = py.shape == ref.shape and diff <= atol + rtol * scale
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: max abs diff = {diff:.3e} "
          f"(rel {diff / scale:.2e}, rtol {rtol:.0e})")
    return 0 if ok else 1


def _disp_vec(g, bc):
    mat = bc.disp_bc.as_matrix([["n"], ["d"]], True,
                               [g.nodes.num, g.griddim]).toarray()
    return np.nan_to_num(mat.reshape(-1), nan=0.0)


def main() -> int:
    g = create_augmented_grid(
        compute_geometry(cart_grid([1, 1, 1], [1.0, 1.0, 1.0]))
    )
    binfo = identify_boundary_elements(g)
    bcfun, num_bc = set_bcfun(
        g,
        binfo,
        np.array([
            [0.12, 0.03],
            [0.02, 0.00],
            [-0.03, 0.04],
            [0.01, 0.02],
            [-0.02, 0.01],
        ]),
    )
    u = np.linspace(0.2, 0.8, num_bc)
    bc = bcfun(u)
    fd_bc = np.zeros((g.griddim * g.nodes.num, num_bc))
    for i in range(num_bc):
        h = 1e-6 * max(abs(u[i]), 1.0)
        up = u.copy()
        um = u.copy()
        up[i] += h
        um[i] -= h
        fd_bc[:, i] = (_disp_vec(g, bcfun(up)) - _disp_vec(g, bcfun(um))) / (2.0 * h)

    props = SimpleNamespace(
        young=np.full(g.cells.num, 1.05e9),
        poisson=np.full(g.cells.num, 0.23),
        rho=np.full(g.cells.num, 1000.0),
    )
    mfuns, _ = set_material_fun(g, props, None)
    uu, extra = vem_lin_elast_ad(
        g, mfuns.efun(u), mfuns.nufun(u), bc, mfuns.loadfun(u)
    )
    full = uu.reshape(-1)
    free = ~extra.disc["isdirdofs"]
    analytic_res = (extra.S @ bc.disp_bc_jac)[free, :].toarray()
    fd_res = np.zeros_like(analytic_res)
    for i in range(num_bc):
        h = 1e-6 * max(abs(u[i]), 1.0)
        up = u.copy()
        um = u.copy()
        up[i] += h
        um[i] -= h
        fd_res[:, i] = (
            vem_reduced_system_residual(
                g, mfuns.efun(up), mfuns.nufun(up), bcfun(up),
                mfuns.loadfun(up), full, extra=extra,
            )[0]
            - vem_reduced_system_residual(
                g, mfuns.efun(um), mfuns.nufun(um), bcfun(um),
                mfuns.loadfun(um), full, extra=extra,
            )[0]
        ) / (2.0 * h)

    failures = 0
    print("Phase 85: calibration BC derivative columns")
    failures += compare("disp_bc_jac", bc.disp_bc_jac.toarray(), fd_bc,
                        rtol=1e-6, atol=1e-10)
    failures += compare("reduced residual BC columns", analytic_res, fd_res,
                        rtol=1e-6, atol=1e-5)
    if failures:
        print(f"\nBC derivative validation failed ({failures} checks).")
    else:
        print("\nPyMRST calibration BC derivatives match central finite differences.")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
