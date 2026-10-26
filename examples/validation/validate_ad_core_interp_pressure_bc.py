"""Validate MRST's `getMultiDimInterpolator`, `pressureBCContribADI` and
`bc2ADbc`: the ports in `ad_core/utils/`.

The reference is MRST 2024a (`dump_ad_core_interp_pressure_bc_reference.m`):
a 3x3 table on decreasing axes queried inside and outside the grid with
`linear`, `nearest` and `none` extrapolation, and with AD queries (MRST
extrapolates linearly for those) - values and both Jacobians; and the
surface fluxes through four pressure boundary faces of a 3x2x2 grid with
gravity, with their Jacobians, after `bc2ADbc`.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

HERE = Path(__file__).resolve().parent
RTOL = 1e-13


def report(ok: bool, text: str) -> bool:
    print(f"{'ok  ' if ok else 'FAIL'} {text}")
    return ok


def _close(name, a, b, tol=RTOL):
    a = np.asarray(a.toarray() if hasattr(a, "toarray") else a, dtype=float)
    b = np.asarray(b, dtype=float).reshape(a.shape)
    nan_same = np.array_equal(np.isnan(a), np.isnan(b))
    fin = ~np.isnan(b)
    err = float(np.abs(a[fin] - b[fin]).max()/max(np.abs(b[fin]).max(), 1e-300)) if fin.any() else 0.0
    return report(nan_same and err < tol, f"{name} rel. diff {err:.1e}")


def main() -> int:
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_core.utils.bc_to_ad_bc import bc_to_ad_bc
    from pymrst.modules.ad_core.utils.get_multi_dim_interpolator import get_multi_dim_interpolator
    from pymrst.modules.ad_core.utils.pressure_bc_contrib_adi import pressure_bc_contrib_adi
    from pymrst.params.gravity.gravity import gravity, isolated_gravity
    from pymrst.params.wells_and_bc import add_bc
    from pymrst.utils.adi import ad_exp, init_variables_adi

    ref = loadmat(HERE/"ref_ad_core_interp_pressure_bc.mat", squeeze_me=True, struct_as_record=False)
    x = [np.array([2.0, 1.0, 0.0]), np.array([10.0, 20.0, 40.0])]
    Fx = np.array([[7, 3, 1], [4, 2, 0], [1, 0.5, 0.25]]) + np.array([[0, 1, 2]]*3)
    xq, yq = np.atleast_1d(ref["xq"]), np.atleast_1d(ref["yq"])
    ok = True
    for method in ("linear", "nearest", "none"):
        fn, _, _ = get_multi_dim_interpolator(x, Fx, method)
        ok &= _close(f"getMultiDimInterpolator, {method} extrapolation", fn(xq, yq),
                     getattr(ref["out"], method))
    a, b = init_variables_adi(xq, yq)
    fn, _, _ = get_multi_dim_interpolator(x, Fx, "none")
    v = fn(a, b)
    ok &= _close("AD query: value (linear extrapolation)", v.val, ref["ad"].val)
    ok &= _close("AD query: Jacobian in x", v.jac[0], ref["ad"].ja)
    ok &= _close("AD query: Jacobian in y", v.jac[1], ref["ad"].jb)

    r = ref["pbc"]
    with isolated_gravity():
        gravity("reset", "on")
        G = compute_geometry(cart_grid([3, 2, 2], [30.0, 20.0, 10.0]))
        nc = G.cells.num
        T_all = np.arange(1, G.faces.num + 1)*1e-13
        bc = add_bc(None, np.atleast_1d(r.faces).astype(np.int64) - 1, "pressure",
                    np.array([2e7, 1.5e7, 1.8e7, 2.2e7]))
        bc = bc_to_ad_bc(G, bc)
        ok &= _close("bc2ADbc: cell-to-face map", bc.cell2bcface, r.map)
        p, sw = init_variables_adi(1.9e7 + 1e5*np.arange(1, nc + 1), 0.3 + 0.02*np.arange(1, nc + 1))
        q, cells = pressure_bc_contrib_adi(
            G, {"T_all": T_all}, [p, p], [1000 + 0*p, 800*ad_exp(1e-9*(p - 2e7))],
            [sw**2/1e-3, (1 - sw)**2/5e-3], [1 + 0*p, ad_exp(1e-9*(p - 2e7))], bc)
    ok &= report(np.array_equal(cells + 1, np.atleast_1d(r.cells)), "pressureBCContribADI: cells")
    for i, key in ((0, "q1"), (1, "q2")):
        ok &= _close(f"pressureBCContribADI phase {i + 1}: flux", q[i].val, getattr(r, key))
        ok &= _close(f"pressureBCContribADI phase {i + 1}: d/dp", q[i].jac[0], getattr(r, key + "p"))
        ok &= _close(f"pressureBCContribADI phase {i + 1}: d/ds", q[i].jac[1], getattr(r, key + "s"))
    print("\nThe interpolator and pressure-BC utilities match MRST" if ok
          else "\nThe interpolator and pressure-BC utilities do NOT match MRST")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
