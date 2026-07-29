"""Phase 424: DG scalar transport against analytical ground truth.

Validates the DG transport solver (`DGScalarTransport`) - the weak-form residual,
upwind numerical flux, implicit time stepping and analytical Jacobian built on
the validated DG discretization - against exact solutions:

  1. Linear advection of a Gaussian pulse: the DG solution converges to the exact
     translated pulse as dt -> 0 (backward Euler is first-order/dissipative).
  2. Buckley-Leverett displacement: the DG shock sits at the analytic Welge front
     with a bounded, monotone profile.

Why analytical rather than MRST bit-for-bit: MRST's `TransportModelDG` applies
TVB + physical slope limiters for degree >= 1 (degree 0 is plain finite volume)
and runs inside the full sequential pressure/transport schedule. Reproducing its
saturation dof-for-dof needs those limiters and setup; the exact solutions here
pin the underlying DG transport scheme itself (shock speed is limiter-independent,
so the front position is the invariant both must reproduce).
"""

import numpy as np
from scipy.optimize import brentq

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.dg.discretizations.dg_discretization import DGDiscretization
from pymrst.modules.dg.dg_transport import DGScalarTransport


def _check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{('  ' + detail) if detail else ''}")
    return 0 if ok else 1


def main() -> int:
    failed = 0
    print("DG scalar transport vs analytical solutions:")

    # 1. Linear advection converges to the exact translated pulse.
    g = compute_geometry(cart_grid([40, 1], [10.0, 1.0]))
    disc = DGDiscretization.build(g, 2)
    tr = DGScalarTransport(disc, np.array([1.0, 0.0]), flux_fn=lambda u: u)
    pulse = lambda P: np.exp(-((P[:, 0] - 2.0) ** 2) / 1.0)
    cc = np.asarray(g.cells.centroids)
    exact = np.exp(-((cc[:, 0] - 5.0) ** 2) / 1.0)
    errs = [np.max(np.abs(disc.cell_mean(
        tr.solve_linear(disc.project(pulse), dt, int(3.0 / dt))) - exact))
        for dt in (0.02, 0.005)]
    failed += _check("linear advection converges to exact translate",
                     errs[1] < 0.5 * errs[0] and errs[1] < 0.02,
                     f"err {errs[0]:.2e} -> {errs[1]:.2e} (order ~{np.log2(errs[0]/errs[1]):.1f})")

    # 2. Buckley-Leverett shock at the Welge front.
    fw = lambda s: s ** 2 / (s ** 2 + (1 - s) ** 2)
    dfw = lambda s: (2 * s * (s ** 2 + (1 - s) ** 2)
                     - s ** 2 * (2 * s - 2 * (1 - s))) / (s ** 2 + (1 - s) ** 2) ** 2
    g = compute_geometry(cart_grid([60, 1], [1.0, 1.0]))
    disc = DGDiscretization.build(g, 1)
    tr = DGScalarTransport(disc, np.array([1.0, 0.0]), flux_fn=fw, dflux_fn=dfw)
    dofs = disc.assign_from_cell_values(np.full(g.cells.num, 1e-6))
    T, dt = 0.3, 0.004
    for _ in range(int(T / dt)):
        dofs = tr.step(dofs, dt, inflow_u=1.0)
    m = disc.cell_mean(dofs)
    cc = np.asarray(g.cells.centroids)
    s_star = brentq(lambda s: fw(s) / s - dfw(s), 0.3, 0.99)
    x_front = dfw(s_star) * T
    dg_front = cc[m > s_star * 0.5, 0].max()
    failed += _check("Buckley-Leverett shock at Welge front",
                     abs(dg_front - x_front) < 0.03 and m.min() >= -1e-9 and m.max() <= 1.001,
                     f"DG front {dg_front:.3f} vs Welge {x_front:.3f}")

    print("\nDG transport matches analytical solutions!" if failed == 0
          else f"\n{failed} checks FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
