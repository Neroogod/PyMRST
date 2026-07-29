"""Phase 425: degree-zero DG Buckley-Leverett transport against MRST.

At degree zero, ``TransportModelDG`` has no slope modes to limit and reduces to
implicit finite-volume upwind transport.  This reproduces the 30-cell, 30-step
MRST schedule dumped by ``dump_phase425_reference.m``.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.dg.dg_transport import DGScalarTransport
from pymrst.modules.dg.discretizations.dg_discretization import DGDiscretization


def _check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{('  ' + detail) if detail else ''}")
    return 0 if ok else 1


def main() -> int:
    ref = loadmat(Path(__file__).with_name("ref_p425_dg0_bl.mat"))["sw"]

    def fw(s):
        return s**2 / (s**2 + (1.0 - s)**2)

    def dfw(s):
        den = s**2 + (1.0 - s)**2
        return (2.0 * s * den - s**2 * (4.0 * s - 2.0)) / den**2

    g = compute_geometry(cart_grid([30, 1]))
    disc = DGDiscretization.build(g, 0)
    transport = DGScalarTransport(disc, [1.0, 0.0], fw, dfw)
    dofs = disc.assign_from_cell_values(np.zeros(g.cells.num))
    states = []
    for _ in range(30):
        dofs = transport.step(dofs, 1.0, inflow_u=1.0, maxit=50)
        states.append(disc.cell_mean(dofs))
    py = np.column_stack(states)

    err = np.abs(py - ref)
    failed = 0
    print("Degree-zero DG Buckley-Leverett transport vs MRST:")
    failed += _check("state dimensions", py.shape == ref.shape,
                     f"Python {py.shape}, MRST {ref.shape}")
    failed += _check("all report-step saturations", err.max() < 1.1e-3,
                     f"max abs diff {err.max():.3e}")
    failed += _check("final saturation profile", err[:, -1].max() < 7e-4,
                     f"max abs diff {err[:, -1].max():.3e}")
    failed += _check("physical saturation bounds",
                     py.min() >= -1e-12 and py.max() <= 1.0 + 1e-12,
                     f"range [{py.min():.3e}, {py.max():.3e}]")

    print("\nDegree-zero DG transport matches MRST!" if failed == 0
          else f"\n{failed} checks FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
