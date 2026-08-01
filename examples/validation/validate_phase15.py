"""Phase 15: dual porosity against MRST's dual-porosity module.

Replays both of MRST's cases - a depleting fractured column (WaterModelDP)
and capillary imbibition (TwoPhaseOilWaterModelDP) - and compares every
timestep of both continua.
"""

import sys
from pathlib import Path

import numpy as np
import scipy.io

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pymrst.gridprocessing import cart_grid, compute_geometry       # noqa: E402
from pymrst.modules.ad_props.init_simple_adi_fluid import (              # noqa: E402
    init_simple_adi_fluid,
)
from pymrst.modules.dual_porosity import (                          # noqa: E402
    KazemiShapeFactor,
    KazemiSinglePhaseTransfer,
    KazemiTwoPhaseTransfer,
    TwoPhaseOilWaterModelDP,
    WaterModelDP,
)
from pymrst.params.rock import make_rock                            # noqa: E402
from pymrst.params.wells_and_bc import add_bc                       # noqa: E402
from pymrst.utils.units import BARSA, CENTI, DARCY, MILLI, POISE    # noqa: E402

REF = Path(__file__).with_name("ref_p15_dual_porosity.mat")
PSIA = 6894.757293168361


def _boundary(g, x, value, sat):
    faces = np.flatnonzero(
        (np.abs(g.faces.centroids[:, 0] - x) < 1e-9)
        & (g.faces.neighbors.min(axis=1) < 0))
    return faces, value, sat


def run_single_phase(ref):
    g = compute_geometry(cart_grid([100, 1, 1], [60.0, 1.0, 1.0]))
    nc = g.cells.num
    rock_m = make_rock(g, np.full((nc, 1), 1 * DARCY), np.full((nc, 1), 0.3))
    rock_f = make_rock(g, np.full((nc, 1), 1000 * DARCY),
                       np.full((nc, 1), 0.01))
    # initSimpleADIFluid('phases','W','c',1e-06): mu = 1 Pa.s, rho = 1, pRef = 0
    fluid = init_simple_adi_fluid(mu=[1.0, 1.0], rho=[1.0, 1.0], n=[1.0, 1.0],
                                  c=[1e-6, 1e-6], p_ref=0.0)

    transfer = KazemiSinglePhaseTransfer(
        shape_factor=KazemiShapeFactor(block_dimension=[10.0, 10.0, 10.0]))

    faces = np.flatnonzero((g.faces.centroids[:, 0] < 1e-9)
                           & (g.faces.neighbors.min(axis=1) < 0))
    bc = add_bc(None, faces, "pressure", 0.0, sat=np.ones((faces.size, 1)))

    model = WaterModelDP(g, rock_f, rock_m, fluid, fluid, transfer, bc=bc)
    st = model.initial_state(1000 * PSIA)

    n, dt = int(ref["n1"]), float(ref["dt1"])
    p = np.zeros((nc, n))
    pm = np.zeros((nc, n))
    for k in range(n):
        st, ok, _ = model.step(st, dt)
        assert ok, f"1ph step {k} did not converge"
        p[:, k] = st.pressure
        pm[:, k] = st.pressure_matrix
    return p, pm


def run_two_phase(ref):
    g = compute_geometry(cart_grid([30, 1, 1], [30.0, 1.0, 1.0]))
    nc = g.cells.num
    rock_m = make_rock(g, np.full((nc, 1), 10 * MILLI * DARCY),
                       np.full((nc, 1), 0.1))
    rock_f = make_rock(g, np.full((nc, 1), 100 * MILLI * DARCY),
                       np.full((nc, 1), 0.01))
    fluid = init_simple_adi_fluid(
        mu=[1 * CENTI * POISE, 1 * CENTI * POISE], rho=[1000.0, 800.0],
        n=[1.0, 1.0], c=[1e-12, 1e-12], p_ref=0.0)

    Pe = float(ref["Pe"])

    def pc(sw):
        return Pe

    transfer = KazemiTwoPhaseTransfer(
        shape_factor=KazemiShapeFactor(block_dimension=[20.0, 20.0, 20.0]),
        pc_ow_matrix=pc)

    left = np.flatnonzero((g.faces.centroids[:, 0] < 1e-9)
                          & (g.faces.neighbors.min(axis=1) < 0))
    right = np.flatnonzero((g.faces.centroids[:, 0] > 30.0 - 1e-9)
                           & (g.faces.neighbors.min(axis=1) < 0))
    bc = add_bc(None, left, "pressure", 70 * BARSA,
                sat=np.tile([1.0, 0.0], (left.size, 1)))
    bc = add_bc(bc, right, "pressure", 0.0,
                sat=np.tile([1.0, 0.0], (right.size, 1)))

    model = TwoPhaseOilWaterModelDP(g, rock_f, rock_m, fluid, fluid, transfer,
                                    bc=bc, pc_ow_matrix=pc)
    st = model.initial_state(0.0, 0.0)

    n, dt = int(ref["n2"]), float(ref["dt2"])
    out = [np.zeros((nc, n)) for _ in range(4)]
    for k in range(n):
        st, ok, _ = model.step(st, dt)
        assert ok, f"2ph step {k} did not converge"
        for arr, v in zip(out, (st.pressure, st.sW, st.pressure_matrix,
                                st.sW_matrix)):
            arr[:, k] = v
    return out


def main() -> int:
    if not REF.exists():
        print(f"{REF.name} missing (run dump_phase15_reference.m in MATLAB)")
        return 0

    ref = scipy.io.loadmat(REF, squeeze_me=True)
    ok = True
    checks = []

    # The single-phase case converges to round-off on both sides, so it is
    # held to it. The two-phase case cannot be: the upstream switch in the
    # transfer term makes the last Newton steps chatter, so MRST's reference
    # is itself converged only to 1e-10 per step (tighten it further and MRST
    # cuts the timestep, which changes the discretization and is a far larger
    # error). 1e-8 over six steps is what that stopping rule can support.
    p, pm = run_single_phase(ref)
    p_ref = np.atleast_2d(ref["p1"])
    pm_ref = np.atleast_2d(ref["pm1"])
    scale = float(np.abs(p_ref).max())
    checks += [
        ("1ph fracture pressure", np.abs(p - p_ref).max() / scale, 1e-12),
        ("1ph matrix pressure", np.abs(pm - pm_ref).max() / scale, 1e-12),
    ]

    p2, sw2, pm2, swm2 = run_two_phase(ref)
    p2r = np.atleast_2d(ref["p2"])
    sw2r = np.atleast_2d(ref["sw2"])
    pm2r = np.atleast_2d(ref["pm2"])
    swm2r = np.atleast_2d(ref["swm2"])
    scale2 = float(np.abs(p2r).max())
    checks += [
        ("2ph fracture pressure", np.abs(p2 - p2r).max() / scale2, 1e-8),
        ("2ph matrix pressure", np.abs(pm2 - pm2r).max() / scale2, 1e-8),
        ("2ph fracture saturation", np.abs(sw2 - sw2r).max(), 1e-8),
        ("2ph matrix saturation", np.abs(swm2 - swm2r).max(), 1e-8),
    ]

    print(f"Dual porosity: {p.shape[1]} steps of depletion (100 cells), "
          f"{sw2.shape[1]} steps of imbibition (30 cells)\n")
    for name, val, tol in checks:
        good = val <= tol
        ok &= good
        print(f"  {'ok  ' if good else 'FAIL'} {name:26s} "
              f"max diff = {val:.2e}  (tol {tol:.0e})")

    print(f"\n  (MRST: fracture drains to {p_ref[0, -1] / PSIA:.1f} psia while "
          f"the matrix holds {pm_ref[0, -1] / PSIA:.1f} psia;")
    print(f"   imbibition fills the matrix to sW = {swm2r[0, -1]:.3f} behind a "
          f"fracture at {sw2r[0, -1]:.3f})")

    print("\nDual porosity matches MATLAB!" if ok else "\ncomparisons FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
