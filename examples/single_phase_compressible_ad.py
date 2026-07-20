"""Single-phase compressible AD simulator (MRST flowSolverTutorialAD.m).

Chapter 7 of the MRST book: a 10x10x10 box with constant-compressibility
rock and fluid is produced through a horizontal bhp-controlled well.
The pressure equation is discretized with backward Euler + TPFA and the
Newton Jacobian comes entirely from automatic differentiation.

The only deviation from the MATLAB tutorial: the hydrostatic initial
pressure uses the closed-form solution of dp/dz = g rho(p) instead of
ode23, so MATLAB and Python start from identical states.
"""

from typing import Dict

import numpy as np
import scipy.sparse.linalg as spla

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.params import NORM_GRAVITY, add_well, gravity_vector, make_rock, pore_volume
from pymrst.modules.ad_core.utils.operators import setup_operators_tpfa
from pymrst.utils.adi import (
    ad_exp,
    ad_sum,
    combine_equations,
    init_variables_adi,
    value,
)
from pymrst.utils.units import BARSA, CENTI, DARCY, DAY, MILLI, POISE

# Model parameters (identical to flowSolverTutorialAD.m)
NX, NY, NZ = 10, 10, 10
DX, DY, DZ = 200.0, 200.0, 50.0
PERM = 30 * MILLI * DARCY
PORO = 0.3
C_ROCK = 1e-6 / BARSA
P_REF = 200 * BARSA
MU = 5 * CENTI * POISE
C_FLUID = 1e-3 / BARSA
RHO_REF = 850.0
RHO_SURFACE = 750.0
BHP_TARGET = 100 * BARSA
NUM_STEPS = 52
TOT_TIME = 365 * DAY
NEWTON_TOL = 1e-5
MAX_ITS = 10


def hydrostatic_pressure(z: np.ndarray) -> np.ndarray:
    """Closed-form solution of dp/dz = g rho_r exp(c(p - p_r)), p(0) = p_r."""
    a = C_FLUID * NORM_GRAVITY * RHO_REF * np.exp(-C_FLUID * P_REF)
    return -np.log(np.exp(-C_FLUID * P_REF) - a * z) / C_FLUID


def run_simulation(verbose: bool = False) -> Dict[str, np.ndarray]:
    """Run the 52-step depletion; return pressure/bhp/rate histories."""
    g = compute_geometry(cart_grid([NX, NY, NZ], [DX, DY, DZ]))
    rock = make_rock(g, PERM, PORO)
    pv_r = pore_volume(g, rock)
    ops = setup_operators_tpfa(g, rock)

    grav = NORM_GRAVITY
    pv = lambda p: pv_r * ad_exp(C_ROCK * (p - P_REF))
    rho = lambda p: RHO_REF * ad_exp(C_FLUID * (p - P_REF))

    # Horizontal well perforating (i=1, j=1..8, k=4) in 0-based indices.
    nperf = 8
    well_cells = 1 + NX * np.arange(1, nperf + 1) + NX * NY * 4
    w = add_well(None, g, rock, well_cells, well_dir="y", radius=0.1,
                 gravity=gravity_vector(3))[0]
    wc, wi, dz = w.cells, w.WI, w.dZ

    gradz = ops.grad(g.cells.centroids[:, 2])
    flux = lambda p: -(ops.T / MU) * (ops.grad(p) - grav * ops.avg(rho(p)) * gradz)
    pres_eq = lambda p, p0, dt: (
        (1.0 / dt) * (pv(p) * rho(p) - pv_r * np.exp(C_ROCK * (p0 - P_REF))
                      * RHO_REF * np.exp(C_FLUID * (p0 - P_REF)))
        + ops.div(ops.avg(rho(p)) * flux(p))
    )

    p_conn = lambda bhp: bhp + grav * dz * rho(bhp)
    q_conn = lambda p, bhp: wi * (rho(p[wc]) / MU) * (p_conn(bhp) - p[wc])
    rate_eq = lambda p, bhp, qs: qs - ad_sum(q_conn(p, bhp)) / RHO_SURFACE
    ctrl_eq = lambda bhp: bhp - BHP_TARGET

    p_init = hydrostatic_pressure(g.cells.centroids[:, 2])
    p_ad, bhp_ad, qs_ad = init_variables_adi(p_init, p_init[wc[0]], 0.0)
    nc = g.cells.num

    dt = TOT_TIME / NUM_STEPS
    p_hist = np.zeros((NUM_STEPS + 1, nc))
    bhp_hist = np.zeros(NUM_STEPS + 1)
    qs_hist = np.zeros(NUM_STEPS + 1)
    p_hist[0] = value(p_ad)
    bhp_hist[0] = value(bhp_ad)[0]

    for step in range(1, NUM_STEPS + 1):
        p0 = value(p_ad).copy()
        converged = False
        for nit in range(MAX_ITS):
            eq1 = pres_eq(p_ad, p0, dt)
            eq1[wc] = eq1[wc] - q_conn(p_ad, bhp_ad)
            eq = combine_equations(
                eq1, rate_eq(p_ad, bhp_ad, qs_ad), ctrl_eq(bhp_ad)
            )
            res = eq.val
            upd = spla.spsolve(eq.jac[0].tocsc(), -res)
            p_ad.val = p_ad.val + upd[:nc]
            bhp_ad.val = bhp_ad.val + upd[nc:nc + 1]
            qs_ad.val = qs_ad.val + upd[nc + 1:]
            residual = np.linalg.norm(res)
            if verbose:
                print(f"  step {step:2d} it {nit + 1}: res = {residual:.4e}")
            if residual <= NEWTON_TOL:
                converged = True
                break
        assert converged, f"Newton did not converge in step {step}"
        p_hist[step] = value(p_ad)
        bhp_hist[step] = value(bhp_ad)[0]
        qs_hist[step] = value(qs_ad)[0]

    return {
        "p_hist": p_hist, "bhp_hist": bhp_hist, "qs_hist": qs_hist,
        "p_init": p_init, "WI": wi, "dZ": dz, "pv_r": pv_r,
    }


if __name__ == "__main__":
    out = run_simulation()

    def test_depletion_physics():
        """Pressure declines monotonically toward bhp; rates are production."""
        p_avg = out["p_hist"].mean(axis=1)
        assert p_avg[0] > 200 * BARSA  # hydrostatic head above datum
        assert np.all(np.diff(p_avg) < 0), "average pressure must decline"
        assert p_avg[-1] > BHP_TARGET, "cannot deplete below bhp"
        assert np.all(out["qs_hist"][1:] < 0), "well must produce"
        assert np.all(np.diff(-out["qs_hist"][2:]) < 0), (
            "production rate must decay"
        )
        assert np.allclose(out["bhp_hist"][1:], BHP_TARGET)
        print("ok test_depletion_physics")

    def test_mass_balance():
        """Produced surface volume equals reservoir mass loss / rhoS."""
        dt = TOT_TIME / NUM_STEPS
        mass = lambda p: np.sum(
            out["pv_r"] * np.exp(C_ROCK * (p - P_REF))
            * RHO_REF * np.exp(C_FLUID * (p - P_REF))
        )
        produced = -out["qs_hist"][1:].sum() * dt * RHO_SURFACE
        lost = mass(out["p_hist"][0]) - mass(out["p_hist"][-1])
        assert np.isclose(produced, lost, rtol=1e-8), (
            f"produced {produced:.6e} kg vs lost {lost:.6e} kg"
        )
        print("ok test_mass_balance")

    def test_drawdown_at_well():
        """Largest drawdown occurs in the perforated layer."""
        drop = out["p_hist"][0] - out["p_hist"][5]
        assert drop.argmax() in (1 + 10 * np.arange(1, 9) + 400)
        print("ok test_drawdown_at_well")

    test_depletion_physics()
    test_mass_balance()
    test_drawdown_at_well()
    print("\nAll tests passed!")
    print(f"final avg pressure: {out['p_hist'][-1].mean() / BARSA:.2f} bar, "
          f"final rate: {-out['qs_hist'][-1] * DAY:.2f} m^3/day")
