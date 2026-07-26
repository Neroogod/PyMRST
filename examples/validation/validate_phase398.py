"""Phase 398: direct port of the complete re-mpfa/functions module."""

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.fv_unsat import getHydraulicProperties, vGM_theta
from pymrst.modules.mpfa import compute_multipoint_trans
from pymrst.modules.re_mpfa import (
    HydraulicHeadMPFAOperators,
    arithmeticMPFA_hydHead,
    newtonSolver,
    timeStepping,
    upstreamMPFA_hydHead,
)
from pymrst.params.rock import make_rock
from pymrst.params.wells_and_bc import add_bc
from pymrst.utils.adi import init_variables_adi


def _grid_and_bc(top_head=-0.1):
    g = compute_geometry(cart_grid([2, 8], [0.2, 1.0]))
    neighbors = np.asarray(g.faces.neighbors)
    boundary = np.any(neighbors < 0, axis=1)
    x = np.asarray(g.faces.centroids)[:, 0]
    y = np.asarray(g.faces.centroids)[:, 1]
    sides = np.flatnonzero(boundary & ((x < 1e-12) | (x > 0.2 - 1e-12)))
    bottom = np.flatnonzero(boundary & (y < 1e-12))
    top = np.flatnonzero(boundary & (y > 1.0 - 1e-12))
    bc = add_bc(None, sides, "flux", 0.0)
    bc = add_bc(bc, bottom, "pressure", -0.5)
    bc = add_bc(bc, top, "pressure", top_head)
    return g, bc


def main():
    g, bc = _grid_and_bc()
    soil = getHydraulicProperties("loam")
    phys = {"flow": {
        "alpha": soil.alpha, "theta_r": soil.theta_r,
        "theta_s": soil.theta_s, "n": soil.n, "m": soil.m,
    }}
    theta, krw, capacity = vGM_theta(phys)
    rock = make_rock(g, 1.0, soil.theta_s)
    discr = compute_multipoint_trans(g, rock)
    operators = HydraulicHeadMPFAOperators(g, discr, bc)
    zeta_cells = 1.0 - np.asarray(g.cells.centroids)[:, -1]
    zeta_faces = 1.0 - np.asarray(g.faces.centroids)[:, -1]
    bc_values = np.zeros(g.faces.num)
    bc_values[np.asarray(bc.face, dtype=int)] = bc.value

    initial = np.full(g.cells.num, -0.5)
    arithmetic = arithmeticMPFA_hydHead(
        krw, g, bc, initial, zeta_cells*0.0, zeta_faces*0.0
    )
    neighbors = np.asarray(g.faces.neighbors)
    internal = np.flatnonzero(np.all(neighbors >= 0, axis=1))
    arithmetic_error = float(np.max(np.abs(
        arithmetic[internal] - krw(np.array([-0.5]))[0]
    )))

    # Synthetic positive/negative face fluxes pin the exact upstream choice.
    trial = np.linspace(-0.8, -0.2, g.cells.num)
    synthetic_flux = np.ones(g.faces.num)
    synthetic_flux[internal[::2]] = -1.0
    upstream = upstreamMPFA_hydHead(
        krw, lambda _: synthetic_flux, lambda _: np.zeros(g.faces.num),
        g, bc, bc_values, trial, np.zeros(g.cells.num), np.zeros(g.faces.num),
    )
    chosen = np.where(
        synthetic_flux[internal] >= 0,
        neighbors[internal, 0], neighbors[internal, 1],
    )
    upstream_error = float(np.max(np.abs(
        upstream[internal] - krw(trial[chosen])
    )))

    volume = np.asarray(g.cells.volumes)
    dt = 1e-3
    (head_ad,) = init_variables_adi(initial)

    def equation(head, old, iterate, step):
        kr_face = arithmeticMPFA_hydHead(
            krw, g, bc, iterate, np.zeros(g.cells.num), np.zeros(g.faces.num)
        )
        raw_flux = operators.F(head) + operators.boundF(bc_values)
        flux = raw_flux * kr_face
        storage = theta(iterate) + capacity(iterate)*(head - iterate)
        return (volume/step)*(storage - theta(old)) + operators.div(flux)

    result = newtonSolver(
        head_ad, equation, 0.0, dt, 1e-10, 40
    )
    final_equation = equation(
        result.hydraulic_head, initial,
        result.hydraulic_head.val, dt,
    )
    residual_norm = float(np.linalg.norm(final_equation.val))
    wetting = float(np.max(result.hydraulic_head.val - initial))

    faster, counter_fast = timeStepping(
        1.0, 0.25, 4.0, 10.0, 2.0, 2,
        np.array([5.0, 10.0]), 1, 3, 7, 2.0, 0.5,
    )
    clipped, counter_clip = timeStepping(
        2.0, 0.25, 4.0, 10.0, 4.0, 5,
        np.array([5.0, 10.0]), 1, 3, 7, 2.0, 0.5,
    )
    slower, _ = timeStepping(
        2.0, 0.25, 4.0, 10.0, 2.0, 9,
        np.array([5.0, 10.0]), 1, 3, 7, 2.0, 0.5,
    )
    timestep_ok = (
        faster == 2.0 and counter_fast == 1
        and clipped == 1.0 and counter_clip == 2
        and slower == 1.0
    )

    print("Phase 398: direct re-mpfa function port")
    print(f"  arithmetic hydraulic-head weighting: {arithmetic_error:.3e}")
    print(f"  upstream flux-direction selection: {upstream_error:.3e}")
    print(f"  Newton final residual norm: {residual_norm:.3e}")
    print(f"  nontrivial infiltration increase: {wetting:.3e}")
    print(f"  adaptive time-step branches: {timestep_ok}")
    print(f"  Newton iterations: {result.iterations}")

    failed = (
        arithmetic_error > 1e-14
        or upstream_error > 1e-14
        or residual_norm > 1e-9
        or wetting < 1e-3
        or not timestep_ok
    )
    print("\nPhase 398 passed." if not failed else "\nPhase 398 FAILED.")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
