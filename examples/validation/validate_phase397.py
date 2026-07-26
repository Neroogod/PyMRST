"""Phase 397: fully implicit Richards step with conservative MPFA fluxes."""

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.fv_unsat import (
    getHydraulicProperties,
    modelRE,
    solverRE,
)
from pymrst.modules.mpfa import compute_multipoint_trans
from pymrst.params.rock import make_rock
from pymrst.params.wells_and_bc import add_bc


def _problem():
    g = compute_geometry(cart_grid([8, 2], [1.0, 0.2]))
    soil = getHydraulicProperties("loam")
    rock = make_rock(g, 1.0, soil.theta_s)
    discr = compute_multipoint_trans(g, rock)
    neighbors = np.asarray(g.faces.neighbors)
    boundary = np.flatnonzero(np.any(neighbors < 0, axis=1))
    values = np.full(boundary.size, -0.5)
    left = g.faces.centroids[boundary, 0] < 1e-12
    values[left] = -0.1
    bc = add_bc(None, boundary, "pressure", values)
    phys = {
        "flow": {
            "alpha": soil.alpha,
            "theta_r": soil.theta_r,
            "theta_s": soil.theta_s,
            "n": soil.n,
            "m": soil.m,
            "gamma": 1.0,
            "mu": 1.0,
        }
    }
    return g, discr, bc, phys


def main():
    g, discr, bc, phys = _problem()
    model = modelRE(g, phys, discr, bc, gravity="off")

    # A constant state with matching boundary data must remain unchanged.
    equilibrium_bc = add_bc(
        None, bc.face, "pressure", np.full(bc.face.size, -0.5)
    )
    equilibrium_model = modelRE(
        g, phys, discr, equilibrium_bc, gravity="off"
    )
    initial = np.full(g.cells.num, -0.5)
    equilibrium = solverRE(
        initial, equilibrium_model, 1e-3,
        tolerance=1e-11, max_iterations=60,
    )
    equilibrium_error = float(np.max(np.abs(
        equilibrium.pressure_head - initial
    )))
    equilibrium_flux = float(np.max(np.abs(equilibrium.face_flux)))

    # Wetting from the left gives a nontrivial nonlinear time step.
    dt = 1e-3
    step = solverRE(
        initial, model, dt, tolerance=1e-11, max_iterations=80
    )
    _, subflux = model.flux(step.pressure_head)
    divergence = np.asarray(discr["e_div"] @ subflux).ravel()[:g.cells.num]
    storage = (
        np.asarray(g.cells.volumes)
        * (model.theta(step.pressure_head) - model.theta(initial))/dt
    )
    cell_balance = float(np.max(np.abs(storage + divergence)))
    global_balance = float(abs(np.sum(storage) + np.sum(divergence)))
    wetting = float(np.max(step.pressure_head - initial))
    bounds_ok = bool(
        np.all(step.pressure_head >= initial - 1e-10)
        and np.all(step.pressure_head <= -0.1 + 1e-10)
    )

    # The MRST-style modified-Picard equation equals the nonlinear equation
    # when its iteration state and trial state coincide.
    zero_source = np.zeros(g.cells.num)
    alias_error = float(np.max(np.abs(
        model.psiEq(
            step.pressure_head, initial, step.pressure_head, dt, zero_source
        )
        - model.residual(step.pressure_head, initial, dt, zero_source)
    )))

    print("Phase 397: fv-unsat Richards/MPFA integration")
    print(f"  equilibrium pressure invariance: {equilibrium_error:.3e}")
    print(f"  equilibrium maximum face flux: {equilibrium_flux:.3e}")
    print(f"  nonlinear residual norm: {step.residual_norm:.3e}")
    print(f"  local cell mass balance: {cell_balance:.3e}")
    print(f"  global mass balance: {global_balance:.3e}")
    print(f"  infiltration head increase: {wetting:.3e}")
    print(f"  bounded wetting profile: {bounds_ok}")
    print(f"  MRST psiEq/nonlinear parity: {alias_error:.3e}")

    failed = (
        equilibrium_error > 1e-12
        or equilibrium_flux > 1e-12
        or step.residual_norm > 1e-10
        or cell_balance > 1e-10
        or global_balance > 1e-10
        or wetting < 1e-3
        or not bounds_ok
        or alias_error > 1e-12
    )
    print("\nPhase 397 passed." if not failed else "\nPhase 397 FAILED.")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
