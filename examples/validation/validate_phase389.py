"""Phase 389: MRST-layout MPFA entry points and rate/BHP wells."""

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.incomp.fluid import init_single_fluid
from pymrst.modules.mpfa import computeMultiPointTrans, incompMPFA
from pymrst.modules.mpfa.compute_multi_point_trans import (
    computeMultiPointTrans as layout_compute,
)
from pymrst.modules.mpfa.private.incomp_mpfa_legacy import incompMPFAlegacy
from pymrst.params.rock import make_rock
from pymrst.params.wells_and_bc import add_well
from pymrst.solvers.state import init_state
from pymrst.utils.units import BARSA, CENTI, DARCY, MILLI, POISE


def main():
    g = compute_geometry(cart_grid([5, 5, 1], [500.0, 500.0, 10.0]))
    rock = make_rock(g, 100.0*MILLI*DARCY, 0.2)
    fluid = init_single_fluid(1.0*CENTI*POISE, 1000.0)
    operators = computeMultiPointTrans(g, rock)

    rate = 5e-3
    wells = add_well(
        None, g, rock, np.array([0]), well_type="rate", val=rate, radius=0.1
    )
    wells = add_well(
        wells,
        g,
        rock,
        np.array([24]),
        well_type="bhp",
        val=100.0*BARSA,
        radius=0.1,
    )
    state = init_state(g, wells, 0.0)
    state = incompMPFA(state, g, operators, fluid, wells=wells)

    injection = float(state.wellSol[0].flux.sum())
    production = float(state.wellSol[1].flux.sum())
    bhp = float(state.wellSol[1].pressure)
    balance = abs(injection + production)
    pressure_order = (
        state.wellSol[0].pressure > state.pressure[0]
        > state.pressure[24] > state.wellSol[1].pressure - 1e-8
    )

    # The MRST-shaped public, private, and compatibility-module entry points
    # must all resolve and give the same state.
    state2 = init_state(g, wells, 0.0)
    state2 = incompMPFAlegacy(state2, g, layout_compute(g, rock), fluid, wells=wells)
    entrypoint_error = float(np.max(np.abs(state.pressure - state2.pressure)))

    print("Phase 389: MPFA wells and MRST module layout")
    print(f"  rate-well injection: {injection:.6e}")
    print(f"  BHP-well production: {production:.6e}")
    print(f"  well material balance: {balance:.3e}")
    print(f"  producer BHP error: {abs(bhp - 100.0*BARSA):.3e}")
    print(f"  pressure ordering: {pressure_order}")
    print(f"  public/private entry-point parity: {entrypoint_error:.3e}")

    failed = (
        abs(injection - rate) > 1e-10
        or abs(production + rate) > 1e-10
        or balance > 1e-10
        or abs(bhp - 100.0*BARSA) > 1e-8
        or not pressure_order
        or entrypoint_error > 1e-10
    )
    print("\nPhase 389 passed." if not failed else "\nPhase 389 FAILED.")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
