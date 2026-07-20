"""Reservoir and well-state initialization, ported from MRST `initState`."""

from typing import Optional, Sequence, Union

import numpy as np

from pymrst.gridprocessing.grid import Grid
from pymrst.params.wells_and_bc import Well
from pymrst.solvers.init_res_sol import init_res_sol
from pymrst.solvers.state_types import State, WellSol

Scalar = Union[float, np.ndarray]


def init_state(
    g: Grid,
    wells: Optional[Sequence[Well]],
    p0: Scalar,
    s0: Scalar = 1.0,
) -> State:
    """Initialize reservoir state plus one `WellSol` per well."""
    state = init_res_sol(g, p0, s0)
    for well in wells or []:
        state.wellSol.append(
            WellSol(flux=np.zeros(well.cells.size), pressure=float(well.val))
        )
    return state


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.params.rock import make_rock
    from pymrst.params.wells_and_bc import add_well

    def test_init_state_with_wells() -> None:
        g = compute_geometry(cart_grid([3, 3, 2]))
        rock = make_rock(g, 1e-13, 0.2)
        wells = add_well(None, g, rock, np.array([0, 9]), well_type="bhp", val=2e7)
        state = init_state(g, wells, 1e7)
        assert len(state.wellSol) == 1
        assert state.wellSol[0].flux.shape == (2,)
        assert state.wellSol[0].pressure == 2e7
        print("ok test_init_state_with_wells")

    test_init_state_with_wells()
    print("\nAll tests passed!")

