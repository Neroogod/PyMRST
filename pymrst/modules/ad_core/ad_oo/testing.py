"""Small models and states for the AD-OO self-tests.

Not an MRST file. The ported state functions each test themselves against a
real model, and building that model is the same few lines every time - a short
line of cells, a simple fluid, phases switched on - so they live here once.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from pymrst.modules.ad_core.statefunctions.state_function import State

DEFAULT_MU: Sequence[float] = (1e-3, 5e-3, 2e-4)
DEFAULT_RHO: Sequence[float] = (1000.0, 700.0, 100.0)
DEFAULT_C: Sequence[float] = (1e-10, 1e-9, 1e-8)
DEFAULT_PRESSURE: float = 100e5


def toy_model(phases: str = "WO", nx: int = 4, *, fluid=None,
              rock_regions: Optional[dict] = None, **fluid_options):
    """A validated `ReservoirModel` on an `nx` x 1 x 1 line of 1 m cells."""
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_core.ad_oo.reservoir_model import ReservoirModel
    from pymrst.modules.ad_props import init_simple_adi_fluid
    from pymrst.params.rock import make_rock
    from pymrst.utils.units import DARCY, MILLI

    grid = compute_geometry(cart_grid([nx, 1, 1], [float(nx), 1.0, 1.0]))
    rock = make_rock(grid, 100*MILLI*DARCY, 0.3)
    if rock_regions is not None:
        rock.regions = rock_regions
    if fluid is None:
        index = ["WOG".index(letter) for letter in phases]
        fluid = init_simple_adi_fluid(
            phases=phases, mu=[DEFAULT_MU[i] for i in index],
            rho=[DEFAULT_RHO[i] for i in index], n=[2.0]*len(phases),
            c=[DEFAULT_C[i] for i in index], **fluid_options)
    model = ReservoirModel(grid, rock, fluid)
    model.water, model.oil, model.gas = ("W" in phases, "O" in phases,
                                         "G" in phases)
    return model.validate_model()


def toy_state(model, pressure: Optional[np.ndarray] = None,
              s: Optional[np.ndarray] = None) -> State:
    """Uniform pressure and evenly split saturations unless given."""
    nc = model.G.cells.num
    nph = model.get_number_of_phases()
    state = State(
        pressure=(np.full(nc, DEFAULT_PRESSURE) if pressure is None
                  else np.asarray(pressure, dtype=float)),
        s=(np.full((nc, nph), 1.0/nph) if s is None
           else np.asarray(s, dtype=float)))
    return model.init_state_function_containers(state)


__all__ = ["toy_model", "toy_state"]


if __name__ == "__main__":
    def test_the_toy_model_has_the_phases_asked_for():
        model = toy_model("WOG")
        assert model.get_phase_names() == "WOG"
        assert model.FlowPropertyFunctions is not None
        print("ok test_the_toy_model_has_the_phases_asked_for")

    def test_the_toy_state_rows_sum_to_one():
        model = toy_model("WO", nx=3)
        state = toy_state(model)
        np.testing.assert_allclose(state["s"].sum(axis=1), 1.0)
        print("ok test_the_toy_state_rows_sum_to_one")

    test_the_toy_model_has_the_phases_asked_for()
    test_the_toy_state_rows_sum_to_one()
    print("\nAll tests passed!")
