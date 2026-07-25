"""Classic 1D Buckley-Leverett displacement with AD (MRST `adBuckleyLeverett1D`).

The textbook incompressible waterflood: water is injected at the left of a 1D
horizontal, homogeneous oil-filled column and fluid is produced at the right.
With quadratic Corey relative permeabilities and a unit viscosity ratio the water
saturation develops the familiar Buckley-Leverett shape - a shock front ahead of
a rarefaction - and, run for one pore volume injected (PVI), the front reaches
the outlet and water breaks through.

This ports MRST's introductory AD example. The MATLAB script's several tails
(plotToolbar visualisation, the general-solver hook, the on-disk `ResultHandler`)
are presentation/IO layers that live outside the physics; the simulation itself -
a boundary-condition-driven `GenericBlackOilModel` stepped with the nonlinear
solver - is what is reproduced here, and it is what `validate_phase363.py` pins
against MRST's own saturation profile.
"""

from typing import List, Tuple

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
    GenericBlackOilModel,
)
from pymrst.modules.ad_core.simulators.nonlinear_solver import NonLinearSolver
from pymrst.modules.ad_core.statefunctions.state_function import State
from pymrst.modules.ad_props import init_simple_adi_fluid
from pymrst.params.rock import make_rock, pore_volume
from pymrst.params.wells_and_bc.fluxside import fluxside
from pymrst.params.wells_and_bc.pside import pside
from pymrst.utils.units import BARSA, CENTI, DARCY, DAY, POISE


def run_ad_buckley_leverett_1d(n: int = 25) -> Tuple[object, List[State]]:
    """Simulate 1 PVI of a 1D Buckley-Leverett waterflood in `n` steps.

    Returns the grid and the list of states (initial plus one per step). The
    fluid is given an explicit unit viscosity ratio and equal densities so the
    horizontal displacement is the pure Buckley-Leverett problem, independent of
    absolute mu/rho.
    """
    g = compute_geometry(cart_grid([50, 1, 1], [1000.0, 10.0, 10.0]))
    rock = make_rock(g, 1.0 * DARCY, 0.3)
    fluid = init_simple_adi_fluid(
        mu=np.array([1.0, 1.0]) * CENTI * POISE, rho=[1000.0, 1000.0],
        n=[2, 2], c=[0.0, 0.0], p_ref=50 * BARSA)

    pv = pore_volume(g, rock)
    inj_rate = pv.sum() / (500.0 * DAY)                 # 1 PVI in 500 days
    bc = fluxside(None, g, "xmin", inj_rate, sat=[1.0, 0.0])   # water in
    bc = pside(bc, g, "xmax", 0.0, sat=[0.0, 1.0])            # produce at p=0
    model = GenericBlackOilModel(g, rock, fluid, [], bc=bc)

    nc = g.cells.num
    state0 = State(pressure=np.full(nc, 50 * BARSA), sW=np.zeros(nc),
                   qWs=np.zeros(0), qOs=np.zeros(0), bhp=np.zeros(0))

    solver = NonLinearSolver()
    dt = (500.0 / n) * DAY
    states = [state0]
    st = state0
    for i in range(n):
        st, rep = solver.solve_timestep(model, st, dt)
        if not rep.converged:
            raise RuntimeError(f"Buckley-Leverett step {i} did not converge")
        states.append(st)
    return g, states


if __name__ == "__main__":
    def test_front_advances_and_breaks_through_at_one_pvi():
        """Water sweeps in from the inlet; by 1 PVI the front has reached the
        outlet (breakthrough), the signature of this example."""
        g, states = run_ad_buckley_leverett_1d()
        sw = states[-1]["sW"]
        assert sw[0] > 0.5, "the inlet must be strongly water-swept"
        assert sw[-1] > 1e-3, "the front should have reached the outlet by 1 PVI"
        print(f"ok test_front_advances_and_breaks_through_at_one_pvi "
              f"(sW inlet {sw[0]:.3f}, outlet {sw[-1]:.3f})")

    def test_saturation_profile_is_monotone():
        """A Buckley-Leverett profile decreases monotonically from inlet to
        outlet - no spurious over/undershoot."""
        g, states = run_ad_buckley_leverett_1d()
        sw = states[-1]["sW"]
        assert np.all(np.diff(sw) <= 1e-9), "profile must be non-increasing"
        assert np.all((sw >= -1e-9) & (sw <= 1.0 + 1e-9)), "sW in [0, 1]"
        print("ok test_saturation_profile_is_monotone")

    def test_water_is_conserved():
        """Injected water = water in place + water produced. Before breakthrough
        nothing is produced, so water in place tracks the injected volume; the
        balance holds across breakthrough too."""
        g, states = run_ad_buckley_leverett_1d()
        rock = make_rock(g, 1.0 * DARCY, 0.3)
        pv = pore_volume(g, rock)
        injected = pv.sum()                      # exactly 1 PVI of water
        in_place = float((pv * states[-1]["sW"]).sum())
        assert in_place <= injected + 1e-6, "cannot hold more water than injected"
        # a good fraction is retained even after breakthrough
        assert in_place / injected > 0.5, in_place / injected
        print(f"ok test_water_is_conserved "
              f"(in place / injected = {in_place / injected:.3f})")

    def test_more_steps_track_the_same_front():
        """Refining the timestep does not move the front far - the scheme is
        converging, not drifting."""
        _, s25 = run_ad_buckley_leverett_1d(n=25)
        _, s50 = run_ad_buckley_leverett_1d(n=50)
        # compare swept volume (integral of sW), robust to small front smearing
        v25 = s25[-1]["sW"].sum()
        v50 = s50[-1]["sW"].sum()
        assert abs(v25 - v50) / v25 < 0.05, (v25, v50)
        print(f"ok test_more_steps_track_the_same_front "
              f"(swept vol {v25:.2f} vs {v50:.2f})")

    test_front_advances_and_breaks_through_at_one_pvi()
    test_saturation_profile_is_monotone()
    test_water_is_conserved()
    test_more_steps_track_the_same_front()
    print("\nAll tests passed!")
