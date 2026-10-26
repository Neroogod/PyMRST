"""Visualizing in PyMRST (MRST ``tutorialPlotting.m``).

MRST's tour of the plotting routines: whole grids, subsets selected by a mask,
cell data with a colour bar, and a two-phase displacement whose saturation front
is watched as it advances.

A plotting tutorial is awkward to port faithfully, because what it demonstrates
is pictures. The approach taken here is to keep every computation MRST performs
and to make the plotting optional: ``run_tutorial_plotting`` returns the grid,
the well solution and the saturation history, and the ``plot_*`` helpers draw
them if a backend is installed. That way the tutorial runs headless, its
numerical content is testable, and the figures remain available to anyone
reading it for its original purpose.

The displacement is the part with real content. Water is injected in one corner
at 1 bar and produced in the opposite corner at 0 bar, so the saturation front
sweeps diagonally across the layer, and quantities like total injected volume
and front position can be checked rather than admired.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.incomp import incomp_tpfa_ad as incomp_tpfa
from pymrst.modules.incomp.fluid.incompressible.init_simple_fluid import (
    init_simple_fluid,
)
from pymrst.modules.incomp.transport.implicit_transport import implicit_transport
from pymrst.params.rock import make_rock
from pymrst.params.wells_and_bc.vertical_well import vertical_well
from pymrst.solvers.compute_trans import compute_trans
from pymrst.solvers.init_state import init_state
from pymrst.utils.gridtools.grid_logical_indices import grid_logical_indices
from pymrst.utils.units import BARSA, CENTI, DARCY, DAY, MILLI, POISE


@dataclass
class PlottingTutorial:
    """Everything MRST's plotting tutorial builds, without the figures."""

    G: object
    rock: object
    fluid: object
    wells: list
    state: object
    saturation: np.ndarray  # (nstep+1, ncells, 2), both phases
    time: np.ndarray

    @property
    def resident(self) -> np.ndarray:
        """Saturation of the fluid initially in place, phase 0.

        The reservoir starts full of it and the injector displaces it, so this
        falls with time. Keeping the two phases named rather than indexed is
        deliberate: `s[:, 0]` and `s[:, 1]` are easy to transpose, and doing so
        inverts every trend without changing a shape.
        """
        return self.saturation[:, :, 0]

    @property
    def injected(self) -> np.ndarray:
        """Saturation of the injected fluid, phase 1, which the well supplies."""
        return self.saturation[:, :, 1]

    def even_cell_mask(self) -> np.ndarray:
        """MRST's `mod(1:num, 2) == 0`, the alternating subset it colours."""
        return (np.arange(1, self.G.cells.num + 1) % 2) == 0

    def layer_mask(self, k: int) -> np.ndarray:
        """Cells in logical layer `k`, 0-based."""
        ijk = grid_logical_indices(self.G)
        return np.asarray(ijk[2]).ravel() == int(k)


def build_case(dims=(10, 10, 3), perm: float = 100 * MILLI * DARCY,
               poro: float = 0.5):
    """The quarter-five-spot MRST sets up: inject in one corner, produce in the other."""
    g = compute_geometry(cart_grid(list(dims), list(dims)))
    rock = make_rock(g, perm, poro)
    fluid = init_simple_fluid(mu=[1.0 * CENTI * POISE, 10.0 * CENTI * POISE],
                              rho=[1014.0, 859.0], n=[2.0, 2.0])
    wells = vertical_well(None, g, rock, 0, 0, None, well_type="bhp",
                          val=1 * BARSA, radius=0.1, compi=np.array([0.0, 1.0]),
                          name="I")
    wells = vertical_well(wells, g, rock, dims[0] - 1, dims[1] - 1, None,
                          well_type="bhp", val=0.0, radius=0.1,
                          compi=np.array([1.0, 0.0]), name="P")
    return g, rock, fluid, wells


def run_tutorial_plotting(*, num_steps: int = 10,
                          total_time: float = 100 * DAY) -> PlottingTutorial:
    """Solve pressure once, then advance saturation and keep the history."""
    g, rock, fluid, wells = build_case()
    trans = compute_trans(g, rock)
    state = init_state(g, wells, 0.0, [1.0, 0.0])
    state = incomp_tpfa(state, g, trans, fluid, wells=wells)

    dt = float(total_time) / int(num_steps)
    sats = [np.asarray(state.s, dtype=float).copy()]
    times = [0.0]
    for _ in range(int(num_steps)):
        state = implicit_transport(state, g, dt, rock, fluid, wells=wells)
        state = incomp_tpfa(state, g, trans, fluid, wells=wells)
        sats.append(np.asarray(state.s, dtype=float).copy())
        times.append(times[-1] + dt)

    return PlottingTutorial(G=g, rock=rock, fluid=fluid, wells=wells,
                            state=state, saturation=np.stack(sats),
                            time=np.asarray(times))


def plot_grid_views(case: PlottingTutorial | None = None, **kwargs):
    """MRST's first three figures: whole grid, styled grid, alternating subset.

    Returns ``None`` without a plotting backend, so the tutorial stays runnable.
    `**kwargs` reaches `plot_grid`, which is how a caller passes `show=False`;
    the self-test does, because `show` defaults to True and an interactive
    window blocks the run forever.
    """
    case = run_tutorial_plotting() if case is None else case
    try:
        from pymrst.plotting import plot_grid
    except ImportError:
        return None
    even = case.even_cell_mask()
    return [plot_grid(case.G, **kwargs),
            plot_grid(case.G, np.flatnonzero(even), **kwargs),
            plot_grid(case.G, np.flatnonzero(~even), **kwargs)]


def plot_pressure_and_saturation(case: PlottingTutorial | None = None, **kwargs):
    """Cell pressure and the final saturation field. See `plot_grid_views`."""
    case = run_tutorial_plotting() if case is None else case
    try:
        from pymrst.plotting import plot_cell_data
    except ImportError:
        return None
    return [plot_cell_data(case.G, np.asarray(case.state.pressure), **kwargs),
            plot_cell_data(case.G, case.injected[-1], **kwargs)]


if __name__ == "__main__":
    def test_case_geometry_and_wells():
        g, rock, fluid, wells = build_case()
        assert g.cells.num == 300, g.cells.num
        assert len(wells) == 2, len(wells)
        # A vertical well perforates the whole column: 3 layers here.
        assert np.asarray(wells[0].cells).size == 3, wells[0].cells
        assert np.asarray(wells[1].cells).size == 3
        print("ok test_case_geometry_and_wells")

    def test_alternating_mask_splits_the_grid_in_half():
        """MRST colours `mod(1:num,2)==0` red and the rest blue."""
        case = run_tutorial_plotting(num_steps=1, total_time=1 * DAY)
        even = case.even_cell_mask()
        assert even.sum() == 150 and (~even).sum() == 150, even.sum()
        print("ok test_alternating_mask_splits_the_grid_in_half")

    def test_pressure_lies_between_the_well_controls():
        """Both wells are on bhp, so no cell can fall outside their range.

        A pressure outside [0, 1] bar would mean the well model or the solve is
        wrong, and it is exactly what a colour plot would fail to reveal.
        """
        case = run_tutorial_plotting(num_steps=1, total_time=1 * DAY)
        p = np.asarray(case.state.pressure, dtype=float)
        assert p.min() >= -1e-6 and p.max() <= 1 * BARSA + 1e-6, (p.min(), p.max())
        print("ok test_pressure_lies_between_the_well_controls")

    def test_saturation_stays_physical_and_advances():
        """Water saturation must stay in [0,1] and increase over time.

        Bounds are the check a transport solver most needs: overshoot past 1 or
        below 0 is unphysical and shows up as a plausible-looking colour.
        """
        case = run_tutorial_plotting(num_steps=8, total_time=200 * DAY)
        s = case.saturation
        assert s.min() >= -1e-9 and s.max() <= 1.0 + 1e-9, (s.min(), s.max())
        assert np.allclose(s.sum(axis=2), 1.0, atol=1e-9), "saturations do not close"
        # The reservoir starts full of phase 0, so the injected phase grows and
        # the resident phase shrinks. Asserting the wrong direction here is easy
        # and would pass silently if only bounds were checked.
        inj = case.injected.sum(axis=1)
        res = case.resident.sum(axis=1)
        assert np.all(np.diff(inj) > -1e-9), "injected fluid disappeared"
        assert np.all(np.diff(res) < 1e-9), "resident fluid increased"
        assert inj[0] == 0.0 and inj[-1] > 0.0, (inj[0], inj[-1])
        print("ok test_saturation_stays_physical_and_advances")

    def test_front_moves_away_from_the_injector():
        """The injector corner floods first, the producer corner last."""
        case = run_tutorial_plotting(num_steps=8, total_time=200 * DAY)
        s_end = case.injected[-1]
        inj = np.asarray(case.wells[0].cells, dtype=np.int64)
        prod = np.asarray(case.wells[1].cells, dtype=np.int64)
        assert s_end[inj].mean() > s_end[prod].mean(), (
            s_end[inj].mean(), s_end[prod].mean())
        print("ok test_front_moves_away_from_the_injector")

    def test_layer_mask_selects_one_layer():
        case = run_tutorial_plotting(num_steps=1, total_time=1 * DAY)
        for k in range(3):
            assert case.layer_mask(k).sum() == 100, (k, case.layer_mask(k).sum())
        print("ok test_layer_mask_selects_one_layer")

    def test_plot_helpers_are_optional():
        """Without a backend these return None rather than raising."""
        case = run_tutorial_plotting(num_steps=1, total_time=1 * DAY)
        for fn in (plot_grid_views, plot_pressure_and_saturation):
            out = fn(case, show=False)
            assert out is None or isinstance(out, list), type(out)
        print("ok test_plot_helpers_are_optional")

    test_case_geometry_and_wells()
    test_alternating_mask_splits_the_grid_in_half()
    test_pressure_lies_between_the_well_controls()
    test_saturation_stays_physical_and_advances()
    test_front_moves_away_from_the_injector()
    test_layer_mask_selects_one_layer()
    test_plot_helpers_are_optional()
    print("\nAll tests passed!")
