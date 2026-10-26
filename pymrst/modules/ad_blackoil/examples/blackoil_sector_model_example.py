"""MRST ``ad-blackoil/examples/blackoilSectorModelExample.m``.

Boundary conditions for pressure support: SPE1's live-oil fluid (with
`krW = s^2`) on a 50x50x1 sector of 1000x1000x100 m, at 300 bar with
saturations [0.2 0.8 0] and `rs = 200`. A centre producer on oil-rate
control (a quarter of the pore volume over ten years, BHP limit 100 bar)
is run on a `ThreePhaseBlackOilModel` over ramped-up steps of at most 30
days, twice: with all four sides held at 300 bar (`pside`, with a
dissolution matrix putting 200 Sm^3 of gas in each Sm^3 of boundary oil)
and with closed boundaries. `blackoil_sector_model_example` compares the
well curves (`plotWellSols`) and the gas saturation in the well cell.

`build_blackoil_sector_model_example`/`run_blackoil_sector_model_example`,
below, are PyMRST's earlier version on the legacy models (the defaults
reproduce the MRST sizes; smaller ones give a quick run), kept for their
callers.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil.examples.utils.schedule import ExampleResult, run_schedule
from pymrst.modules.ad_blackoil.models import GenericBlackOilModelAD
from pymrst.modules.ad_blackoil.models.generic_black_oil_model import blackoil_status
from pymrst.modules.ad_core import State, rampup_timesteps
from pymrst.modules.ad_props import init_deck_blackoil_fluid
from pymrst.modules.deckformat import read_spe1_deck
from pymrst.params import make_rock, pore_volume, pside, vertical_well
from pymrst.utils.units import BARSA, DARCY, DAY, YEAR


ROOT = Path(__file__).resolve().parents[5]
DEFAULT_DECK = ROOT / "mrst-2024a-Geomech/examples/data/SPE1/BENCH_SPE1.DATA"



def blackoil_sector_model_example(plot: bool = True) -> SimpleNamespace:
    """MRST `blackoilSectorModelExample`."""
    from pymrst.modules.ad_blackoil.models import ThreePhaseBlackOilModel
    from pymrst.modules.ad_core.ad_oo.simulate_schedule_ad import simulate_schedule_ad
    from pymrst.modules.ad_core.timesteps.timestep_selectors import rampup_timesteps
    from pymrst.modules.ad_core.utils.schedule import simple_schedule
    from pymrst.modules.test_suite.dataset_functions.spe1 import setup_spe1
    from pymrst.solvers.init_res_sol import init_res_sol

    _, _, fluid, deck, _ = setup_spe1()
    fluid = dict(fluid)
    fluid["krW"] = lambda s: s**2
    # Set up a simple grid with an initial state
    G = compute_geometry(cart_grid([50, 50, 1], [1000.0, 1000.0, 100.0]))
    rock = make_rock(G, 0.3*DARCY, 0.3)
    s0 = np.array([0.2, 0.8, 0.0])
    p0 = 300*BARSA
    state0 = init_res_sol(G, p0, s0)
    state0.rs = np.full(G.cells.num, 200.0)
    model = ThreePhaseBlackOilModel(G, rock, fluid, disgas=True)
    # Set up driving forces: `ij = ceil(G.cartDims./2)`, 1-based
    T = 10*YEAR
    ij = np.ceil(np.asarray(G.cartDims)/2).astype(int) - 1
    W = vertical_well(None, G, rock, ij[0], ij[1], None,
                      val=-0.25*np.sum(model.operators.pv)/T, well_type="orat",
                      compi=np.array([1, 1, 1])/3, sign=-1, name="Producer")
    W[0].lims = {"bhp": 100*BARSA}
    bc = None
    for side in ("xmin", "xmax", "ymin", "ymax"):
        bc = pside(bc, G, side, p0, sat=s0)
    bc.dissolution = np.array([[1, 0, 0],      # Water fractions in phases
                               [0, 1, 0],      # Oil fractions in phases
                               [0, 200, 1]],   # Gas fractions in phases
                              dtype=float)
    dt = rampup_timesteps(T, 30*DAY)
    schedule = simple_schedule(dt, W=W, bc=bc)
    ws, states, report = simulate_schedule_ad(state0, model, schedule)
    schedule2 = simple_schedule(dt, W=W, bc=None)
    ws_c, states_c, report_c = simulate_schedule_ad(state0, model, schedule2)
    # Compare the two results
    cell = int(np.ravel(W[0].cells)[0])
    gas_open = np.array([np.asarray(x["s"])[cell, 2] for x in states])
    gas_closed = np.array([np.asarray(x["s"])[cell, 2] for x in states_c])
    if plot:
        import matplotlib.pyplot as plt
        from pymrst.modules.ad_core.plotting.plot_well_sols import plot_well_sols
        plot_well_sols([ws, ws_c], np.cumsum(schedule.step_val),
                       datasetnames=["Constant pressure", "Closed boundaries"])
        # Plot the gas in the well cells
        plt.figure()
        plt.plot(np.arange(1, len(states) + 1), np.column_stack([gas_open, gas_closed]),
                 linewidth=2)
        plt.ylabel("S_g")
        plt.xlabel("Step #")
        plt.legend(["Constant pressure", "Closed boundaries"])
        plt.show()
    return SimpleNamespace(G=G, model=model, W=W, bc=bc, schedule=schedule, state0=state0,
                           ws=ws, states=states, report=report, ws_c=ws_c,
                           states_c=states_c, report_c=report_c,
                           gas_open=gas_open, gas_closed=gas_closed)

@dataclass
class SectorModelCase:
    open_model: GenericBlackOilModelAD
    closed_model: GenericBlackOilModelAD
    state0: State
    timesteps: np.ndarray
    producer_cell: int


@dataclass
class SectorModelResult:
    open_boundary: ExampleResult
    closed_boundary: ExampleResult
    producer_cell: int

    @property
    def gas_at_producer(self) -> dict[str, np.ndarray]:
        cell = self.producer_cell
        return {
            "open": self.open_boundary.history["sG"][:, cell],
            "closed": self.closed_boundary.history["sG"][:, cell],
        }


def build_blackoil_sector_model_example(
    dims=(50, 50, 1),
    physical_dims=(1000.0, 1000.0, 100.0),
    total_time: float = 10.0 * YEAR,
    max_timestep: float = 30.0 * DAY,
    rampup_steps: int = 8,
    deck_path=None,
) -> SectorModelCase:
    """Build both variants of the sector experiment without running them."""
    dims = tuple(int(v) for v in dims)
    if len(dims) != 3 or min(dims) < 1:
        raise ValueError("dims must contain three positive integers")
    if total_time <= 0.0 or max_timestep <= 0.0:
        raise ValueError("total_time and max_timestep must be positive")

    grid = compute_geometry(cart_grid(dims, physical_dims))
    rock = make_rock(grid, 0.3 * DARCY, 0.3)
    deck = read_spe1_deck(Path(deck_path) if deck_path is not None else DEFAULT_DECK)
    fluid = replace(init_deck_blackoil_fluid(deck, rock), kr_w=lambda sw: sw**2)

    pv = float(pore_volume(grid, rock).sum())
    oil_rate = -0.25 * pv / (10.0 * YEAR)
    i, j = int(np.ceil(dims[0] / 2.0)) - 1, int(np.ceil(dims[1] / 2.0)) - 1
    wells = vertical_well(
        None, grid, rock, i, j,
        well_type="orat", val=oil_rate, compi=np.full(3, 1.0 / 3.0),
        sign=-1, name="P1", lims={"bhp": 100.0 * BARSA},
    )
    producer_cell = int(wells[0].cells[0])

    p0 = np.full(grid.cells.num, 300.0 * BARSA)
    sw = np.full(grid.cells.num, 0.2)
    so = np.full(grid.cells.num, 0.8)
    sg = np.zeros(grid.cells.num)
    state0 = State(
        pressure=p0, sW=sw, sO=so, sG=sg,
        rs=np.full(grid.cells.num, 200.0),
        status=blackoil_status(sw, so, sg, disgas=True),
        qWs=np.zeros(1), qOs=np.array([oil_rate]), qGs=np.zeros(1),
        bhp=np.array([300.0 * BARSA]),
    )

    sat = np.array([0.2, 0.8, 0.0])
    bc = None
    for side in ("xmin", "xmax", "ymin", "ymax"):
        bc = pside(bc, grid, side, 300.0 * BARSA, sat=sat)
    bc.dissolution = np.array([[1.0, 0.0, 0.0],
                               [0.0, 1.0, 0.0],
                               [0.0, 200.0, 1.0]])

    open_model = GenericBlackOilModelAD(grid, rock, fluid, wells, disgas=True, bc=bc)
    # Each facility owns mutable well controls, so the two runs need separate
    # Well instances even though their initial controls are identical.
    closed_wells = [replace(w, lims=None if w.lims is None else dict(w.lims)) for w in wells]
    closed_model = GenericBlackOilModelAD(grid, rock, fluid, closed_wells, disgas=True)
    timesteps = rampup_timesteps(float(total_time), float(max_timestep), rampup_steps)
    return SectorModelCase(open_model, closed_model, state0, timesteps, producer_cell)


def run_blackoil_sector_model_example(**kwargs) -> SectorModelResult:
    """Run the pressure-supported and sealed sector simulations."""
    case = build_blackoil_sector_model_example(**kwargs)
    keys = ("pressure", "sW", "sO", "sG", "rs", "qOs", "qGs", "bhp")
    open_result = run_schedule(case.open_model, case.state0, case.timesteps,
                               history_keys=keys)
    closed_result = run_schedule(case.closed_model, case.state0, case.timesteps,
                                 history_keys=keys)
    return SectorModelResult(open_result, closed_result, case.producer_cell)


blackoilSectorModelExample = blackoil_sector_model_example

__all__ = ["blackoilSectorModelExample", "blackoil_sector_model_example",
           "build_blackoil_sector_model_example", "run_blackoil_sector_model_example"]


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")

    from pymrst.params.gravity.gravity import isolated_gravity

    with isolated_gravity():
        OUT = blackoil_sector_model_example(plot=True)

    def test_the_boundaries_hold_the_pressure_up():
        p_open = OUT.states[-1]["pressure"].mean()
        p_closed = OUT.states_c[-1]["pressure"].mean()
        assert p_open > p_closed, (p_open/BARSA, p_closed/BARSA)
        print("ok test_the_boundaries_hold_the_pressure_up")

    def test_the_closed_reservoir_frees_more_gas_at_the_well():
        assert OUT.gas_closed[-1] > OUT.gas_open[-1], (OUT.gas_closed[-1], OUT.gas_open[-1])
        print("ok test_the_closed_reservoir_frees_more_gas_at_the_well")

    def test_the_well_is_in_the_centre_cell():
        """MRST's `ceil(cartDims/2)` is cell (25, 25), 0-based 24 + 24*50."""
        assert int(np.ravel(OUT.W[0].cells)[0]) == 24 + 24*50
        print("ok test_the_well_is_in_the_centre_cell")

    test_the_boundaries_hold_the_pressure_up()
    test_the_closed_reservoir_frees_more_gas_at_the_well()
    test_the_well_is_in_the_centre_cell()
    print("\nAll tests passed!")
