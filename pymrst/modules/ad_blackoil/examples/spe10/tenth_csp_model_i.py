"""MRST ``ad-blackoil/examples/spe10/tenthCSP_Model_I.m``: model one of the
tenth SPE comparative solution project.

A 100x1x20 Cartesian box (25x25x2.5 ft cells) with the benchmark's
geostatistical permeability, an incompressible, immiscible oil-gas fluid with
the official SGOF relative permeabilities, a gas injector at I = 1 (6.97
m^3/day) and a producer at I = 100 (95 psia), initially full of oil at 100
psia at the top, run for ten years with ramp-up time steps. MRST's script,
section by section; `plot=False` leaves out its figures, `nsteps` (a PyMRST
addition) runs only the first report steps.

`setup_tenth_csp_model_i_ad` and `run_tenth_csp_model_i_ad`, below, are
PyMRST's earlier version on the legacy two-phase model, kept for their
callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry, remove_cells
from pymrst.modules.ad_blackoil.models import GenericBlackOilModelAD
from pymrst.modules.ad_core import NonLinearSolver, SimulationReport, State
from pymrst.modules.ad_core.utils.schedule import simple_schedule
from pymrst.modules.ad_props import ADFluid, interp_table
from pymrst.modules.deckformat import load_spe10_model1
from pymrst.params import NORM_GRAVITY, add_well, make_rock
from pymrst.utils.units import CENTI, DAY, FT, POISE, PSIA

#: Ten years (3650 days) of ramp-up time steps, in days.
TIMESTEP_DAYS = np.r_[[0.1, 0.2, 0.3, 0.4], np.repeat(0.5, 6), np.ones(6),
                      np.repeat(2.0, 5), np.repeat(5.0, 8), np.repeat(10.0, 9),
                      np.repeat(20.0, 5), np.repeat(30.0, 50), np.repeat(50.0, 38)]


def tenth_csp_model_i(plot: bool = True, nsteps: Optional[int] = None) -> SimpleNamespace:
    """MRST `tenthCSP_Model_I`."""
    from pymrst.modules.ad_blackoil.models.generic_black_oil_model import GenericBlackOilModel
    from pymrst.modules.ad_core.ad_oo.simulate_schedule_ad import simulate_schedule_ad
    from pymrst.modules.ad_core.initialization.get_initialization_regions_black_oil import (
        get_initialization_regions_black_oil,
    )
    from pymrst.modules.ad_core.initialization.init_state_black_oil_ad import (
        init_state_black_oil_ad,
    )
    from pymrst.modules.ad_props import init_simple_adi_fluid
    from pymrst.modules.ad_props.props.assign_sgof import assign_sgof
    from pymrst.modules.ad_props.utils.handle_regions import Regions
    from pymrst.modules.spe10.get_spe10_model_1_relperm import get_spe10_model_1_relperm
    from pymrst.modules.spe10.get_spe10_model_1_rock import get_spe10_model_1_rock
    from pymrst.params.gravity.gravity import gravity
    from pymrst.utils.gridtools.grid_logical_indices import grid_logical_indices
    from pymrst.utils.units import BARSA, KILOGRAM, METER, MILLI, DARCY

    gravity("reset", "on")
    # Model geometry
    cartDims = np.array([100, 1, 20])
    physDims = cartDims*np.array([25.0, 25.0, 2.5])*FT
    G = compute_geometry(cart_grid(cartDims, physDims))
    # Petrophysical properties
    rock = get_spe10_model_1_rock()
    if plot:
        from pymrst.modules.ad_blackoil.examples.utils.figures import cell_data_figure
        cell_data_figure(G, np.log10(rock.perm[:, 0]/(MILLI*DARCY)), view=(-37.5, 30))
    # Define sources and sinks
    IJK = grid_logical_indices(G)
    I = np.flatnonzero(IJK[0] == 0)
    P = np.flatnonzero(IJK[0] == G.cartDims[0] - 1)
    W = add_well([], G, rock, I, compi=np.array([0.0, 1.0]), well_type="rate",
                 val=6.97*METER**3/DAY, radius=0.5*FT, well_dir="z", sign=+1, name="I",
                 ref_depth=0*FT)
    W = add_well(W, G, rock, P, compi=np.array([1.0, 1.0]), well_type="bhp",
                 val=95*PSIA, radius=0.5*FT, well_dir="z", sign=-1, name="P",
                 ref_depth=0*FT)
    # Official benchmark relative permeability data
    kr_deck = get_spe10_model_1_relperm()
    if plot:
        _plot_relperm(kr_deck["PROPS"]["SGOF"][0])
    # Fluid properties
    fluid = init_simple_adi_fluid(mu=np.array([1.0, 0.01])*CENTI*POISE,
                                  rho=np.array([700.0, 1.0])*KILOGRAM/METER**3,
                                  c_r=6.0e-5/BARSA, phases="OG")
    # Replace the synthetic relative permeability curves with the benchmark's
    reg = Regions()
    reg.sat = 1
    reg.interp1d = interp_table
    fluid_kr = assign_sgof(dict(vars(fluid)), kr_deck["PROPS"]["SGOF"], reg)
    fluid.krG = fluid_kr["krG"][0]
    fluid.krO = fluid_kr["krOG"][0]
    # Form reservoir model: an incompressible, immiscible oil/gas system
    model = GenericBlackOilModel(G, rock, fluid, gravity=gravity(), disgas=False,
                                 vapoil=False, water=False, oil=True, gas=True)
    # Initialise formation: full of oil, 100 psia at the top
    region = get_initialization_regions_black_oil(model, 0, datum_pressure=100*PSIA)
    state0, _ = init_state_black_oil_ad(model, region)
    if plot:
        from pymrst.modules.ad_blackoil.examples.utils.figures import cell_data_figure
        cell_data_figure(G, np.asarray(state0["pressure"])/PSIA,
                         "Initial Pressure Distribution [Psi]", view=(-37.5, 30))
    # Ten years (3650*day), ramp-up time-steps
    timesteps = TIMESTEP_DAYS*DAY
    if nsteps is not None:
        timesteps = timesteps[:nsteps]
    schedule = simple_schedule(timesteps, W=W)
    fn = None
    if plot:
        from pymrst.modules.ad_core.plotting.get_plot_after_step import get_plot_after_step
        fn = get_plot_after_step(state0, model, schedule, view=[50, 50], field="s:1", wells=W)
    wellSols, states, report = simulate_schedule_ad(state0, model, schedule, afterStepFn=fn)
    return SimpleNamespace(G=G, rock=rock, W=W, fluid=fluid, model=model, state0=state0,
                           schedule=schedule, wellSols=wellSols, states=states, report=report)


def _plot_relperm(sgof: np.ndarray) -> None:
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(sgof[:, 0], sgof[:, 1:3], "*-", linewidth=2, markersize=5)
    plt.legend(["kr_g", "kr_o"], loc="best")
    plt.xlabel("S_g")
    plt.title("Relative Permeability, Model I 10th CSP")


tenthCSP_Model_I = tenth_csp_model_i


# ---------------------------------------------------------------------------
# PyMRST's earlier version.

@dataclass
class SPE10Model1Result:
    dataset: dict
    model: GenericBlackOilModelAD
    schedule: object
    initial_state: State
    states: list[State]
    report: SimulationReport
    history: dict[str, np.ndarray]


def _fluid(dataset, p_ref):
    table = dataset["sgof"]
    sg, krg, kro = table[:, 0], table[:, 1], table[:, 2]
    one = lambda p: p*0.0 + 1.0
    fluid = ADFluid(
        # Internally the generic two-phase slots represent oil and gas.
        rhoS=np.array([700.0, 1.0]),
        mu=np.array([1.0, 0.01])*CENTI*POISE,
        b_w=one, b_o=one,
        kr_w=lambda so: interp_table(sg, kro, 1.0-so),
        kr_o=lambda gas_saturation: interp_table(sg, krg, gas_saturation),
    )
    # MRST initSimpleADIFluid uses the linear rock multiplier
    # 1 + cR*(p - pRef), with its default pRef=0.
    fluid.pv_mult = lambda p: 1.0 + (6.0e-5/1.0e5)*p
    return fluid


def _timesteps():
    days = np.r_[
        [0.1, 0.2, 0.3, 0.4], np.repeat(0.5, 6), np.ones(6),
        np.repeat(2.0, 5), np.repeat(5.0, 8), np.repeat(10.0, 9),
        np.repeat(20.0, 5), np.repeat(30.0, 50), np.repeat(50.0, 38),
    ]
    return days*DAY


def setup_tenth_csp_model_i_ad(*, layers=None, active=None, data_root=None):
    """Construct the official 100×1×20 Model I or an active-cell subset."""
    dataset = load_spe10_model1(
        layers=layers, active=active, data_root=data_root
    )
    full = cart_grid(dataset["cartDims"], np.asarray(dataset["cartDims"])*
                     np.array([25.0, 25.0, 2.5])*FT)
    grid = remove_cells(full, ~dataset["active"])
    grid = compute_geometry(grid)
    rock = make_rock(grid, dataset["perm"], dataset["poro"][:, None])
    original = np.asarray(grid.cells.indexMap, dtype=int)
    inject_cells = np.flatnonzero(original % 100 == 0)
    produce_cells = np.flatnonzero(original % 100 == 99)
    if not inject_cells.size or not produce_cells.size:
        raise ValueError("active selection must retain both I=1 and I=100 wells")
    wells = add_well(
        None, grid, rock, inject_cells, well_type="rate", val=6.97/DAY,
        radius=0.5*FT, well_dir="z", compi=np.array([0.0, 1.0]),
        sign=1, name="I", ref_depth=0.0,
        gravity=np.array([0.0, 0.0, NORM_GRAVITY]),
    )
    wells = add_well(
        wells, grid, rock, produce_cells, well_type="bhp", val=95.0*PSIA,
        radius=0.5*FT, well_dir="z", compi=np.array([1.0, 1.0]),
        sign=-1, name="P", ref_depth=0.0,
        gravity=np.array([0.0, 0.0, NORM_GRAVITY]),
    )
    datum = 100.0*PSIA
    fluid = _fluid(dataset, datum)
    # MRST's own tenthCSP_Model_I.m never overrides GenericBlackOilModelAD's
    # tolerances, so it runs at ReservoirModel.m's toleranceCNV=1e-3,
    # toleranceMB=1e-7 defaults. This used to pass tol_cnv=1e-7, tol_mb=1e-9,
    # tol_well=1e-9 - a hundred times tighter than the case it is a port of,
    # not a deliberately harder target - which cost real, unnecessary Newton
    # iterations (measured: 784 vs 609 for the equivalent MRST run) without
    # being asked for by anything the MRST script does. Leaving the
    # tolerances unset here falls through to GenericBlackOilModelAD's own
    # defaults, which already match MRST's (tol_cnv=1e-3, tol_mb=1e-7).
    model = GenericBlackOilModelAD(grid, rock, fluid, wells, gravity=NORM_GRAVITY)
    pressure = datum + 700.0*NORM_GRAVITY*grid.cells.centroids[:, 2]
    state0 = State(
        pressure=pressure, sW=np.ones(grid.cells.num),
        qWs=np.array([0.0, 0.0]), qOs=np.array([6.97/DAY, 0.0]),
        bhp=np.array([pressure[inject_cells].mean(), 95.0*PSIA]),
    )
    schedule = simple_schedule(_timesteps(), W=wells)
    return dataset, model, schedule, state0


def run_tenth_csp_model_i_ad(nsteps=None, *, layers=None, active=None,
                          data_root=None, verbose=True):
    """Run the complete ten-year benchmark or an initial report subset."""
    dataset, model, schedule, state0 = setup_tenth_csp_model_i_ad(
        layers=layers, active=active, data_root=data_root
    )
    count = schedule.n_steps if nsteps is None else min(int(nsteps), schedule.n_steps)
    solver = NonLinearSolver(max_iterations=30, max_timestep_cuts=8,
                             use_line_search=True, use_relaxation=True)
    report, states, state = SimulationReport(), [], state0.copy()
    for step_index in range(count):
        if verbose:
            print(f"SPE10 Model I step {step_index+1}/{count}", flush=True)
        state, step = solver.solve_timestep(
            model, state, float(schedule.step_val[step_index])
        )
        report.steps.append(step)
        if not step.converged:
            raise RuntimeError(f"SPE10 Model I step {step_index+1} failed: {step}")
        states.append(state.copy())
    history = {
        "time": np.cumsum(schedule.step_val[:count]),
        "pressure": np.vstack([s["pressure"] for s in states]) if states else np.empty((0, model.G.cells.num)),
        "sO": np.vstack([s["sW"] for s in states]) if states else np.empty((0, model.G.cells.num)),
        "sG": np.vstack([1.0-s["sW"] for s in states]) if states else np.empty((0, model.G.cells.num)),
        "qOs": np.vstack([s["qWs"] for s in states]) if states else np.empty((0, 2)),
        "qGs": np.vstack([s["qOs"] for s in states]) if states else np.empty((0, 2)),
        "bhp": np.vstack([s["bhp"] for s in states]) if states else np.empty((0, 2)),
    }
    return SPE10Model1Result(
        dataset, model, schedule, state0, states, report, history
    )



__all__ = ["SPE10Model1Result", "TIMESTEP_DAYS", "run_tenth_csp_model_i_ad",
           "setup_tenth_csp_model_i_ad", "tenthCSP_Model_I", "tenth_csp_model_i"]


if __name__ == "__main__":
    def test_setup_is_mrsts():
        out = tenth_csp_model_i(plot=False, nsteps=2)
        assert out.G.cells.num == 2000 and len(out.W) == 2
        assert out.model.oil and out.model.gas and not out.model.water
        assert len(out.states) == 2
        np.testing.assert_allclose(TIMESTEP_DAYS.sum(), 3650.0)
        print("ok test_setup_is_mrsts")

    test_setup_is_mrsts()
    print("\nAll tests passed!")
