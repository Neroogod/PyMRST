"""SPE10 Model I oil-gas benchmark setup and simulation."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry, remove_cells
from pymrst.modules.ad_blackoil.models import GenericBlackOilModel
from pymrst.modules.ad_core import NonLinearSolver, SimulationReport, State
from pymrst.modules.ad_core.utils.schedule import simple_schedule
from pymrst.modules.ad_props import ADFluid, interp_table
from pymrst.modules.deckformat import load_spe10_model1
from pymrst.params import NORM_GRAVITY, add_well, make_rock
from pymrst.utils.units import CENTI, DAY, FT, POISE, PSIA


@dataclass
class SPE10Model1Result:
    dataset: dict
    model: GenericBlackOilModel
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


def setup_tenth_csp_model_i(*, layers=None, active=None, data_root=None):
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
    model = GenericBlackOilModel(
        grid, rock, fluid, wells, gravity=NORM_GRAVITY,
        tol_cnv=1e-7, tol_mb=1e-9, tol_well=1e-9,
    )
    pressure = datum + 700.0*NORM_GRAVITY*grid.cells.centroids[:, 2]
    state0 = State(
        pressure=pressure, sW=np.ones(grid.cells.num),
        qWs=np.array([0.0, 0.0]), qOs=np.array([6.97/DAY, 0.0]),
        bhp=np.array([pressure[inject_cells].mean(), 95.0*PSIA]),
    )
    schedule = simple_schedule(_timesteps(), W=wells)
    return dataset, model, schedule, state0


def run_tenth_csp_model_i(nsteps=None, *, layers=None, active=None,
                          data_root=None, verbose=True):
    """Run the complete ten-year benchmark or an initial report subset."""
    dataset, model, schedule, state0 = setup_tenth_csp_model_i(
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


if __name__ == "__main__":
    result = run_tenth_csp_model_i()
    print(f"Completed {len(result.states)} SPE10 Model I steps")
