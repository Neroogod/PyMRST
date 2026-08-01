"""Executable PyMRST port of MRST ``spe9/simulateSPE9.m``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import copy
import numpy as np

from pymrst.gridprocessing import compute_geometry, process_grdecl
from pymrst.modules.ad_blackoil.models.generic_black_oil_model import GenericBlackOilModel, initial_state_from_deck
from pymrst.modules.ad_core import NonLinearSolver, SimulationReport, State
from pymrst.modules.ad_core.utils.convert_deck_schedule_to_mrst import convert_deck_schedule_to_mrst
from pymrst.modules.ad_props import init_deck_blackoil_fluid
from pymrst.modules.deckformat import read_spe9_deck
from pymrst.params import NORM_GRAVITY, make_rock
from pymrst.utils.units import DAY

ROOT = Path(__file__).resolve().parents[6]
DEFAULT_DECK = ROOT/"mrst-2024a-Geomech/examples/data/SPE9/BENCH_SPE9.DATA"


@dataclass
class SPE9Result:
    deck: dict
    model: GenericBlackOilModel
    schedule: object
    initial_state: State
    states: list[State]
    report: SimulationReport
    summary: dict[str, np.ndarray]


def setup_spe9(deck_path=None):
    """Import and assemble the 9000-cell, 26-well SPE9 problem."""
    deck = read_spe9_deck(deck_path or DEFAULT_DECK)
    grid = compute_geometry(process_grdecl(deck["grdecl"]))
    ix = np.asarray(getattr(grid.cells, "indexMap", np.arange(grid.cells.num)), int)
    rock = make_rock(grid, deck["perm"][ix], deck["poro"][ix, None])
    fluid = init_deck_blackoil_fluid(deck)
    model = GenericBlackOilModel(
        grid, rock, fluid, [], disgas=True, gravity=NORM_GRAVITY,
        dp_max_rel=0.1, ds_max=0.1, tol_cnv=5e-3, tol_mb=5e-4,
        tol_well=1e-5,
    )
    schedule = convert_deck_schedule_to_mrst(model, deck["schedule"])
    model.facility.wells = copy.deepcopy(schedule.controls[0].W)
    state_deck = dict(deck)
    for key in ("pressure", "swat", "sgas", "rs", "rv"):
        state_deck[key] = np.asarray(deck[key])[ix]
    state0 = initial_state_from_deck(model, state_deck)
    return deck, model, schedule, state0


def _summary(states, times, names):
    if not states:
        return {"time": np.zeros(0)}
    qws = np.vstack([s["qWs"] for s in states])
    qos = np.vstack([s["qOs"] for s in states])
    qgs = np.vstack([s["qGs"] for s in states])
    bhp = np.vstack([s["bhp"] for s in states])
    out = {"time": np.asarray(times), "FOPR": -qos.clip(max=0).sum(axis=1),
           "FGPR": -qgs.clip(max=0).sum(axis=1), "FWIR": qws.clip(min=0).sum(axis=1)}
    for i, name in enumerate(names):
        out[f"{name}:WBHP"] = bhp[:, i]
        out[f"{name}:WOPR"] = -np.minimum(qos[:, i], 0.0)
        out[f"{name}:WGPR"] = -np.minimum(qgs[:, i], 0.0)
        out[f"{name}:WWIR"] = np.maximum(qws[:, i], 0.0)
    return out


def simulate_spe9(nsteps=None, *, deck_path=None, verbose=True):
    """Run the full 35-step/900-day schedule or an initial subset."""
    deck, model, schedule, state0 = setup_spe9(deck_path)
    count = schedule.n_steps if nsteps is None else min(int(nsteps), schedule.n_steps)
    solver = NonLinearSolver(max_iterations=35, max_timestep_cuts=8,
                             use_line_search=False, use_relaxation=True,
                             relaxation_min=0.2)
    report, states, times = SimulationReport(), [], []
    state, elapsed, active = state0.copy(), 0.0, None
    for i in range(count):
        control_index = int(schedule.step_control[i])
        if control_index != active:
            model.facility.wells = copy.deepcopy(schedule.controls[control_index].W)
            model.facility.within_limits = True
            active = control_index
        if verbose:
            print(f"SPE9 step {i+1:02d}/{count:02d} ({elapsed/DAY:.0f} days)", flush=True)
        state, step = solver.solve_timestep(model, state, float(schedule.step_val[i]))
        report.steps.append(step)
        if not step.converged:
            raise RuntimeError(f"SPE9 step {i+1} failed: {step}")
        elapsed += float(schedule.step_val[i]); times.append(elapsed); states.append(state.copy())
    return SPE9Result(deck, model, schedule, state0, states, report,
                      _summary(states, times, [w.name for w in model.wells]))


if __name__ == "__main__":
    result = simulate_spe9()
    print(f"Completed {len(result.states)} SPE9 steps")
