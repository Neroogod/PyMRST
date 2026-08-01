"""Full two-phase oil-water EGG benchmark workflow."""

from __future__ import annotations

from dataclasses import dataclass
import copy
from pathlib import Path
import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry, remove_cells
from pymrst.modules.ad_blackoil.models import GenericBlackOilModel
from pymrst.modules.ad_core import NonLinearSolver, SimulationReport, State
from pymrst.modules.ad_core.utils.convert_deck_schedule_to_mrst import convert_deck_schedule_to_mrst
from pymrst.modules.ad_props import ADFluid, interp_table
from pymrst.modules.deckformat import read_egg_deck
from pymrst.params import NORM_GRAVITY, make_rock
from pymrst.utils.units import BARSA, DAY


@dataclass
class EGGResult:
    deck: dict
    model: GenericBlackOilModel
    schedule: object
    initial_state: State
    states: list[State]
    report: SimulationReport
    history: dict[str, np.ndarray]


def compare_egg_production_history(result: EGGResult, reference) -> dict[str, float]:
    """Return maximum relative well-history errors against an MRST fixture."""
    from scipy.io import loadmat

    data = loadmat(Path(reference), squeeze_me=True)
    aliases = {"qWs": ("qWs_history", "qWs1"),
               "qOs": ("qOs_history", "qOs1"),
               "bhp": ("bhp_history", "bhp1")}
    errors = {}
    for name, keys in aliases.items():
        key = next((candidate for candidate in keys if candidate in data), None)
        if key is None:
            raise KeyError(f"EGG reference has no {keys[0]} or {keys[1]}")
        actual = np.atleast_2d(np.asarray(result.history[name], float))
        expected = np.atleast_2d(np.asarray(data[key], float))
        if expected.shape != actual.shape:
            raise ValueError(
                f"{name} history shape differs: PyMRST {actual.shape}, "
                f"MRST {expected.shape}"
            )
        floor = 1e-7 if name.startswith("q") else 1e5
        errors[name] = float(np.max(
            np.abs(actual-expected)/np.maximum(np.abs(expected), floor)
        ))
    return errors


def _compressible_b(table):
    def b(p):
        x = table["c"]*(p-table["p_ref"])
        return (1.0+x+0.5*x*x)/table["b_ref"]
    return b


def _egg_fluid(deck):
    sw = deck["swof"][:, 0]
    so = 1.0-sw[::-1]
    return ADFluid(
        rhoS=deck["rhoS"],
        mu=np.array([deck["pvtw"]["mu_ref"], deck["pvcd"]["mu_ref"]]),
        b_w=_compressible_b(deck["pvtw"]), b_o=_compressible_b(deck["pvcd"]),
        kr_w=lambda saturation: interp_table(sw, deck["swof"][:, 1], saturation),
        kr_o=lambda saturation: interp_table(so, deck["swof"][::-1, 2], saturation),
    )


def setup_egg(*, realization=0, deck_path=None, data_root=None):
    """Build EGG grid, realization properties, fluid, wells and initial state."""
    deck = read_egg_deck(deck_path, realization=realization, data_root=data_root)
    nx, ny, nz = deck["cartDims"]
    grid = cart_grid([nx, ny, nz], [nx*8.0, ny*8.0, nz*4.0])
    grid.nodes.coords[:, 2] += 4000.0
    grid = remove_cells(grid, ~deck["actnum"])
    grid = compute_geometry(grid)
    ix = np.asarray(grid.cells.indexMap, dtype=int)
    rock = make_rock(grid, deck["perm"][ix], deck["poro"][ix, None])
    fluid = _egg_fluid(deck)
    model = GenericBlackOilModel(
        grid, rock, fluid, [], gravity=NORM_GRAVITY,
        tol_cnv=5e-3, tol_mb=5e-4, tol_well=1e-5,
    )
    schedule = convert_deck_schedule_to_mrst(model, deck["schedule"])
    model.facility.wells = copy.deepcopy(schedule.controls[0].W)

    p_ref = 400*BARSA
    rho_o = fluid.b_o(p_ref)*fluid.rhoS[1]
    # The deck's EQUIL datum is z=4000 m at 400 bar.  Cell centers begin at
    # 4002 m, so the first active layer is already hydrostatically above it.
    pressure = p_ref + NORM_GRAVITY*rho_o*(
        grid.cells.centroids[:, 2]-4000.0
    )
    nw = len(model.wells)
    qws, qos, bhp = np.zeros(nw), np.zeros(nw), np.zeros(nw)
    for i, well in enumerate(model.wells):
        if well.type == "rate":
            qws[i] = well.val*well.compi[0]
            qos[i] = well.val*well.compi[1]
            bhp[i] = pressure.mean()+10*BARSA
        else:
            bhp[i] = well.val
    state0 = State(pressure=pressure, sW=np.full(grid.cells.num, 0.1),
                   qWs=qws, qOs=qos, bhp=bhp)
    return deck, model, schedule, state0


def _history(states, names):
    if not states:
        return {"time": np.zeros(0)}
    out = {key: np.vstack([np.asarray(state[key], float) for state in states])
           for key in ("pressure", "sW", "qWs", "qOs", "bhp")}
    out["water_cut"] = np.abs(out["qWs"])/np.maximum(
        np.abs(out["qWs"]+out["qOs"]), 1e-30
    )
    for i, name in enumerate(names):
        out[f"{name}:bhp"] = out["bhp"][:, i]
        out[f"{name}:qWs"] = out["qWs"][:, i]
        out[f"{name}:qOs"] = out["qOs"][:, i]
    return out


def run_egg_example(nsteps=None, *, realization=0, deck_path=None,
                    data_root=None, verbose=True):
    """Run all 123 EGG report steps, or an initial subset."""
    deck, model, schedule, state0 = setup_egg(
        realization=realization, deck_path=deck_path, data_root=data_root
    )
    count = schedule.n_steps if nsteps is None else min(int(nsteps), schedule.n_steps)
    solver = NonLinearSolver(max_iterations=30, max_timestep_cuts=8,
                             use_line_search=True, use_relaxation=True,
                             relaxation_min=0.1)
    report, states, state = SimulationReport(), [], state0.copy()
    for i in range(count):
        if verbose:
            print(f"EGG step {i+1:03d}/{count:03d}", flush=True)
        state, step = solver.solve_timestep(model, state, float(schedule.step_val[i]))
        report.steps.append(step)
        if not step.converged:
            raise RuntimeError(f"EGG step {i+1} failed: {step}")
        states.append(state.copy())
    history = _history(states, [well.name for well in model.wells])
    history["time"] = np.cumsum(schedule.step_val[:count])
    return EGGResult(deck, model, schedule, state0, states, report, history)


if __name__ == "__main__":
    result = run_egg_example()
    print(f"Completed {len(result.states)} EGG steps")
