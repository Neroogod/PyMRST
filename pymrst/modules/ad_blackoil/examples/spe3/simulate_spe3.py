"""Full PyMRST port of MRST ``spe3/simulateSPE3.m``.

The official 9 x 9 x 4 wet-gas deck is imported directly.  SPE3 enables
vaporized oil without dissolved gas, starts with no free oil, injects gas for
15 report steps, shuts injection, and continues depletion for ten years.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pymrst.gridprocessing import compute_geometry, tensor_grid
from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
    GenericBlackOilModel, initial_state_from_deck,
)
from pymrst.modules.ad_core import NonLinearSolver, SimulationReport, State
from pymrst.modules.ad_props import init_deck_blackoil_fluid
from pymrst.modules.deckformat import read_spe3_deck
from pymrst.params import NORM_GRAVITY, add_well, make_rock

ROOT = Path(__file__).resolve().parents[6]
DEFAULT_DECK = ROOT / "mrst-2024a-Geomech/examples/data/SPE3/BENCH_SPE3.DATA"


@dataclass
class SPE3Result:
    deck: dict
    model: GenericBlackOilModel
    initial_state: State
    states: list[State]
    report: SimulationReport
    time: np.ndarray
    history: dict[str, np.ndarray]


def _make_wells(grid, rock, definitions):
    wells = None
    for definition in definitions.values():
        wells = add_well(
            wells, grid, rock, definition["cells"],
            well_type=definition["type"], val=definition["val"],
            radius=definition["radius"], compi=definition["compi"],
            ref_depth=definition["ref_depth"], sign=definition["sign"],
            name=definition["name"], lims=definition.get("lims"),
            gravity=np.array([0.0, 0.0, NORM_GRAVITY]),
        )
    return wells


def setup_spe3(deck_path: str | Path | None = None):
    """Import the SPE3 deck and construct grid, rock, wet-gas model and state."""
    deck = read_spe3_deck(deck_path or DEFAULT_DECK)
    grid = compute_geometry(tensor_grid(deck["x"], deck["y"], deck["z"]))
    rock = make_rock(grid, deck["perm"], deck["poro"][:, None])
    fluid = init_deck_blackoil_fluid(deck)
    wells = _make_wells(grid, rock, deck["controls"][0])
    model = GenericBlackOilModel(
        grid, rock, fluid, wells, disgas=False, vapoil=True,
        gravity=NORM_GRAVITY, tol_cnv=5e-3, tol_mb=5e-4, tol_well=1e-5,
    )
    return deck, model, initial_state_from_deck(model, deck)


def _apply_control(model, definitions):
    by_name = {w.name: w for w in model.wells}
    for name, definition in definitions.items():
        well = by_name[name]
        well.type = definition["type"]
        well.val = float(definition["val"])
        well.lims = dict(definition.get("lims") or {})
        well.status = True
    model.facility.within_limits = True


def _history(states):
    keys = ("pressure", "sW", "sO", "sG", "rs", "rv",
            "qWs", "qOs", "qGs", "bhp")
    return {key: np.vstack([np.asarray(st[key], float) for st in states])
            for key in keys}


def simulate_spe3(
    nsteps: int | None = None,
    *,
    deck_path: str | Path | None = None,
    verbose: bool = True,
) -> SPE3Result:
    """Run all 25 SPE3 report steps, or the first ``nsteps`` for a quick check."""
    deck, model, state0 = setup_spe3(deck_path)
    timesteps = deck["tsteps"] if nsteps is None else deck["tsteps"][:int(nsteps)]
    controls = deck["step_control"][:timesteps.size]
    solver = NonLinearSolver(
        max_iterations=60, use_line_search=True, use_relaxation=True,
        relaxation_min=0.1,
    )
    report = SimulationReport()
    state, states, elapsed = state0.copy(), [], []
    total, active = 0.0, None
    for index, (dt, control) in enumerate(zip(timesteps, controls), start=1):
        if int(control) != active:
            _apply_control(model, deck["controls"][int(control)])
            active = int(control)
        if verbose:
            print(f"SPE3 step {index:02d}/{len(timesteps):02d}", flush=True)
        state, step = solver.solve_timestep(model, state, float(dt))
        report.steps.append(step)
        if not step.converged:
            raise RuntimeError(f"SPE3 step {index} failed to converge: {step}")
        states.append(state.copy())
        total += float(dt)
        elapsed.append(total)
    return SPE3Result(deck, model, state0, states, report,
                      np.asarray(elapsed), _history(states))


if __name__ == "__main__":
    result = simulate_spe3()
    print(f"Completed {len(result.states)} SPE3 steps")
