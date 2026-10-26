"""MRST ``ad-core/examples/ensemblePackedProblemsExample.m``.

An ensemble of `n` (50) realizations of a 1D waterflood: 100 cells over
1000 m, porosity from `gaussianField` (between 0.01 and 0.5, an 11-cell kernel
of standard deviation 2.5) and permeability from it by Carman-Kozeny,
`K = poro^3 (1e-5)^2 / (0.81*72*(1 - poro)^2)`. Water enters through the
left boundary at half the bulk volume over five years, the right boundary is
held at 100 bar, and the fluid is slightly compressible (with rock
compressibility). Each realization is a packed problem under the base name
'EnsembleExample1D', named 'Case <i>' and described by its mean porosity and
permeability; a `PackedProblemManager` simulates them in background sessions
(`simulateProblemsBatch`), and the results are read back together.

With `plot`, MRST's figures: porosity and permeability of the first four
realizations, the saturation profiles step by step with their mean, the water
saturation at the outlet over time, and a histogram of the front positions.

MRST draws the porosity fields from MATLAB's `rng(0)`; Python's generator
gives different draws, so `poro` (cells x realizations) can be passed to use
given fields. `simulate='here'` runs the problems in this session
(`simulateProblem`) instead of in background sessions, which in PyMRST need
`cloudpickle` to store the models.

`ensemble_packed_problems_example_ad` (in `packed_problem_examples.py`) is
PyMRST's earlier surrogate, kept for its callers.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Optional

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil.models.generic_black_oil_model import GenericBlackOilModel
from pymrst.modules.ad_core.simulators.sim_runner.get_multiple_packed_simulator_outputs import (
    get_multiple_packed_simulator_outputs,
)
from pymrst.modules.ad_core.simulators.sim_runner.pack_simulation_problem import (
    pack_simulation_problem,
)
from pymrst.modules.ad_core.simulators.sim_runner.packed_problem_manager import (
    PackedProblemManager,
)
from pymrst.modules.ad_core.timesteps.rampup_timesteps import rampup_timesteps
from pymrst.modules.ad_core.utils.simple_schedule import simple_schedule
from pymrst.modules.ad_props import init_simple_adi_fluid
from pymrst.params.rock import make_rock
from pymrst.params.rock.gaussian_field import gaussian_field
from pymrst.params.wells_and_bc import fluxside, pside
from pymrst.solvers.init_res_sol import init_res_sol
from pymrst.utils.units import BARSA, CENTI, DARCY, MILLI, POISE, YEAR

BASE_NAME = "EnsembleExample1D"
FRONT_TOL = 0.01


def ensemble_packed_problems_example(n: int = 50, poro: Optional[np.ndarray] = None,
                                     plot: bool = False,
                                     simulate: str = "batch") -> SimpleNamespace:
    """MRST `ensemblePackedProblemsExample`: the problems and their results."""
    setup = setup_ensemble(n, poro)
    ppm = PackedProblemManager(setup.problems)
    if simulate == "batch":
        ppm.simulate_problems_batch()
    elif simulate == "here":
        ppm.simulate_problem()
    else:
        raise ValueError(f"Unknown simulate option {simulate}")
    ws, states, reports, names = get_multiple_packed_simulator_outputs(setup.problems)[:4]
    out = SimpleNamespace(**vars(setup), ppm=ppm, ws=ws, states=states, reports=reports,
                          names=names)
    if plot:
        _plot(out)
    return out


def setup_ensemble(n: int = 50, poro: Optional[np.ndarray] = None) -> SimpleNamespace:
    """The grid, fields, schedule and packed problems of the example."""
    G = compute_geometry(cart_grid([100], [1000.0]))
    time = 5*YEAR
    irate = np.sum(G.cells.volumes)*0.5/time
    if poro is None:
        rng = np.random.default_rng(0)
        poro = np.column_stack([np.reshape(gaussian_field(G.cartDims, [0.01, 0.5], [11, 3], 2.5,
                                                          rng=rng), -1)
                                for _ in range(n)])
    poro = np.asarray(poro, dtype=float)
    K = poro**3*(1e-5)**2/(0.81*72*(1 - poro)**2)
    bc = fluxside(None, G, "XMin", irate, sat=np.array([1.0, 0.0]))
    bc = pside(bc, G, "XMax", 100*BARSA, sat=np.array([1.0, 0.0]))
    dt = rampup_timesteps(time, time/100)
    schedule = simple_schedule(dt, bc=bc)
    fluid = init_simple_adi_fluid(mu=[1*CENTI*POISE, 5*CENTI*POISE], n=[2, 2], phases="wo",
                                  c_r=1e-8/BARSA, c=[1e-5/BARSA, 1e-4/BARSA])
    state0 = init_res_sol(G, 100*BARSA, np.array([0.0, 1.0]))
    problems = []
    for i in range(poro.shape[1]):
        k, p = K[:, i], poro[:, i]
        rock = make_rock(G, k, p)
        model = GenericBlackOilModel(G, rock, fluid, gas=False)
        description = (f"Average porosity {np.mean(p):1.2f}, "
                       f"average perm {np.mean(k)/(MILLI*DARCY):1.2f} md")
        problems.append(pack_simulation_problem(state0, model, schedule, BASE_NAME,
                                                Name=f"Case {i + 1}",
                                                Description=description))
    return SimpleNamespace(G=G, poro=poro, K=K, schedule=schedule, problems=problems)


def front_positions(states_at_step: list, nc: int) -> tuple:
    """Per realization, the last cell with water saturation at least
    `FRONT_TOL` (1-based, 0 if none), and the smallest and largest of them."""
    positions = np.zeros(len(states_at_step), dtype=int)
    mv, Mv = nc, 0
    for i, state in enumerate(states_at_step):
        s = np.asarray(state["s"])[:, 0]
        found = np.flatnonzero(s >= FRONT_TOL)
        if found.size:
            positions[i] = found[-1] + 1
            Mv = max(Mv, positions[i])
            mv = min(mv, positions[i])
    return positions, mv, Mv


def _plot(out: SimpleNamespace) -> None:
    import matplotlib.pyplot as plt
    x = out.G.cells.centroids[:, 0]
    subs = range(min(4, out.poro.shape[1]))
    labels = [f"Realization {i + 1}" for i in subs]
    fig, ax = plt.subplots()
    ax.plot(x, out.poro[:, list(subs)])
    ax.set_ylim(0, 0.6)
    ax.set_ylabel("Porosity")
    ax.set_xlabel("Position")
    ax.legend(labels, loc="upper center", ncol=len(labels))
    ax2 = ax.twinx()
    ax2.plot(x, out.K[:, list(subs)]/(MILLI*DARCY), "--")
    ax2.set_ylabel("Permeability [md]")
    np_ = len(out.states)
    colors = plt.cm.viridis(np.linspace(0, 1, np_))
    nstep = out.schedule.step_val.size
    fig, ax = plt.subplots()
    for step in range(nstep):
        ax.clear()
        avg = 0
        for i in range(np_):
            s = np.asarray(out.states[i][step]["s"])[:, 0]
            ax.plot(x, s, color=colors[i], alpha=0.4, linewidth=2)
            avg = avg + s
        ax.plot(x, avg/np_, "k", linewidth=2)
        ax.set_ylabel("S_w")
        plt.pause(0.001)
    fig, ax = plt.subplots()
    for i in range(np_):
        ax.plot([np.asarray(st["s"])[-1, 0] for st in out.states[i]], color=colors[i],
                alpha=0.4, linewidth=2)
    ax.set_title("Water breakthrough")
    bins = np.linspace(1, out.G.cells.num + 1, 25)
    fig, ax = plt.subplots()
    for step in range(nstep):
        positions, mv, Mv = front_positions([st[step] for st in out.states], out.G.cells.num)
        ax.clear()
        ax.hist(positions - 0.5, bins)
        ax.set_ylim(0, 10)
        ax.set_title(f"Step {step + 1}: Front between {mv} and {Mv}")
        plt.pause(0.5)


ensemblePackedProblemsExample = ensemble_packed_problems_example

__all__ = ["ensemblePackedProblemsExample", "ensemble_packed_problems_example",
           "front_positions", "setup_ensemble"]


if __name__ == "__main__":
    import contextlib
    import io

    from pymrst.modules.ad_core.simulators.sim_runner.clear_packed_simulator_output import (
        clear_packed_simulator_output,
    )

    def test_three_realizations_simulate_here():
        setup = setup_ensemble(3)
        assert setup.poro.shape == (100, 3)
        assert np.isclose(setup.poro.min(), 0.01) and np.isclose(setup.poro.max(), 0.5)
        assert setup.problems[2].Name == "Case 3"
        assert setup.problems[0].Description.startswith("Average porosity 0.")
        with contextlib.redirect_stdout(io.StringIO()):
            clear_packed_simulator_output(setup.problems, Prompt=False)
            out = ensemble_packed_problems_example(3, simulate="here")
            clear_packed_simulator_output(out.problems, Prompt=False)
        nstep = out.schedule.step_val.size
        assert [len(s) for s in out.states] == [nstep]*3
        positions, mv, Mv = front_positions([s[-1] for s in out.states], 100)
        assert 0 < mv <= Mv <= 100
        print("ok test_three_realizations_simulate_here")

    test_three_realizations_simulate_here()
    print("\nAll tests passed!")
