"""Run the AD-OO simulateSPE9 port, as published and with a direct linear
solver, and save each run's per-step arrays for
`validate_ad_oo_simulate_spe9.py`. Both runs take tens of minutes."""

from pathlib import Path
import sys

import numpy as np

from pymrst.modules.ad_blackoil.examples.spe9.simulate_spe9 import simulate_spe9
from pymrst.modules.ad_blackoil.examples.utils.local_dataset_path import local_dataset_path
from pymrst.modules.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad
from pymrst.modules.ad_core.simulators.sim_runner.get_packed_simulator_output import (
    get_packed_simulator_output,
)
from pymrst.modules.ad_core.simulators.sim_runner.pack_simulation_problem import (
    pack_simulation_problem,
)
from pymrst.modules.ad_core.simulators.sim_runner.simulate_packed_problem import (
    simulate_packed_problem,
)
from pymrst.modules.ad_core.solvers.concrete_solvers import BackslashSolverAD
from pymrst.modules.ad_core.utils.well_sol_to_vector import well_sol_to_vector

HERE = Path(__file__).resolve().parent


def save(problem, name):
    ws, states, reports, _ = get_packed_simulator_output(problem)
    qWs, qOs, qGs, bhp = well_sol_to_vector(ws)
    np.savez(HERE/f"py_simulate_spe9_{name}.npz",
             p=np.array([s["pressure"] for s in states]),
             rs=np.array([s["rs"] for s in states]),
             s=np.array([np.asarray(s["s"]).ravel(order="F") for s in states]),
             qWs=qWs, qOs=qOs, qGs=qGs, bhp=bhp,
             its=np.array([r["Iterations"] for r in reports]),
             nmini=np.array([len(r["StepReports"]) for r in reports]),
             types=np.array([",".join(w.type for w in step) for step in ws]),
             names=np.array([w.name for w in ws[0]]),
             lsolver=type(problem.SimulatorSetup.NonLinearSolver.LinearSolver).__name__,
             backend=type(problem.SimulatorSetup.model.AutoDiffBackend).__name__,
             basename=problem.BaseName)


def main(variants):
    if "published" in variants:
        save(simulate_spe9(plot=False), "published")
    if "direct" in variants:
        fn = local_dataset_path("spe9", "BENCH_SPE9.DATA")/"BENCH_SPE9.DATA"
        state0, model, schedule, nls = init_eclipse_problem_ad(fn, useMex=True, rowMajorAD=True)
        nls.LinearSolver = BackslashSolverAD()
        problem = pack_simulation_problem(state0, model, schedule, "SPE9_direct",
                                          NonLinearSolver=nls)
        simulate_packed_problem(problem, restartStep=1)
        save(problem, "direct")


if __name__ == "__main__":
    main(sys.argv[1:] or ["direct", "published"])
