"""Run the blackoilTutorialSPE9 port without its figures and save what
`validate_ad_oo_blackoil_tutorial_spe9.py` compares: the tutorial's CPR
solver on the first linearised system, and the tutorial run with a direct
linear solver (as `dump_blackoil_tutorial_spe9_reference.m` does; the CPR
run gives up in step 1 in MRST and in the port), and report step 18 on
MRST's ministeps (see the validator on why). Takes several minutes."""

from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from pymrst.modules.ad_blackoil.examples.spe9.blackoil_tutorial_spe9 import (
    blackoil_tutorial_spe9,
)
from pymrst.modules.ad_core.utils.well_sol_to_vector import well_sol_to_vector
from pymrst.params.gravity.gravity import isolated_gravity

HERE = Path(__file__).resolve().parent


def first_system_cpr() -> dict:
    """The tutorial's CPR on the first linearised system (one-day step from
    the initial state), and the direct solution of the same system."""
    import scipy.sparse.linalg as spla
    from pymrst.modules.ad_core.utils.convert_deck_schedule_to_mrst import (
        convert_deck_schedule_to_mrst,
    )
    from pymrst.modules.ad_core.utils.select_linear_solver_ad import select_linear_solver_ad
    from pymrst.modules.ad_core.utils.select_model_from_deck import select_model_from_deck
    from pymrst.modules.test_suite.dataset_functions.spe9 import setup_spe9
    from pymrst.utils.units import DAY
    G, rock, fluid, deck, state0 = setup_spe9()
    model = select_model_from_deck(G, rock, fluid, deck)
    schedule = convert_deck_schedule_to_mrst(model, deck)
    model.drsMaxRel, model.dpMaxRel, model.dsMaxAbs = np.inf, .1, .1
    linsolve = select_linear_solver_ad(model)
    forces = model.get_driving_forces(schedule.controls[0])
    model = model.validate_model(forces)
    st0 = model.validate_state(state0)
    model, st0 = model.update_for_changed_controls(st0, forces)
    model, st0 = model.prepare_report_step(st0, st0, DAY, forces)
    model, st = model.prepare_timestep(st0, st0, DAY, forces)
    problem, _ = model.get_equations(st0, st, DAY, forces, iteration=1)
    _, cpr_dx, cpr_report = linsolve.solve_linear_problem(problem, model)
    problem = problem.assemble_system()
    direct_dx = spla.spsolve(problem.A.tocsc(), problem.b)      # MRST's b = -r: the increment
    return dict(cpr_dx=cpr_dx, cpr_residual=cpr_report["Residual"],
                cpr_iterations=cpr_report["Iterations"], direct_dx=direct_dx)


def knife_edge_step(out: SimpleNamespace, step: int = 18) -> dict:
    """Report step `step` from the run's own previous state, on MRST's
    ministeps: MRST's 20-day attempt fails and is cut to two halves."""
    from pymrst.modules.ad_core.ad_oo.nonlinear_solver import NonLinearSolver
    from pymrst.modules.ad_core.ad_oo.physical_model import copy_state
    from pymrst.modules.ad_core.ad_oo.wrapper_model import value_copy_model
    from pymrst.modules.ad_core.solvers.concrete_solvers import BackslashSolverAD
    k = step - 1
    control = int(out.schedule.step_control[k])
    forces = out.model.get_driving_forces(out.schedule.controls[control])
    model = value_copy_model(out.model).validate_model(forces)
    state = model.validate_state(copy_state(out.states[k - 1]))
    nls = NonLinearSolver(LinearSolver=BackslashSolverAD())
    its = 0
    for _ in range(2):
        state, report, _ = nls.solve_timestep(state, float(out.schedule.step_val[k])/2, model,
                                              **forces, controlId=control)
        its += report["Iterations"]
    return dict(knife_p=state["pressure"], knife_rs=state["rs"],
                knife_sat=np.asarray(state["s"]).ravel(order="F"),
                knife_bhp=np.array([w.bhp for w in state["wellSol"]]), knife_its=its)


def main() -> None:
    from pymrst.modules.ad_core.solvers.concrete_solvers import BackslashSolverAD
    with isolated_gravity():
        first = first_system_cpr()
        timer = perf_counter()
        out = blackoil_tutorial_spe9(plot=False, LinearSolver=BackslashSolverAD())
        walltime = perf_counter() - timer
        knife = knife_edge_step(out)
    qWs, qOs, qGs, bhp = well_sol_to_vector(out.wellsols)
    reports = out.reports["ControlstepReports"]
    np.savez(HERE/"py_blackoil_tutorial_spe9.npz",
             lsolver=type(out.linsolve).__name__,
             psolver=type(out.linsolve.ellipticSolver).__name__,
             column=out.column, Rs_frac=out.Rs_frac, sgn=out.sgn, P=out.P, kr=out.kr,
             krO=out.krO, pc=out.pc, pvt=out.pvt, Bo=out.Bo, muO=out.muO,
             saturated=out.saturated,
             p=np.array([s["pressure"] for s in out.states]),
             rs_states=np.array([s["rs"] for s in out.states]),
             sat=np.array([np.asarray(s["s"]).ravel(order="F") for s in out.states]),
             qWs=qWs, qOs=qOs, qGs=qGs, bhp=bhp,
             its=np.array([r["Iterations"] for r in reports]),
             nmini=np.array([len(r["StepReports"]) for r in reports]),
             types=np.array([",".join(w.type for w in ws) for ws in out.wellsols]),
             names=np.array([w.name for w in out.wellsols[0]]),
             Tcomp=out.Tcomp, T=out.T, bhp_comp=out.bhp_comp, gas_comp=out.gas_comp,
             bhp_mrst=out.bhp_mrst, gas_mrst=out.gas_mrst, ctrls=out.ctrls,
             walltime=walltime, **first, **knife)
    print(f"saved after {walltime:.0f} s")


if __name__ == "__main__":
    main()
