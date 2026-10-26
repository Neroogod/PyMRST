"""Validate MRST ``LinearizedProblem``: the port in
`ad_core/solvers/linearized_problem.py`.

The reference is MRST 2024a (`dump_ad_core_linearized_problem_reference.m`):
the first Newton system of a 4x3x2 two-well oil-water GenericBlackOilModel
with gravity. On the port's own equations for the same step (which match
MRST's to round-off) the validator checks the assembled `A` and `b`
(MRST's `b = -residual`), the equation norms, a reordering, the elimination
of a well equation, the reduction to the cell equations, the full increment
recovered from the reduced solve, and `popIncrement`.
"""

from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.io import loadmat

HERE = Path(__file__).resolve().parent
RTOL = 1e-12


def report(ok: bool, text: str) -> bool:
    print(f"{'ok  ' if ok else 'FAIL'} {text}")
    return ok


def _close(name: str, a, b, tol: float = RTOL) -> bool:
    a = a.toarray() if sp.issparse(a) else np.asarray(a, dtype=float)
    b = b.toarray() if sp.issparse(b) else np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return report(False, f"{name}: shape {a.shape} vs MRST {b.shape}")
    err = float(np.abs(a - b).max()/max(np.abs(b).max(), 1e-300))
    return report(err < tol, f"{name} rel. diff {err:.1e}")


def build_problem(return_model: bool = False):
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_blackoil.models import GenericBlackOilModel
    from pymrst.modules.ad_core.utils.init_well_sol_ad import init_well_sol_ad
    from pymrst.modules.ad_core.solvers.linearized_problem import LinearizedProblem
    from pymrst.modules.ad_props import init_simple_adi_fluid
    from pymrst.params import make_rock
    from pymrst.params.gravity.gravity import gravity
    from pymrst.params.wells_and_bc import add_well
    from pymrst.solvers.init_res_sol import init_res_sol
    from pymrst.utils.units import BARSA, CENTI, DARCY, DAY, MILLI, POISE

    gravity("reset", "on")
    G = compute_geometry(cart_grid([4, 3, 2], [400.0, 300.0, 20.0]))
    rock = make_rock(G, 100*MILLI*DARCY, 0.25)
    fluid = init_simple_adi_fluid(phases="WO", mu=np.array([1, 4])*CENTI*POISE, rho=[1000, 800],
                                  n=[2, 2], c=np.array([1e-9, 1e-8])/BARSA)
    model = GenericBlackOilModel(G, rock, fluid, gas=False)
    W = add_well(None, G, rock, np.array([0, 12]), well_type="rate", val=50/DAY,
                 compi=np.array([1.0, 0.0]), name="I")
    W = add_well(W, G, rock, np.array([11, 23]), well_type="bhp", val=150*BARSA,
                 compi=np.array([1.0, 0.0]), name="P")
    state0 = init_res_sol(G, 200*BARSA, np.array([0.2, 0.8]))
    state0 = model.validate_state(state0)
    state0["wellSol"] = init_well_sol_ad(W, model, state0)
    dt = 10*DAY
    forces = model.get_valid_driving_forces()
    forces["W"] = W
    model = model.validate_model(forces)
    state = model.validate_state(state0)
    model, state = model.prepare_timestep(state, state0, dt, forces)
    p, _ = model.get_equations(state0, state, dt, forces, iteration=1)
    problem = LinearizedProblem(list(p.equations), list(p.types), list(p.equation_names),
                                list(p.primary_variables), p.state, p.dt)
    return (problem, model) if return_model else problem


def main() -> int:
    from pymrst.params.gravity.gravity import isolated_gravity
    ref = loadmat(HERE/"ref_ad_core_linearized_problem.mat", squeeze_me=True,
                  struct_as_record=False)
    with isolated_gravity():
        problem = build_problem()
    info = ref["info"]
    ok = report(problem.equationNames == [str(n) for n in info.names]
                and problem.types == [str(t) for t in info.types]
                and problem.primaryVariables == [str(v) for v in info.vars],
                f"{problem.numeq()} equations, their names, types and primary variables as MRST's")
    A, b = problem.get_linear_system()
    ok &= report(problem.A is None, "assembling leaves the caller's problem untouched")
    ok &= _close("A", A, ref["A"])
    ok &= _close("b = -residual", b, ref["b"])
    ok &= _close("norm", problem.norm(), info.norm2)
    ok &= _close("norm(inf)", problem.norm(np.inf), info.normInf)
    perm = np.atleast_1d(ref["perm"]) - 1
    Ar, br = problem.reorder_equations(perm).get_linear_system()
    ok &= _close("reordered A", Ar, ref["Ar"])
    ok &= _close("reordered b", br, ref["br"])
    pe, _ = problem.eliminate_variable(str(ref["wellName"]))
    Ae, be = pe.get_linear_system()
    ok &= report(problem.numeq() == len(info.names), "eliminating leaves the caller's problem untouched")
    ok &= _close(f"A with {ref['wellName']} eliminated", Ae, ref["Ae"])
    ok &= _close(f"b with {ref['wellName']} eliminated", be, ref["be"])
    pc, eliminated = problem.reduce_to_single_variable_type("cell")
    Ac, bc = pc.get_linear_system()
    ok &= _close("cell-reduced A", Ac, ref["Ac"])
    ok &= _close("cell-reduced b", bc, ref["bc"])
    xc = spla.spsolve(sp.csc_matrix(Ac), bc)
    dx = pc.recover_from_single_variable_type(problem, pc.store_increments(xc), eliminated)
    ok &= _close("full increment recovered from the reduced solve",
                 np.concatenate([np.ravel(d) for d in dx]), ref["dxfull"], 1e-10)
    dvar, rest, popped = problem.pop_increment(dx, problem.primaryVariables[0], 3)
    ok &= _close("popIncrement", dvar, ref["dvar"], 1e-10)
    ok &= report(popped.primaryVariables == problem.primaryVariables[1:] and len(rest) == len(dx) - 1,
                 "popIncrement drops the variable from the increments and the problem")
    print("\nLinearizedProblem matches MRST" if ok else "\nLinearizedProblem does NOT match MRST")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
