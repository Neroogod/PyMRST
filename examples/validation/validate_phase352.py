"""Phase 352 (AC15): the linear-solver class hierarchy.

Adds MRST's `LinearSolverAD` base + `LinearizedProblem` and the concrete
siblings (`BackslashSolverAD`, `NoOpSolverAD`, `GMRES_ILUSolverAD`) to PyMRST's
existing `CPRSolver` / `DirectSolver`. Proved on a real reservoir Newton
Jacobian - the one thing a linear solver must do is return the increment that
zeroes the linearized residual:

1. **Backslash is exact and matches the existing DirectSolver.** On a genuine
   black-oil Jacobian, `BackslashSolverAD` solves ``A dx = rhs`` to round-off
   and returns the same increment as PyMRST's validated `DirectSolver` - so it
   is a drop-in.
2. **GMRES+ILU converges to the same increment.** The iterative solver reaches
   the direct solution to its tolerance.
3. **NoOp returns zero**, and `LinearizedProblem` splits an increment into the
   model's named primary-variable blocks.

No `.mat`: the solvers are checked against the exact solution of a real
assembled system and against the already-validated direct solver.
"""

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
    GenericBlackOilModel,
)
from pymrst.modules.ad_core.solvers import (
    BackslashSolverAD,
    DirectSolver,
    GMRES_ILUSolverAD,
    LinearizedProblem,
    NoOpSolverAD,
)
from pymrst.modules.ad_core.statefunctions.state_function import State
from pymrst.modules.ad_props import init_simple_adi_fluid
from pymrst.params import add_well, make_rock
from pymrst.utils.adi import combine_equations
from pymrst.utils.units import BARSA, CENTI, DARCY, DAY, MILLI, POISE


def real_jacobian():
    """A genuine black-oil Newton system (A, rhs)."""
    g = compute_geometry(cart_grid([6, 5, 2], [300.0, 250.0, 20.0]))
    rock = make_rock(g, 100 * MILLI * DARCY, 0.25)
    fluid = init_simple_adi_fluid(
        mu=np.array([1.0, 5.0]) * CENTI * POISE, rho=[1000.0, 800.0],
        n=[2, 2], c=[1e-6 / BARSA, 1e-4 / BARSA], p_ref=200 * BARSA)
    w = add_well(None, g, rock, np.array([0]), well_type="rate", val=2e-3,
                 compi=np.array([1.0, 0.0]), sign=1)
    w = add_well(w, g, rock, np.array([g.cells.num - 1]), well_type="bhp",
                 val=150 * BARSA, compi=np.array([0.0, 1.0]), sign=-1)
    model = GenericBlackOilModel(g, rock, fluid, w)
    st = State(pressure=np.full(g.cells.num, 205 * BARSA),
               sW=np.full(g.cells.num, 0.45), qWs=np.array([2e-3, 0.0]),
               qOs=np.array([0.0, -2e-3]), bhp=np.array([150 * BARSA] * 2))
    eqs = model.get_equations(st, st.copy(), 30.0 * DAY)
    eq = combine_equations(*eqs)
    return eq.jac[0].tocsr(), -np.asarray(eq.val, float)


def main():
    print("Phase 352 (AC15): linear-solver hierarchy")
    A, rhs = real_jacobian()
    print(f"  real black-oil Jacobian: {A.shape[0]} unknowns")

    dx_direct = DirectSolver().solve(A, rhs)
    dx_back = BackslashSolverAD().solve(A, rhs)
    r_back = np.max(np.abs(A @ dx_back - rhs))
    d_dir = np.max(np.abs(dx_back - dx_direct))
    print(f"  BackslashSolverAD: residual {r_back:.2e}, vs DirectSolver "
          f"{d_dir:.2e}")
    assert r_back < 1e-8 and d_dir < 1e-10

    dx_gmres = GMRES_ILUSolverAD(tolerance=1e-11).solve(A, rhs)
    d_gmres = np.max(np.abs(dx_gmres - dx_direct))
    print(f"  GMRES_ILUSolverAD: matches direct to {d_gmres:.2e}")
    assert d_gmres < 1e-6

    assert np.allclose(NoOpSolverAD().solve(A, rhs), 0.0)
    prob = LinearizedProblem(A, -rhs, var_names=["all"], var_sizes=[rhs.size])
    assert np.allclose(prob.store_increments(dx_direct)["all"], dx_direct)
    print("  NoOpSolverAD returns zero; LinearizedProblem splits the increment")

    print("Phase 352 passed - the solver hierarchy solves a real reservoir "
          "system, matching the validated direct solver.")


if __name__ == "__main__":
    main()
