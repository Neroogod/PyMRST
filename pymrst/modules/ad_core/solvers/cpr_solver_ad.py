"""MRST ``CPRSolverADAD`` compatibility surface for PyMRST's CPR solver."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from pymrst.modules.ad_core.solvers.linear_solver_ad import (
    LinearizedProblem,
    LinearizedProblemAD,
    LinearSolverAD,
    _assembled,
    _rhs,
)
from pymrst.modules.ad_core.solvers.linear_solvers import CPRSolver


class CPRSolverADAD(LinearSolverAD):
    """Two-stage CPR/GMRES solver using PyMRST's validated implementation."""

    def __init__(self, *, relative_tolerance=1e-4, tolerance=1e-8,
                 max_iterations=200, **options):
        super().__init__(
            tolerance=tolerance, max_iterations=max_iterations,
            verbose=bool(options.pop("verbose", False)),
        )
        self.relativeTolerance = float(relative_tolerance)
        self.backend = CPRSolver(
            rtol=tolerance, maxiter=max_iterations,
            **options,
        )

    @property
    def elliptic_solver(self):
        """The pressure-stage solver, read/written straight through to the
        backend so this always agrees with what the real solve path
        (`backend._elliptic_solver`) uses - assigning `cpr.elliptic_solver =
        AMGCLSolverAD()` after construction takes effect immediately, not
        just for `apply_two_stage_preconditioner`'s own direct calls."""
        return self.backend.elliptic_solver

    @elliptic_solver.setter
    def elliptic_solver(self, solver):
        self.backend.elliptic_solver = solver

    def solve_linear_system(self, A, b):
        # Without model/state metadata CPR cannot form its pressure weighting;
        # the backend deliberately performs an exact sparse direct solve.
        result = self.backend.solve(A, b)
        self.last_iterations = self.backend.last_iterations
        return result

    def solve(self, jac, rhs, model=None, state=None, dt=None, **kwargs):
        result = self.backend.solve(
            jac, rhs, model=model, state=state, dt=dt
        )
        self.last_iterations = self.backend.last_iterations
        return result

    def solve_linear_problem(self, problem: LinearizedProblem,
                             model=None, state=None):
        """`(dx, result, report)`, as MRST's `solveLinearProblem` returns.

        CPR needs the model and state to form its pressure weighting, which the
        base class's matrix-only `solve_linear_system` cannot pass, so the
        preparation stack is skipped and the backend solves the system whole.
        """
        problem = _assembled(problem)
        result = np.asarray(self.solve(
            problem.A, _rhs(problem), model=model,
            state=problem.state if state is None else state,
            dt=problem.dt,
        ), dtype=float).ravel()
        report = self.get_solve_report(Iterations=int(self.last_iterations))
        return self.store_increments(problem, result), result, report

    def apply_two_stage_preconditioner(self, r, A, Ap, L, U, p_inx,
                                       elliptic_solver=None,
                                       elliptic_sign=1.0):
        """One application of CPR's two-stage preconditioner.

        Stage one solves the pressure system `Ap`; stage two applies an
        incomplete factorisation to what stage one left behind. The order is
        the method: the pressure stage removes the elliptic, globally coupled
        part of the residual - the part a local factorisation cannot see - and
        ILU then cleans up the local, hyperbolic remainder.

        The `r = r - A @ x` between the stages is not bookkeeping. Without it
        the second stage is applied to the *original* residual and the two
        corrections overlap instead of composing, which still converges and
        does so at roughly the rate of ILU alone - CPR's advantage disappears
        with no other symptom.

        The scaling is MRST's and it exists because the pressure solve has an
        *absolute* tolerance while CPR wants a *relative* one. Scaling the
        right-hand side by `(abstol/reltol)/||r_p||` makes the inner solver's
        fixed absolute tolerance behave as the relative tolerance asked for;
        the result is divided back out immediately, so it does not change the
        preconditioner - only how hard the inner solve works.
        """
        solver = elliptic_solver if elliptic_solver is not None \
            else getattr(self, "elliptic_solver", None)
        if solver is None:
            raise ValueError(
                "apply_two_stage_preconditioner needs an elliptic solver for "
                "the pressure stage; pass elliptic_solver=... or set "
                "self.elliptic_solver")

        r = np.asarray(r, dtype=float).ravel()
        p_inx = np.asarray(p_inx)
        rp = r[p_inx]

        scaling = 1.0
        if np.isfinite(self.relativeTolerance):
            norm = float(np.linalg.norm(rp))
            if norm > 0.0:
                scaling = (getattr(solver, "tolerance", self.tolerance)
                           / self.relativeTolerance)*(1.0/norm)
        rp = elliptic_sign*rp*scaling

        x = np.zeros_like(r)
        x[p_inx] = np.asarray(
            solver.solve_linear_system(Ap, rp), dtype=float).ravel()/scaling

        # Compose the stages rather than overlapping them.
        residual = r - np.asarray(A @ x, dtype=float).ravel()
        return x + spla.spsolve_triangular(
            sp.csr_matrix(U), spla.spsolve_triangular(
                sp.csr_matrix(L), residual, lower=True), lower=False)

    applyTwoStagePreconditioner = apply_two_stage_preconditioner

    def get_description(self):
        return "Constrained pressure residual (CPR) with GMRES."


class CPRSolverAD(LinearSolverAD):
    """MRST `CPRSolverAD`."""

    def __init__(self, **options):
        super().__init__(tolerance=1e-8, max_iterations=25)
        from pymrst.utils.units import BARSA
        self.ellipticSolver = None
        self.relativeTolerance = 1e-4
        self.pressureScaling = 1/(200*BARSA)
        self.diagonalTol = 1e-2
        self.trueIMPES = False
        self.ellipticVarName = "pressure"
        self.ellipticSign = -1
        self.reduceToCell = True
        for key, item in options.items():
            key = {"maxIterations": "max_iterations", "keepNumber": "keep_number"}.get(key, key)
            if not hasattr(self, key):
                raise TypeError(f"unknown option {key!r}")
            setattr(self, key, item)
        if self.ellipticSolver is None:
            from pymrst.modules.ad_core.solvers.concrete_solvers import BackslashSolverAD
            self.ellipticSolver = BackslashSolverAD()
        assert isinstance(self.ellipticSolver, LinearSolverAD)

    @property
    def maxIterations(self) -> int:
        return self.max_iterations

    @maxIterations.setter
    def maxIterations(self, value: int) -> None:
        self.max_iterations = value

    def solve_linear_system(self, A, b):
        raise NotImplementedError("Not supported directly - this is a preconditioner")

    def solve_linear_problem(self, problem0, model=None):
        """MRST `solveLinearProblem`: `(dx, result, report)`."""
        from time import perf_counter

        from pymrst.modules.ad_core.solvers.gmres_ilu_solver_ad import ilu
        from pymrst.utils.adi import value
        timer = perf_counter()
        problem0 = _copy_problem(problem0)
        if problem0.A is not None:
            problem0 = problem0.clear_system() or problem0
            if self.verbose:
                print("System already assembled. CPR will re-assemble!")
        oil = _first_index(problem0.index_of_equation_name("oil"))
        if oil is not None and oil != 0:
            order = list(range(problem0.num_eq()))
            order[oil], order[0] = 0, oil
            problem0 = problem0.reorder_equations(order) or problem0
        p_index = _first_index(problem0.index_of_primary_variable(self.ellipticVarName))
        problem = _copy_problem(problem0)
        is_current = problem.index_of_type("cell")
        keep_number0 = self.keep_number
        sample = next(eq for eq in problem.equations if hasattr(eq, "jac"))
        assert type(sample).__name__ == "ADI", (
            "MRST's CPRSolverAD is used with the sparse AD backend only.")
        num_vars = problem.num_vars or [block.shape[1] for block in sample.jac]
        offsets = np.concatenate([[0], np.cumsum(num_vars)])
        p_inx = np.arange(offsets[p_index], offsets[p_index + 1])
        eliminated = None
        if self.reduceToCell and self.keep_number is None:
            problem, eliminated = self.reduce_to_variable(_copy_problem(problem0), is_current)
        cell_eq_no = int(np.count_nonzero(is_current))
        n_cell = int(problem.get_equation_var_num(0)[0])
        stype = "trueIMPES" if self.trueIMPES else "simple"
        scale = model.get_scaling_factors_cpr(problem, problem.equation_names, stype)
        for i, factor in enumerate(scale):
            if np.size(value(factor)) > 1 or np.any(value(factor) != 0):
                problem.equations[i] = problem.equations[i]*factor
        eqs = list(problem.equations)
        if np.isfinite(self.diagonalTol):
            is_elliptic = np.zeros((n_cell, cell_eq_no), dtype=bool)
            for i in range(cell_eq_no):
                pj = sp.csr_array(eqs[i].jac[p_index])
                assert pj.shape[1] == n_cell
                d = pj.diagonal()
                sod = np.asarray(abs(pj).sum(axis=1)).ravel() - np.abs(d)
                with np.errstate(divide="ignore", invalid="ignore"):
                    is_elliptic[d/sod > self.diagonalTol, i] = True
            is_elliptic[~is_elliptic.any(axis=1), p_index] = True
            bad = np.flatnonzero(~is_elliptic[:, p_index])
            if bad.size:
                is_zero_diag = np.column_stack([
                    sp.csr_array(eqs[p_index].jac[i]).diagonal()[bad] == 0
                    for i in range(cell_eq_no)])
                candidate = is_elliptic[bad, :] & ~is_zero_diag
                replacement = np.where(candidate.any(axis=1), candidate.argmax(axis=1), -1)
                for i in np.unique(replacement):
                    if i == p_index:
                        continue
                    assert i >= 0, "No replacement equation for a non-elliptic row"
                    rows = bad[replacement == i]
                    swapped_i = problem.equations[i] + 0.0
                    swapped_i[rows] = eqs[p_index][rows]
                    swapped_p = problem.equations[p_index] + 0.0
                    swapped_p[rows] = eqs[i][rows]
                    problem.equations[i] = swapped_i
                    problem.equations[p_index] = swapped_p
            # MRST weights by the first cell equation's mask, from the
            # pre-swap equations.
            problem.equations[p_index] = is_elliptic[:, 0]*eqs[p_index]
        else:
            is_elliptic = np.ones((n_cell, cell_eq_no), dtype=bool)
        for i in range(cell_eq_no):
            if i == p_index or np.all(np.asarray(value(scale[i])) == 0):
                continue
            problem.equations[p_index] = problem.equations[p_index] + is_elliptic[:, i]*eqs[i]
        if self.pressureScaling != 1:
            for i, eq in enumerate(problem.equations):
                jac = [j.copy() for j in eq.jac]
                jac[p_index] = jac[p_index]/self.pressureScaling
                problem.equations[i] = type(eq)(np.array(eq.val, copy=True), jac)
        problem = _assembled(problem.clear_system() or problem)
        A, b = problem.A, _rhs(problem)
        A, b, lsys, _ = self.reduce_linear_system(sp.csr_array(A), b, False, None)
        A = sp.csr_array(A)
        L, U = ilu(A, {"type": "nofill"})
        if self.keep_number is None:
            Ap = sp.csr_array(problem.equations[p_index].jac[p_index])
        else:
            Ap = A[p_inx][:, p_inx]
        Ap = self.ellipticSign*Ap
        self.ellipticSolver = self.ellipticSolver.setup_solver(Ap, b[p_inx]) or self.ellipticSolver
        from pymrst.modules.ad_core.solvers.gmres_ilu_solver_ad import _lu_solve
        prec = lambda r: self.apply_two_stage_preconditioner(r, A, Ap, L, U, p_inx, _lu_solve)
        assert np.all(np.isfinite(b)), "Linear system rhs must have finite entries."
        t_prep = perf_counter() - timer
        Minv_A = spla.LinearOperator(A.shape, lambda v: prec(A @ v), dtype=float)
        Minv_b = prec(b)
        its = [0]

        def count(_):
            its[0] += 1

        maxit = int(min(self.max_iterations, A.shape[0]))
        try:
            cpr_sol, flag = spla.gmres(Minv_A, Minv_b, rtol=self.tolerance, atol=0.0,
                                       restart=maxit, maxiter=1, callback=count,
                                       callback_type="pr_norm")
        finally:
            self.ellipticSolver = (self.ellipticSolver.cleanup_solver(Ap, b[p_inx])
                                   or self.ellipticSolver)
        relres = (np.linalg.norm(Minv_b - Minv_A @ cpr_sol)
                  / max(np.linalg.norm(Minv_b), np.finfo(float).tiny))
        t_solve = perf_counter() - timer - t_prep
        cpr_sol = self.recover_linear_system(cpr_sol, lsys)
        if self.pressureScaling != 1:
            cpr_sol[p_inx] = cpr_sol[p_inx]/self.pressureScaling
        dx = self.store_increments(problem, cpr_sol)
        if self.reduceToCell and keep_number0 is None:
            if isinstance(problem, LinearizedProblemAD):
                dx = LinearizedProblemAD.recover_from_single_variable_type(problem, problem0, dx,
                                                                           eliminated)
            else:
                dx = problem.recover_from_single_variable_type(problem0, dx, eliminated)
        self.keep_number = keep_number0
        t_post = perf_counter() - timer - t_prep - t_solve
        messages = {1: "GMRES did not converge. Reached maximum iterations",
                    2: "GMRES did not converge. Preconditioner was ill-conditioned.",
                    3: "GMRES stagnated. Unable to reduce residual."}
        do_print = self.verbose if flag == 0 else True
        if flag != 0:
            print(messages.get(flag, messages[1]), end="")
        elif do_print:
            print("GMRES converged:", end="")
        if do_print:
            print(f" Final residual: {relres:1.2e} after {its[0]} iterations "
                  f"(tol: {self.tolerance:1.2e}) ")
        result = np.concatenate([np.ravel(value(d)) for d in dx])
        self.last_iterations = its[0]
        report = self.get_solve_report(Iterations=its[0], Converged=flag == 0,
                                       SolverTime=t_solve + t_prep + t_post,
                                       LinearSolutionTime=t_solve, PreparationTime=t_prep,
                                       PostProcessTime=t_post, Residual=relres)
        report["FlagGMRES"] = flag
        return dx, result, report

    def apply_two_stage_preconditioner(self, r, A, Ap, L, U, p_inx, lu_solve):
        """Pressure solve on the elliptic block, then ILU(0) on the rest."""
        es = self.ellipticSolver
        rp = r[p_inx]
        if np.isfinite(self.relativeTolerance):
            scaling = (es.tolerance/self.relativeTolerance)*(1/np.linalg.norm(rp))
        else:
            scaling = 1.0
        rp = self.ellipticSign*rp*scaling
        x = np.zeros(r.shape)
        x[p_inx] = np.asarray(es.solve_linear_system(Ap, rp)).ravel()/scaling
        r = r - A @ x
        return x + lu_solve(L, U, r)

    def get_description(self):
        _, sn_sub = _description(self.ellipticSolver)
        return (f"Matlab implementation of constrained pressure residual (CPR) with "
                f"dynamic row-sum. Elliptic solver: {sn_sub}.", f"Matlab-CPR-{sn_sub}")

    solveLinearSystem = solve_linear_system
    solveLinearProblem = solve_linear_problem
    applyTwoStagePreconditioner = apply_two_stage_preconditioner
    getDescription = get_description


def _first_index(found):
    """The position MRST's `find(mask, 1)` gives, or None: the faithful problem
    returns a mask, `LinearizedProblemAD` a position."""
    if isinstance(found, np.ndarray):
        hits = np.flatnonzero(found)
        return int(hits[0]) if hits.size else None
    return found


def _copy_problem(problem):
    """MATLAB's value copy of a `LinearizedProblem`: fresh lists, shared AD
    objects (which are never modified in place here)."""
    from copy import copy
    out = copy(problem)
    out.equations = list(problem.equations)
    out.types = list(problem.types)
    out.equation_names = list(problem.equation_names)
    out.primary_variables = list(problem.primary_variables)
    return out


def _description(solver):
    d = solver.get_description()
    return d if isinstance(d, tuple) else (d, type(solver).__name__)


__all__ = ["CPRSolverAD", "CPRSolverADAD"]


if __name__ == "__main__":
    from pymrst.modules.ad_core.solvers.concrete_solvers import BackslashSolverAD
    from pymrst.modules.ad_core.utils.convert_deck_schedule_to_mrst import (
        convert_deck_schedule_to_mrst,
    )
    from pymrst.modules.ad_core.utils.select_model_from_deck import select_model_from_deck
    from pymrst.modules.test_suite.dataset_functions.spe1 import setup_spe1
    from pymrst.params.gravity.gravity import isolated_gravity
    from pymrst.utils.adi import value

    def _spe1_problem():
        G, rock, fluid, deck, state = setup_spe1()
        model = select_model_from_deck(G, rock, fluid, deck)
        schedule = convert_deck_schedule_to_mrst(model, deck)
        forces = model.get_driving_forces(schedule.controls[0])
        model = model.validate_model(forces)
        state = model.validate_state(state)
        dt = schedule.step_val[0]
        model, st = model.prepare_timestep(state, state, dt, forces)
        model.FacilityModel = model.FacilityModel.validate_model(forces)
        model, st = model.prepare_report_step(st, state, dt, forces)
        model, st = model.prepare_timestep(st, state, dt, forces)
        problem, _ = model.get_equations(state, st, dt, forces, iteration=1)
        return model, problem

    def test_cpr_reproduces_the_direct_increment():
        with isolated_gravity():
            model, problem = _spe1_problem()
            dx_cpr, _, report = CPRSolverAD(tolerance=1e-13, maxIterations=200) \
                .solve_linear_problem(problem, model)
            dx_ref, _, _ = BackslashSolverAD().solve_linear_problem(problem, model)
        assert report["Converged"]
        # A controlled rate's increment is zero to round-off, so the well
        # rates are compared on the scale of the largest rate increment.
        rates = [i for i, n in enumerate(problem.primary_variables) if n.startswith("q")]
        rate_scale = max(np.abs(np.ravel(value(dx_ref[i]))).max() for i in rates)
        for i, (a, b) in enumerate(zip(dx_cpr, dx_ref)):
            a, b = np.ravel(value(a)), np.ravel(value(b))
            scale = rate_scale if i in rates else np.abs(b).max()
            assert np.allclose(a, b, rtol=1e-8, atol=1e-8*scale), np.abs(a - b).max()
        print(f"ok test_cpr_reproduces_the_direct_increment ({report['Iterations']} GMRES iterations)")

    def test_the_callers_problem_is_left_unchanged():
        with isolated_gravity():
            model, problem = _spe1_problem()
            names, before = list(problem.equation_names), [value(e).copy() for e in problem.equations]
            CPRSolverAD().solve_linear_problem(problem, model)
        assert problem.equation_names == names
        assert all(np.array_equal(value(a), b) for a, b in zip(problem.equations, before))
        print("ok test_the_callers_problem_is_left_unchanged")

    def test_mrsts_defaults():
        s = CPRSolverAD()
        assert (s.relativeTolerance, s.diagonalTol, s.ellipticSign, s.reduceToCell) == (1e-4, 1e-2, -1, True)
        assert type(s.ellipticSolver).__name__ == "BackslashSolverAD"
        print("ok test_mrsts_defaults")

    test_cpr_reproduces_the_direct_increment()
    test_the_callers_problem_is_left_unchanged()
    test_mrsts_defaults()
    print("\nAll tests passed!")
