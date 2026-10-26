"""MRST ``ad-core/solvers/NonLinearSolver.m``.

Newton's method over a report step for an AD-OO model. `solve_timestep`
splits the step into ministeps chosen by a timestep selector, cutting on
failure; `solve_ministep` iterates `model.step_function` until the model
reports convergence or failure, with optional relaxation or line search
once oscillation or stagnation is detected.

The legacy `pymrst.modules.ad_core.simulators.nonlinear_solver` drives the
legacy model contract; this one drives the AD-OO models.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from pymrst.modules.ad_core.ad_oo.physical_model import (
    StepReport,
    copy_state,
    merge_options,
)
from pymrst.utils.adi import value

ALIASES: Dict[str, str] = {"converged": "Converged", "iterations": "Iterations",
                           "dt": "Timestep"}
MAX_STORED_REPORTS: int = 128


class MinistepReport(StepReport):
    """MRST's ministep report, also readable by PyMRST's timestep selectors
    (which ask for `converged`, `iterations` and `dt`)."""

    def __getattr__(self, name: str) -> Any:
        return super().__getattr__(ALIASES.get(name, name))


class NonLinearSolver:
    """MRST `NonLinearSolver`."""

    def __init__(self, **options: Any):
        self.maxIterations = 25
        self.minIterations = 1
        self.maxTimestepCuts = 6
        self.LinearSolver = None
        self.verbose = None
        self.reportLevel = 0
        self.identifier = ""
        self.timeStepSelector = None
        self.useRelaxation = False
        self.relaxationParameter = 1.0
        self.relaxationType = "dampen"
        self.relaxationIncrement = 0.1
        self.relaxationDecrement = None
        self.minRelaxation = 0.5
        self.maxRelaxation = 1.0
        self.oscillationThreshold = 1.0
        self.useLinesearch = False
        self.alwaysUseStabilization = False
        self.linesearchReductionFactor = 0.5
        self.linesearchDecreaseFactor = 1.0
        self.linesearchMaxIterations = 10
        self.linesearchConvergenceNames: List[str] = []
        self.linesearchResidualScaling = None
        self.linesearchReductionFn = None
        self.enforceResidualDecrease = False
        self.acceptanceFactor = 1.0
        self.stagnateTol = 1e-2
        self.errorOnFailure = True
        self.continueOnFailure = False
        self.convergenceIssues = False
        self.previousIncrement = None
        self.previousStepReport = None
        merge_options(self, options)
        if self.LinearSolver is None:
            from pymrst.modules.ad_core.solvers.concrete_solvers import BackslashSolverAD
            self.LinearSolver = BackslashSolverAD()
        if self.timeStepSelector is None:
            from pymrst.modules.ad_core.timesteps.simple_time_step_selector import (
                SimpleTimeStepSelector,
            )
            self.timeStepSelector = SimpleTimeStepSelector()
        if self.verbose is None:
            self.verbose = False

    @property
    def max_iterations(self) -> int:
        """PyMRST's timestep selectors read the iteration cap by this name."""
        return self.maxIterations

    def solve_timestep(self, state0: Any, dT: float, model: Any,
                       initialGuess: Any = None, **forces_arg: Any):
        """Advance one report step; `(state, report, ministates)`."""
        timer = perf_counter()
        driving_forces = dict(model.get_valid_driving_forces())
        driving_forces["controlId"] = np.nan
        driving_forces.update(forces_arg)
        state = copy_state(state0 if initialGuess is None else initialGuess)
        model, state = model.prepare_report_step(state, state0, dT, driving_forces)
        initial_guess = state
        assert dT >= 0, f"{self.get_id()}Negative timestep detected."
        converged = False
        done = False
        early_done = False
        it_count = cutting_count = step_count = accept_count = 0
        t_local = 0.0
        state0 = copy_state(state0)
        if "time" not in state0:
            state0["time"] = 0.0
        t_start = state0["time"]
        is_final_ministep = False
        state0_inner = state0
        state_prev = None
        dt_prev = None
        reports: List[MinistepReport] = []
        ministates: List[Any] = []
        stepsel = self.timeStepSelector
        stepsel.new_control_step(driving_forces)
        dt_min = dT/(2**self.maxTimestepCuts)
        timestep_failure = False
        dt = dT
        while not done:
            if timestep_failure:
                dt_selector = stepsel.cut_timestep(dt_prev, dt, model, self,
                                                   state_prev, state0_inner)
            else:
                dt_selector = stepsel.pick_timestep(dt_prev, dt, model, self,
                                                    state_prev, state0_inner)
            dt_model = model.get_maximum_timestep(state, state0_inner,
                                                  dT - t_local, driving_forces)
            dt_choice = min(dt_selector, dt_model)
            if t_local + dt_choice >= dT:
                is_final_ministep = True
                dt = dT - t_local
            else:
                dt = dt_choice
            if self.verbose and dt < dT:
                print(f"{self.get_id()}Solving ministep : {dt:g} s "
                      f"({dt/dT*100:.2f} % of control step, control step "
                      f"currently {t_local/dT*100:.2f} % complete)")
            state = copy_state(state)
            state["time"] = t_start + t_local + dt
            state, failure, rep = self.solve_ministep(model, state, state0_inner,
                                                      dt, driving_forces)
            converged = rep["Converged"]
            its = rep["Iterations"]
            rep["LocalTime"] = state["time"]
            reports.append(rep)
            if not is_final_ministep or dt/dt_choice > 0.9:
                stepsel.store_timestep(rep)
            it_count += its
            step_count += 1
            if converged:
                t_local += dt
                dt_prev = dt
                state_prev = state0_inner
                state0_inner = state
                accept_count += 1
                ministates.append(state)
                timestep_failure = False
            else:
                stop_now = dt <= dt_min or (failure and self.continueOnFailure)
                if not (stop_now and self.continueOnFailure):
                    state = initial_guess if accept_count == 0 else state0_inner
                msg = f"{self.get_id()}Did not find a solution: "
                msg_fail = ""
                if failure:
                    msg_fail = reports[-1]["NonlinearReport"][-1]["FailureMsg"]
                    msg += f"Model step resulted in failure state. Reason: {msg_fail}"
                else:
                    msg += "Maximum number of substeps stopped timestep reduction"
                if stop_now:
                    if self.errorOnFailure:
                        raise RuntimeError(msg)
                    if self.verbose:
                        import warnings
                        warnings.warn(msg)
                    converged = False
                    if not (self.continueOnFailure and failure):
                        break
                    timestep_failure = True
                else:
                    if self.verbose:
                        reason = (f"Solver failure after {its - 1} iterations"
                                  if failure else
                                  f"Solver did not converge in {its - 1} iterations")
                        print(f"{self.get_id()}{reason} for timestep of length "
                              f"{dt:g} s. Cutting timestep.")
                    timestep_failure = True
                    cutting_count += 1
                is_final_ministep = False
            early_done = bool(driving_forces["stopFunction"](model, state,
                                                            state0_inner))
            done = early_done or (is_final_ministep and converged)
        elapsed = perf_counter() - timer
        if self.verbose:
            print(f"{self.get_id()}Solved timestep with {accept_count} accepted "
                  f"ministep(s) (Iteration(s): {step_count - accept_count} "
                  f"rejected, {it_count} total)")
        report = StepReport(Iterations=it_count, Converged=converged,
                            EarlyStop=early_done, WallTime=elapsed,
                            MinistepCuttingCount=cutting_count,
                            StepReports=reports)
        return state, report, ministates

    def solve_ministep(self, model: Any, state: Any, state0: Any, dt: float,
                       driving_forces: Dict[str, Any]):
        """Newton iterations at a fixed `dt`; `(state, failure, report)`."""
        self.relaxationParameter = self.maxRelaxation
        self.convergenceIssues = False
        max_its = 0 if model.stepFunctionIsLinear else self.maxIterations
        reports: List[Any] = []
        model, state = model.prepare_timestep(state, state0, dt, driving_forces)
        store_states = self.reportLevel > 2
        res: Optional[np.ndarray] = None
        prev_best = None
        converged = failure = False
        i = 0
        for i in range(1, max_its + 2):
            state, step_report = model.step_function(state, state0, dt,
                                                     driving_forces,
                                                     self.LinearSolver, self, i)
            converged = bool(step_report["Converged"])
            failure = bool(step_report["Failure"])
            if store_states:
                step_report["state"] = state
            reports.append(step_report)
            if converged or failure:
                break
            residuals = np.asarray(step_report["Residuals"], dtype=float)
            if i > 1 and self.enforceResidualDecrease:
                if np.all(residuals >= prev_best):
                    break
            prev_best = residuals
            if self.useRelaxation or self.useLinesearch or self.alwaysUseStabilization:
                if i == 1:
                    res = np.full((max_its + 1, residuals.size), np.nan)
                res[i - 1, :] = residuals
                is_ok = np.asarray(step_report["ResidualsConverged"], dtype=bool)
                bad = self.check_for_oscillations(res, i) | self.check_for_stagnation(res, i)
                relax = (np.sum(bad & ~is_ok) >= self.oscillationThreshold*np.sum(~is_ok)
                         and not np.all(is_ok))
                if relax:
                    if self.verbose and not self.convergenceIssues:
                        print("Convergence issues detected: Activating "
                              + ("line-search." if self.useLinesearch else "relaxation."))
                    self.convergenceIssues = True
                    dw = (self.relaxationIncrement if self.relaxationDecrement is None
                          else self.relaxationDecrement)
                    self.relaxationParameter = max(self.relaxationParameter - dw,
                                                   self.minRelaxation)
                else:
                    self.relaxationParameter = min(
                        self.relaxationParameter + self.relaxationIncrement,
                        self.maxRelaxation)
        if "Solved" in reports[-1]:
            its = i - int(not reports[-1]["Solved"])
        else:
            its = i - int(converged)
        self.convergenceIssues = False
        if converged:
            state, final_update = model.update_after_convergence(state0, state, dt,
                                                                 driving_forces)
            reports[-1]["FinalUpdate"] = final_update
        report = MinistepReport(NonlinearReport=reports, Converged=converged,
                                Timestep=dt, Iterations=its)
        return state, failure, report

    def stabilize_newton_increments(self, model: Any, problem: Any,
                                    dx: List[Any]):
        dx_prev = self.previousIncrement
        w = self.relaxationParameter
        report = {"relaxationParameter": w}
        if w < 1:
            kind = self.relaxationType.lower()
            if kind == "dampen":
                dx = [d*w for d in dx]
            elif kind == "sor":
                if dx_prev is None:
                    return dx, report
                dx = [d*w + (1 - w)*p for d, p in zip(dx, dx_prev)]
            elif kind != "none":
                raise ValueError("Unknown relaxationType: Valid options are "
                                 "'dampen', 'none' or 'sor'")
        self.previousIncrement = dx
        return dx, report

    def apply_linesearch(self, model: Any, state0: Any, state: Any, problem0: Any,
                         dx: List[Any], driving_forces: Any, **options: Any):
        assert 0 < self.linesearchReductionFactor < 1, (
            "NonLinearSolver.linesearchReductionFactor must be less than unity "
            "and positive.")
        iteration = problem0.iteration_no
        dt = problem0.dt

        def assemble(trial):
            problem, _ = model.get_equations(state0, trial, dt, driving_forces,
                                             resOnly=True, iteration=iteration,
                                             **options)
            return problem

        def update(step):
            return model.update_state(copy_state(state), problem0, step,
                                      driving_forces)

        factor = self.linesearchDecreaseFactor
        converged = False
        val0, tol, names = model.get_convergence_values(problem0)
        val0 = np.asarray(val0)/np.asarray(tol)
        if self.linesearchResidualScaling is None or \
                np.size(self.linesearchResidualScaling) != np.size(tol):
            self.linesearchResidualScaling = np.asarray(tol)
        active = self.get_active_names(names)
        v_best = self.linesearch_apply_update(val0, val0 <= 1, active)
        its = 0
        for its in range(1, self.linesearchMaxIterations + 1):
            state_next, update_report = update(dx)
            val = np.asarray(model.get_convergence_values(assemble(state_next))[0])
            val = val/np.asarray(tol)
            ok = val <= 1
            v = self.linesearch_apply_update(val, ok, active)
            if np.all(ok) or (np.any(v < v_best*factor) and v.sum() <= v_best.sum()):
                if self.verbose:
                    print(f"Linesearch reduction successful after {its} steps!")
                converged = True
                break
            dx = [d*self.linesearchReductionFactor for d in dx]
        if self.verbose and not converged:
            print("Linesearch was unable to reduce residual.")
        return state_next, update_report, {"Iterations": its, "Converged": converged}

    def check_for_oscillations(self, res: np.ndarray, index: int) -> np.ndarray:
        """1-based `index`: does the residual change direction?"""
        if index < 3:
            return np.zeros(res.shape[1], dtype=bool)
        old, mid, nxt = res[index - 3], res[index - 2], res[index - 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            return (nxt - mid)/(mid - old) < 0

    def check_for_stagnation(self, res: np.ndarray, index: int) -> np.ndarray:
        if index < 2:
            return np.zeros(res.shape[1], dtype=bool)
        prev, nxt = res[index - 2], res[index - 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.abs(nxt - prev)/prev < self.stagnateTol

    def get_id(self) -> str:
        return f"{self.identifier}: " if self.identifier else ""

    def linesearch_apply_update(self, v: np.ndarray, ok: np.ndarray,
                                active: np.ndarray) -> np.ndarray:
        v = (np.asarray(v)/np.asarray(self.linesearchResidualScaling))[active]
        ok = np.asarray(ok)[active]
        if self.linesearchReductionFn is not None:
            return self.linesearchReductionFn(v)
        return v*~ok

    def get_active_names(self, names: Sequence[str]) -> np.ndarray:
        if not self.linesearchConvergenceNames:
            return np.ones(len(names), dtype=bool)
        return np.array([n in self.linesearchConvergenceNames for n in names])

    solveTimestep = solve_timestep
    solveMinistep = solve_ministep
    stabilizeNewtonIncrements = stabilize_newton_increments
    applyLinesearch = apply_linesearch
    checkForOscillations = check_for_oscillations
    checkForStagnation = check_for_stagnation
    getId = get_id
    linesearchApplyUpdate = linesearch_apply_update
    getActiveNames = get_active_names


__all__ = ["NonLinearSolver", "MinistepReport"]


if __name__ == "__main__":
    def test_defaults_match_mrst():
        solver = NonLinearSolver()
        assert solver.maxIterations == 25 and solver.maxTimestepCuts == 6
        assert type(solver.LinearSolver).__name__ == "BackslashSolverAD"
        assert type(solver.timeStepSelector).__name__ == "SimpleTimeStepSelector"
        print("ok test_defaults_match_mrst")

    def test_oscillation_needs_a_change_of_direction():
        solver = NonLinearSolver()
        res = np.array([[1.0, 1.0], [0.5, 2.0], [0.8, 3.0]])
        np.testing.assert_array_equal(solver.check_for_oscillations(res, 3),
                                      [True, False])
        print("ok test_oscillation_needs_a_change_of_direction")

    def test_stagnation_is_a_small_relative_change():
        solver = NonLinearSolver()
        res = np.array([[1.0, 1.0], [0.995, 0.5]])
        np.testing.assert_array_equal(solver.check_for_stagnation(res, 2),
                                      [True, False])
        print("ok test_stagnation_is_a_small_relative_change")

    def test_dampening_scales_every_increment():
        solver = NonLinearSolver(relaxationParameter=0.5)
        dx, _ = solver.stabilize_newton_increments(None, None, [np.ones(2), 2*np.ones(1)])
        np.testing.assert_allclose(dx[0], 0.5)
        np.testing.assert_allclose(dx[1], 1.0)
        print("ok test_dampening_scales_every_increment")

    def test_report_is_readable_by_the_selectors():
        rep = MinistepReport(Converged=True, Iterations=3, Timestep=2.0)
        assert rep.converged and rep.iterations == 3 and rep.dt == 2.0
        print("ok test_report_is_readable_by_the_selectors")

    test_defaults_match_mrst()
    test_oscillation_needs_a_change_of_direction()
    test_stagnation_is_a_small_relative_change()
    test_dampening_scales_every_increment()
    test_report_is_readable_by_the_selectors()
    print("\nAll tests passed!")
