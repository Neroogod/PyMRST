"""Newton driver for the AD-OO models, ported from MRST ad-core/solvers.

What the hand-rolled models had was a fixed Newton loop: iterate up to N
times, give up. What MRST has, and what a model actually needs to survive
a real schedule, is:

- **Relaxation** (`solveTimestep`/`stabilizeNewtonIncrements`): damp the
  increment when the residual is not falling, so a step that would
  otherwise oscillate converges instead.
- **Line search**: if a full Newton step increases the residual, back off
  along the same direction rather than take it.
- **Timestep cutting**: when the Newton loop fails outright, halve the
  step and try again, rather than fail the simulation.
- **Adaptive timestepping**: with a selector from `ad_core.timesteps`, the
  solver *chooses* each ministep from what the last one cost instead of
  taking whatever it was handed - growing through the easy stretches of a
  schedule and shrinking before the hard ones rather than after.
- **Convergence reports**: what converged, what did not, and by how much -
  so a failure says why.

`SimulationReport` records all of it, which is what makes a hard case
debuggable instead of merely broken.
"""

from dataclasses import dataclass, field
from time import perf_counter
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse.linalg as spla

from pymrst.modules.ad_core.statefunctions.state_function import State
from pymrst.utils.adi import combine_equations, value


@dataclass
class LinearSolveDiagnostic:
    """One Newton linear solve, independent of the concrete solver API."""

    solver: str
    converged: bool
    iterations: int
    residual_norm: float
    relative_residual: float
    rhs_norm: float
    wall_seconds: float
    degrees_of_freedom: int


@dataclass
class StepReport:
    """What happened in one attempted timestep."""

    converged: bool
    iterations: int
    dt: float
    residuals: List[Dict[str, float]] = field(default_factory=list)
    cuts: int = 0
    line_searches: int = 0
    # the ministeps a report step was actually taken in, when a selector chose
    # them - `[dt1, dt2, ...]`, summing to dt
    ministeps: List[float] = field(default_factory=list)
    # Wall time, in seconds. `assembly_seconds` is the residual and Jacobian
    # construction; the linear-solve time is on each `LinearSolveDiagnostic`.
    # Recorded because a timing breakdown is otherwise unrecoverable after the
    # fact - which is what `getReportTimings` and the MEX-acceleration example
    # are about.
    wall_seconds: float = 0.0
    assembly_seconds: float = 0.0
    # One entry per Newton correction. Direct solvers report zero Krylov
    # iterations; their factorization is represented by wall time and residual.
    linear_solves: List[LinearSolveDiagnostic] = field(default_factory=list)

    @property
    def linear_iterations(self) -> List[int]:
        return [solve.iterations for solve in self.linear_solves]

    @property
    def total_linear_iterations(self) -> int:
        return sum(self.linear_iterations)

    def __repr__(self) -> str:
        status = "converged" if self.converged else "FAILED"
        extra = (f", {len(self.ministeps)} ministeps" if self.ministeps
                 else f", {self.cuts} cuts")
        return (f"StepReport({status}, dt={self.dt:.3g} s, "
                f"{self.iterations} its{extra})")


@dataclass
class SimulationReport:
    """Every step of a schedule, converged or not."""

    steps: List[StepReport] = field(default_factory=list)

    @property
    def converged(self) -> bool:
        return all(s.converged for s in self.steps)

    @property
    def total_iterations(self) -> int:
        return sum(s.iterations for s in self.steps)

    @property
    def total_cuts(self) -> int:
        return sum(s.cuts for s in self.steps)

    @property
    def total_linear_iterations(self) -> int:
        return sum(s.total_linear_iterations for s in self.steps)


class NonLinearSolver:
    """Newton with relaxation, line search, and timestep cutting."""

    def __init__(
        self,
        max_iterations: int = 25,
        max_timestep_cuts: int = 6,
        use_line_search: bool = False,
        use_relaxation: bool = False,
        relaxation_min: float = 0.5,
        relaxation_increment: float = 0.1,
        line_search_max: int = 5,
        always_use_stabilization: bool = False,
        oscillation_threshold: float = 1.0,
        linear_solver=None,
        timestep_selector=None,
        verbose: bool = True,
        observer=None,
        progress: Optional[str] = None,
    ):
        self.max_iterations = max_iterations
        self.max_timestep_cuts = max_timestep_cuts
        self.use_line_search = use_line_search
        self.use_relaxation = use_relaxation
        self.relaxation_min = relaxation_min
        self.relaxation_increment = relaxation_increment
        self.line_search_max = line_search_max
        # MRST's `alwaysUseStabilization`/`oscillationThreshold`
        # (NonLinearSolver.m): a debug switch to force stabilization every
        # iteration, and the fraction of still-unconverged residuals that
        # must be oscillating/stagnating before `convergenceIssues` activates.
        # `convergence_issues` is reset per ministep attempt and is a
        # *lagging* signal - see `_update_convergence_issues` - matching
        # PhysicalModel.m:755 (`is_struggling = alwaysUseStabilization ||
        # convergenceIssues`), which line search is gated on.
        self.always_use_stabilization = always_use_stabilization
        self.oscillation_threshold = oscillation_threshold
        self.convergence_issues = False
        # MRST NonLinearSolver's own relaxation model (Phase 2), which is a
        # different scheme from `use_relaxation` above rather than a rename of
        # it: PyMRST damps heuristically when the residual rises, MRST applies
        # a fixed `relaxation_parameter` under a named `relaxation_type` and
        # decides *when* to apply it from oscillation and stagnation detection.
        #
        # Both now exist. The MRST path is inert at the defaults - a relaxation
        # parameter of 1 means `stabilize_newton_increments` returns the
        # increment untouched - so nothing that ran before changes.
        self.relaxation_type = "dampen"
        self.relaxation_parameter = 1.0
        self.previous_increment = None
        self.stagnate_tol = 1.0e-2
        self.identifier = ""
        self.linesearch_convergence_names = None
        # None means a sparse direct solve. A linear solver may exploit the
        # block structure of the model's Jacobian instead - see
        # `ad_mechanics.PoroelasticSchurSolver`, which is 22x faster on a
        # field-scale coupled system and returns the same answer.
        self.linear_solver = linear_solver
        # None means "take the dt you are given, and halve it if Newton fails".
        # A selector instead *chooses* each ministep from what the last one
        # cost - see `ad_core.timesteps`.
        self.timestep_selector = timestep_selector
        self.verbose = verbose
        # ``verbose`` historically only reported failures and timestep cuts.
        # Keep it as the convenient public switch, but route display through a
        # small, dependency-free progress layer so successful report steps are
        # visible too.  ``newton`` additionally prints nonlinear residuals.
        if progress is None:
            progress = "steps" if verbose else "off"
        progress = str(progress).lower()
        aliases = {"none": "off", "false": "off", "true": "steps",
                   "step": "steps", "detailed": "newton"}
        progress = aliases.get(progress, progress)
        if progress not in {"off", "steps", "newton"}:
            raise ValueError("progress must be 'off', 'steps', or 'newton'")
        self.progress = progress
        # Optional, dependency-free observation hook. It receives only copied
        # scalar/report metadata and cannot participate in solver decisions.
        # Failures are isolated and retained for diagnostics.
        self.observer = observer
        self.observer_failures: List[Dict[str, str]] = []
        self._observation_context: Dict[str, int] = {}

    def _progress(self, message: str, *, detail: bool = False) -> None:
        """Print solver progress without affecting the numerical path."""
        if self.progress == "off" or (detail and self.progress != "newton"):
            return
        print(message, flush=True)

    @staticmethod
    def _format_simulated_time(seconds: float) -> str:
        days = float(seconds) / 86400.0
        if abs(days) >= 1.0:
            return f"{days:.3g} d"
        if abs(seconds) >= 1.0:
            return f"{seconds:.3g} s"
        return f"{seconds:.3e} s"

    def _report_step_progress(self, index: int, total: int, dt: float,
                              simulated_time: float, report: StepReport,
                              wall_seconds: float) -> None:
        status = "accepted" if report.converged else "FAILED"
        ministeps = len(report.ministeps)
        subdivision = (f", {ministeps} ministeps" if ministeps > 1 else "")
        self._progress(
            f"Report step {index + 1:>3}/{total:<3} | {status:<8} | "
            f"time {self._format_simulated_time(simulated_time):>9} | "
            f"dt {self._format_simulated_time(dt):>9} | "
            f"{report.iterations} Newton iterations | {report.cuts} cuts"
            f"{subdivision} | {wall_seconds:.2f} s wall"
        )

    def _observe(self, event: str, **payload) -> None:
        if self.observer is None:
            return
        try:
            self.observer(event, {**self._observation_context, **payload})
        except Exception as error:
            self.observer_failures.append({
                "event": event,
                "type": type(error).__name__,
                "message": str(error),
            })

    # -- one Newton loop at a fixed dt --------------------------------------
    def solve_ministep(self, model, state0: State, state: State,
                       dt: float) -> Tuple[State, StepReport]:
        """Newton to convergence at this dt, or report failure."""
        report = StepReport(converged=False, iterations=0, dt=dt)
        self._observe("ministep_started", dt=float(dt))
        st = state
        relax = 1.0
        prev_norm = np.inf
        started = perf_counter()
        # MRST resets `convergenceIssues` before a ministep attempt starts
        # (NonLinearSolver.m:358) and again once it ends (line 447), so it
        # never leaks stabilization state from one ministep - report step,
        # cut retry, or Newton attempt - into the next.
        self.convergence_issues = False

        # MRST calls `prepareTimestep` at the top of `solveMinistep`
        # (NonLinearSolver.m:368). It is per *ministep*, not per report step:
        # a cut step re-evaluates which wells are shut at the new dt.
        prepare = getattr(model, "prepare_timestep", None)
        if prepare is not None:
            model, st = prepare(st, state0, dt, None)

        for it in range(self.max_iterations):
            assembled = perf_counter()
            eqs = model.get_equations(state0, st, dt)
            converged, res = model.get_convergence(eqs, dt)
            report.assembly_seconds += perf_counter() - assembled
            report.residuals.append(res)
            report.iterations = it
            norm = max(res.values()) if res else 0.0
            self._progress(
                f"    Newton {it + 1:02d}: residual {norm:.6e}"
                f"{' (converged)' if converged else ''}",
                detail=True,
            )
            self._observe(
                "nonlinear_iteration_completed",
                iteration=int(it), dt=float(dt), converged=bool(converged),
                residuals={str(name): float(number)
                           for name, number in res.items()},
                relaxation=float(relax),
            )

            if converged:
                report.converged = True
                if hasattr(model, "after_convergence"):
                    st = model.after_convergence(st, state0, dt)
                report.wall_seconds = perf_counter() - started
                self.convergence_issues = False
                return st, report

            # `is_struggling` is read *before* this iteration's own
            # stabilization check updates `convergence_issues` below -
            # PhysicalModel.m:755 reads the solver's `convergenceIssues` as
            # it stood when this iteration's Newton correction was computed,
            # which was set by the *previous* iteration's residual history.
            is_struggling = self.always_use_stabilization or self.convergence_issues

            if self.use_relaxation:
                relax = self._update_relaxation(relax, norm, prev_norm)
            prev_norm = norm

            try:
                increment = self._solve_linear(eqs, model, st, dt)
                diagnostic = self._last_linear_diagnostic
                report.linear_solves.append(diagnostic)
                self._observe(
                    "linear_solve_completed",
                    iteration=int(it), dt=float(dt),
                    wall_seconds=diagnostic.wall_seconds,
                    increment_norm=float(np.linalg.norm(increment)),
                    degrees_of_freedom=diagnostic.degrees_of_freedom,
                    solver=diagnostic.solver,
                    linear_iterations=diagnostic.iterations,
                    residual_norm=diagnostic.residual_norm,
                    relative_residual=diagnostic.relative_residual,
                )
            except RuntimeError as error:
                # Iterative field solvers can fail on an oversized ministep.
                # That is a timestep failure, not a simulation crash: return
                # control to `solve_timestep` / `_solve_adaptive`, which can
                # cut dt and retry from the last converged state.
                report.residuals[-1] = dict(report.residuals[-1])
                report.residuals[-1]["linear_failure"] = float("inf")
                diagnostic = getattr(self, "_last_linear_diagnostic", None)
                if diagnostic is None:
                    diagnostic = LinearSolveDiagnostic(
                        solver="unknown", converged=False, iterations=0,
                        residual_norm=float("inf"), relative_residual=float("inf"),
                        rhs_norm=float("nan"), wall_seconds=0.0,
                        degrees_of_freedom=0,
                    )
                report.linear_solves.append(diagnostic)
                self._observe(
                    "linear_solve_failed", iteration=int(it), dt=float(dt),
                    error_type=type(error).__name__, message=str(error),
                    solver=diagnostic.solver,
                    linear_iterations=diagnostic.iterations,
                )
                if self.verbose:
                    print(f"  linear solve failed at dt={dt:.3g} s: {error}")
                report.wall_seconds = perf_counter() - started
                return st, report
            st_new = model.update_state(st, relax * increment)

            # MRST only spends a line search's extra `ResOnly` assemblies
            # when the solve is actually struggling (PhysicalModel.m:756);
            # otherwise the plain Newton update above already stands, exactly
            # as `model.updateState` would leave it in MRST's `else` branch.
            if self.use_line_search and is_struggling:
                st_new, n_ls = self._line_search(
                    model, state0, st, increment, relax, dt, norm
                )
                report.line_searches += n_ls

            st = st_new

            # Refresh `convergence_issues` from the residual history through
            # this iteration, for the *next* iteration's `is_struggling`
            # check - matching NonLinearSolver.m:402, which runs this same
            # check whenever any of the three stabilization switches are on.
            if self.use_relaxation or self.use_line_search or self.always_use_stabilization:
                self._update_convergence_issues(report.residuals, it, model)

        self.convergence_issues = False
        assembled = perf_counter()
        eqs = model.get_equations(state0, st, dt)
        converged, res = model.get_convergence(eqs, dt)
        report.assembly_seconds += perf_counter() - assembled
        report.residuals.append(res)
        self._observe(
            "nonlinear_iteration_completed",
            iteration=int(self.max_iterations), dt=float(dt),
            converged=bool(converged),
            residuals={str(name): float(number) for name, number in res.items()},
            relaxation=float(relax),
        )
        report.converged = converged
        report.iterations = self.max_iterations
        if converged and hasattr(model, "after_convergence"):
            st = model.after_convergence(st, state0, dt)
        report.wall_seconds = perf_counter() - started
        return st, report

    def _solve_linear(self, eqs, model=None, state=None,
                      dt: Optional[float] = None) -> np.ndarray:
        eq = combine_equations(*eqs)
        matrix = eq.jac[0]
        rhs = -np.asarray(eq.val, dtype=float).reshape(-1)
        solver = self.linear_solver
        solver_name = ("scipy.sparse.linalg.spsolve" if solver is None else
                       (solver.get_description() if hasattr(solver, "get_description")
                        else type(solver).__name__))
        started = perf_counter()
        try:
            if solver is not None:
                increment = solver.solve(
                    matrix, rhs, model=model, state=state, dt=dt,
                )
                iterations = int(getattr(solver, "last_iterations", 0))
            else:
                increment = spla.spsolve(matrix.tocsc(), rhs)
                iterations = 0
        except Exception:
            elapsed = perf_counter() - started
            self._last_linear_diagnostic = LinearSolveDiagnostic(
                solver=solver_name, converged=False,
                iterations=int(getattr(solver, "last_iterations", 0)) if solver is not None else 0,
                residual_norm=float("inf"), relative_residual=float("inf"),
                rhs_norm=float(np.linalg.norm(rhs)), wall_seconds=float(elapsed),
                degrees_of_freedom=int(rhs.size),
            )
            raise
        elapsed = perf_counter() - started
        increment = np.asarray(increment, dtype=float).reshape(-1)
        residual_norm = float(np.linalg.norm(matrix @ increment - rhs))
        rhs_norm = float(np.linalg.norm(rhs))
        relative = residual_norm / max(rhs_norm, np.finfo(float).tiny)
        self._last_linear_diagnostic = LinearSolveDiagnostic(
            solver=solver_name, converged=bool(np.isfinite(relative)),
            iterations=iterations, residual_norm=residual_norm,
            relative_residual=relative, rhs_norm=rhs_norm,
            wall_seconds=float(elapsed), degrees_of_freedom=int(rhs.size),
        )
        return increment

    # -- MRST NonLinearSolver.m, ported (Phase 2) ---------------------------

    def check_for_oscillations(self, res, index: int):
        """Per-equation: is the residual bouncing rather than descending?

        MRST's test is the ratio of successive changes,
        `(r_n - r_{n-1}) / (r_{n-1} - r_{n-2})`. A negative ratio means the
        residual reversed direction - it went down then up, or up then down -
        which is the signature of a Newton step overshooting. It needs three
        iterations before it can say anything.

        Oscillating is not the same as diverging, and that is the point: a
        residual that oscillates while trending down is converging badly and
        wants damping, not a timestep cut.
        """
        res = np.atleast_2d(np.asarray(res, dtype=float))
        if index < 2:
            return np.zeros(res.shape[1], dtype=bool)
        old, mid, new = res[index - 2], res[index - 1], res[index]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = (new - mid)/(mid - old)
        return np.where(np.isfinite(ratio), ratio < 0.0, False)

    checkForOscillations = check_for_oscillations

    def check_for_stagnation(self, res, index: int):
        """Per-equation: has the residual stopped moving?

        Relative to the previous residual, so it is scale-free. A stagnated
        equation is the one that will still be there in ten more iterations,
        and it is worth distinguishing from a slow one.
        """
        res = np.atleast_2d(np.asarray(res, dtype=float))
        if index < 1:
            return np.zeros(res.shape[1], dtype=bool)
        prev, new = res[index - 1], res[index]
        with np.errstate(divide="ignore", invalid="ignore"):
            change = np.abs(new - prev)/prev
        return np.where(np.isfinite(change), change < self.stagnate_tol, False)

    checkForStagnation = check_for_stagnation

    # Named residuals to their model tolerance attribute, MRST's
    # `checkConvergence` per-name `ResidualsConverged`. Every model in this
    # port that exposes `get_convergence` names its criteria from this set
    # (`GenericBlackOilModelAD.get_convergence`, and the compositional models
    # that share its CNV/MB/well convention) - a name this map does not
    # recognise is conservatively treated as not individually converged,
    # matching MRST's own behaviour when a criterion cannot be evaluated.
    _RESIDUAL_TOLERANCE_ATTRS = {
        "CNV": "tol_cnv", "MB": "tol_mb",
        "well": "tol_well", "control": "tol_well", "extra": "tol_well",
    }

    def _residual_is_ok(self, res: Dict[str, float], names, model) -> np.ndarray:
        """Per-name convergence (MRST's `stepReport.ResidualsConverged`)."""
        out = np.zeros(len(names), dtype=bool)
        for i, name in enumerate(names):
            attr = self._RESIDUAL_TOLERANCE_ATTRS.get(name)
            if attr is None:
                continue
            tol = getattr(model, attr, None)
            value_i = res.get(name)
            if tol is None or value_i is None:
                continue
            out[i] = value_i <= tol
        return out

    def _update_convergence_issues(self, residuals: List[Dict[str, float]],
                                   index: int, model) -> None:
        """MRST's oscillation/stagnation check (NonLinearSolver.m:402-438).

        Sets `self.convergence_issues`, read by `PhysicalModel.m`'s
        `is_struggling` gate the *next* time it is checked - this method is
        called once per completed iteration, after that iteration's own
        residual is already in `residuals`, so the signal it produces is
        exactly one iteration behind by construction (matching MRST: the
        outer loop's check runs after `stepFunction` returns).
        """
        names = list(residuals[-1].keys())
        history = np.array(
            [[r.get(n, np.nan) for n in names] for r in residuals], dtype=float
        )
        is_ok = self._residual_is_ok(residuals[-1], names, model)
        if np.all(is_ok):
            self.convergence_issues = False
            return
        is_oscillating = self.check_for_oscillations(history, index)
        is_stagnated = self.check_for_stagnation(history, index)
        bad = is_oscillating | is_stagnated
        nonconverged = ~is_ok
        n_nonconverged = int(nonconverged.sum())
        self.convergence_issues = bool(
            n_nonconverged > 0
            and (bad & nonconverged).sum() >= self.oscillation_threshold * n_nonconverged
        )

    def stabilize_newton_increments(self, model, problem, dx):
        """Damp the Newton increment, by whichever relaxation type is set.

        `dampen` scales the step; `sor` mixes it with the previous step, which
        smooths a direction that is swinging between iterations rather than
        merely being too long. Both are inert at a relaxation parameter of 1.

        Returns `(dx, report)`. The previous increment is remembered either
        way, so switching to `sor` mid-run has something to mix with.
        """
        weight = float(self.relaxation_parameter)
        report = {"relaxationParameter": weight,
                  "relaxation_parameter": weight}
        kind = str(self.relaxation_type).lower()
        if kind not in ("dampen", "sor", "none"):
            raise ValueError(
                f"unknown relaxation_type {self.relaxation_type!r}; expected "
                f"'dampen', 'sor' or 'none'")

        previous = self.previous_increment
        if weight < 1.0 and kind != "none":
            if kind == "dampen":
                dx = [d*weight for d in dx]
            elif previous is not None:
                dx = [d*weight + (1.0 - weight)*p
                      for d, p in zip(dx, previous)]
        self.previous_increment = list(dx)
        return dx, report

    stabilizeNewtonIncrements = stabilize_newton_increments

    def get_id(self) -> str:
        """A prefix for this solver's messages, so nested solvers are
        distinguishable in a log."""
        return f"{self.identifier}: " if self.identifier else ""

    getId = get_id

    def get_active_names(self, names):
        """Which residuals the line search is allowed to judge itself by.

        By default all of them. Restricting the set matters when one equation
        is expected to worsen while the one you care about improves - a line
        search watching everything would refuse a step that is right.
        """
        names = list(names)
        if not self.linesearch_convergence_names:
            return np.ones(len(names), dtype=bool)
        allowed = {str(n).lower() for n in self.linesearch_convergence_names}
        return np.array([str(n).lower() in allowed for n in names], dtype=bool)

    getActiveNames = get_active_names

    def apply_linesearch(self, model, state0, state, problem, dx, forces=None,
                         **kwargs):
        """MRST's name for the back-tracking line search this port already had.

        Delegates to `_line_search`; the increment and residual bookkeeping
        differ between the two frameworks, so this is a named entry point
        rather than a second implementation.
        """
        norm0 = kwargs.pop("norm0", float("inf"))
        dt = kwargs.pop("dt", getattr(problem, "dt", None) or 1.0)
        relax = kwargs.pop("relax", 1.0)
        updated, iterations = self._line_search(model, state0, state, dx,
                                                relax, dt, norm0)
        return updated, {"line_search_iterations": iterations}, {}

    applyLinesearch = apply_linesearch

    def linesearch_apply_update(self, model, state, dx, alpha: float):
        """Apply a scaled increment - the step a line search takes each try."""
        return model.update_state(state, alpha*dx)

    linesearchApplyUpdate = linesearch_apply_update

    def _update_relaxation(self, relax: float, norm: float,
                           prev_norm: float) -> float:
        """Damp when the residual is not falling; relax back when it is."""
        if norm > prev_norm:
            return max(relax - self.relaxation_increment, self.relaxation_min)
        return min(relax + self.relaxation_increment, 1.0)

    def _line_search(self, model, state0: State, st: State,
                     increment: np.ndarray, relax: float, dt: float,
                     norm0: float) -> Tuple[State, int]:
        """Back off along the Newton direction until the residual improves."""
        alpha = relax
        best = model.update_state(st, alpha * increment)
        for i in range(self.line_search_max):
            eqs = model.get_equations(state0, best, dt)
            _, res = model.get_convergence(eqs, dt)
            if max(res.values()) <= norm0:
                return best, i
            alpha *= 0.5
            best = model.update_state(st, alpha * increment)
        return best, self.line_search_max

    # -- one timestep, with cutting -----------------------------------------
    def solve_timestep(self, model, state0: State, dt: float,
                       control=None) -> Tuple[State, StepReport]:
        """Take a step of dt, cutting it in half on failure and re-trying.

        A cut step is taken as several sub-steps that sum to dt, so the
        schedule's report times are still honoured.
        """
        # MRST calls `prepareReportstep` here, at the top of `solveTimestep`
        # (NonLinearSolver.m:145), before any branch - so every path below,
        # adaptive and flow-discretized included, gets it. It is once per
        # report step by construction, which is what controls that read
        # `state0` depend on.
        prepare = getattr(model, "prepare_report_step", None)
        if prepare is not None:
            model, state0 = prepare(state0, state0, dt, control)

        if self.timestep_selector is not None:
            return self._solve_adaptive(model, state0, dt, control)

        flow = getattr(model, "FlowDiscretization", None)
        if flow is not None:
            from pymrst.modules.ad_core.statefunctions.flux.flow_state import (
                ImplicitFlowStateBuilder,
            )
            builder = flow.get_flow_state_builder()
            if not isinstance(builder, ImplicitFlowStateBuilder):
                return self._solve_flow_discretized(model, state0, dt, flow)

        self._observation_context.update(attempt_index=0, ministep_index=0)
        st, report = self.solve_ministep(model, state0, state0.copy(), dt)
        if report.converged:
            return st, report

        # The residual history of the most recent attempt that failed. Both
        # exits below used to build a fresh StepReport and leave `residuals`
        # at its empty default, so a step that converged only after cutting -
        # and, worse, a step that failed outright - reported no history at all.
        # Failure analysis reads exactly that history, so the one path where a
        # classification is wanted was the one path with nothing to classify:
        # a starved solve on Norne came back `converged=False` with zero
        # residuals and was classified `unknown`.
        last_failure = report

        for cut in range(1, self.max_timestep_cuts + 1):
            n_sub = 2 ** cut
            dt_sub = dt / n_sub
            self._observe(
                "timestep_cut", cut=int(cut), subdivisions=int(n_sub),
                report_dt=float(dt), trial_dt=float(dt_sub),
            )
            if self.verbose:
                print(f"  step failed; cutting to {n_sub} x {dt_sub:.3g} s")

            st = state0
            ok = True
            iters = 0
            residuals: List[Dict[str, float]] = []
            linear_solves: List[LinearSolveDiagnostic] = []
            for substep in range(n_sub):
                self._observation_context.update(
                    attempt_index=cut, ministep_index=substep,
                )
                st_new, rep = self.solve_ministep(model, st, st.copy(), dt_sub)
                iters += rep.iterations
                # Concatenated across the sub-steps, matching `iterations`,
                # which is already their sum.
                residuals.extend(rep.residuals)
                linear_solves.extend(rep.linear_solves)
                if not rep.converged:
                    ok = False
                    last_failure = rep
                    break
                st = st_new
            if ok:
                return st, StepReport(converged=True, iterations=iters, dt=dt,
                                      cuts=cut, residuals=residuals,
                                      linear_solves=linear_solves)

        return st, StepReport(converged=False, iterations=report.iterations,
                              dt=dt, cuts=self.max_timestep_cuts,
                              residuals=list(last_failure.residuals),
                              linear_solves=list(last_failure.linear_solves))

    def _solve_flow_discretized(self, model, state0, dt, flow):
        """Cross one report step using the flow builder's CFL/ministep policy."""
        st = state0
        elapsed = 0.0
        total_iterations = cuts = 0
        ministeps: List[float] = []
        linear_solves: List[LinearSolveDiagnostic] = []
        # Every attempt costs wall time, including the ones that are thrown
        # away - a report step that converged only after three cuts really did
        # take that long, and attributing only the successful attempt would
        # make cutting look free.
        wall = assembly = 0.0
        residuals: List[Dict[str, float]] = []
        forces = {"W": list(getattr(getattr(model, "facility", None),
                                    "wells", []))}
        floor = float(dt) / 2.0 ** self.max_timestep_cuts

        while elapsed < float(dt) * (1.0 - 1e-12):
            remaining = float(dt) - elapsed
            maximum = flow.get_maximum_timestep(
                model, st, st, remaining, forces
            )
            dt_try = min(remaining, float(maximum))
            if not np.isfinite(dt_try) or dt_try <= 0.0:
                dt_try = remaining

            attempt = 0
            while True:
                self._observation_context.update(
                    attempt_index=attempt, ministep_index=len(ministeps),
                )
                guess = st.copy()
                flow, guess = flow.prepare_timestep(
                    model, guess, st, dt_try, forces
                )
                candidate, rep = self.solve_ministep(
                    model, st, guess, dt_try
                )
                total_iterations += rep.iterations
                linear_solves.extend(rep.linear_solves)
                wall += rep.wall_seconds
                assembly += rep.assembly_seconds
                residuals.extend(rep.residuals)
                if rep.converged:
                    st = candidate
                    elapsed += dt_try
                    ministeps.append(dt_try)
                    break
                cuts += 1
                attempt += 1
                dt_try *= 0.5
                if dt_try < floor:
                    return st, StepReport(
                        converged=False, iterations=total_iterations, dt=dt,
                        cuts=cuts, ministeps=ministeps,
                        linear_solves=linear_solves, residuals=residuals,
                        wall_seconds=wall, assembly_seconds=assembly,
                    )

        return st, StepReport(
            converged=True, iterations=total_iterations, dt=dt,
            cuts=cuts, ministeps=ministeps,
            linear_solves=linear_solves, residuals=residuals,
            wall_seconds=wall, assembly_seconds=assembly,
        )

    def _solve_adaptive(self, model, state0: State, dt: float,
                        control=None) -> Tuple[State, StepReport]:
        """Cross a report step in as many ministeps as the selector wants.

        The report step `dt` is a *reporting* interval, not a numerical one:
        the answer is only required at its end. So the selector is free to
        cross it in whatever ministeps it likes, growing through the easy
        stretches and shrinking through the hard ones - and the last ministep
        is always trimmed to land exactly on `dt`, because a schedule that
        quietly simulates a different length of time is worse than no schedule.
        """
        sel = self.timestep_selector
        sel.new_control_step(control)

        # The floor. Below this we are not converging, we are subdividing, and
        # it is time to say so rather than grind.
        dt_min = dt / 2.0 ** self.max_timestep_cuts

        st = state0
        state_prev: Optional[State] = None
        dt_prev: Optional[float] = None
        t_local = 0.0
        dt_try = dt
        failed = False
        total_its = cuts = 0
        taken: List[float] = []
        linear_solves: List[LinearSolveDiagnostic] = []
        attempt = 0

        while t_local < dt * (1.0 - 1e-12):
            pick = sel.cut_timestep if failed else sel.pick_timestep
            dt_sel = pick(dt_prev, dt_try, model, self, state_prev, st)

            remaining = dt - t_local
            is_final = dt_sel >= remaining * (1.0 - 1e-12)
            dt_try = remaining if is_final else dt_sel

            self._observation_context.update(
                attempt_index=attempt, ministep_index=len(taken),
            )
            st_new, rep = self.solve_ministep(model, st, st.copy(), dt_try)
            attempt += 1
            total_its += rep.iterations
            linear_solves.extend(rep.linear_solves)

            # Do not let a *trimmed* final ministep into the history. It was
            # cut short to land on the report time, not because the physics
            # asked for it, and a controller that learns from it would conclude
            # the problem had suddenly got easier.
            if not is_final or dt_try > 0.9 * dt_sel:
                sel.store_timestep(rep)

            if rep.converged:
                state_prev, st = st, st_new
                dt_prev = dt_try
                t_local += dt_try
                taken.append(dt_try)
                failed = False
            else:
                cuts += 1
                failed = True
                if dt_try <= dt_min:
                    if self.verbose:
                        print(f"  ministep of {dt_try:.3g} s failed at the "
                              f"floor ({dt_min:.3g} s); giving up")
                    return st, StepReport(converged=False,
                                          iterations=total_its, dt=dt,
                                          cuts=cuts, ministeps=taken,
                                          linear_solves=linear_solves)
                if self.verbose:
                    print(f"  ministep of {dt_try:.3g} s failed; cutting")

        return st, StepReport(converged=True, iterations=total_its, dt=dt,
                              cuts=cuts, ministeps=taken,
                              linear_solves=linear_solves)

    # -- a whole schedule ----------------------------------------------------
    def simulate_schedule(self, model, state0: State, schedule,
                          after_step_fn=None,
                          ) -> Tuple[List[State], SimulationReport]:
        """Run every step of a schedule; return the states and the report.

        `schedule` is either a bare sequence of timesteps - the wells are then
        whatever the model was built with, fixed for the run - or a structured
        `ad_core.utils.schedule.Schedule`, whose control periods change the
        wells / boundary / sources as time advances. The structured form is
        what lets a well open partway through a run, or a rate step down.

        `after_step_fn` is MRST's `'afterStepFn'`, called after each accepted
        report step as::

            model, states, reports, solver, ok = fn(
                model, states, reports, solver, schedule, simtime)

        and the run **stops when it returns `ok = False`**. That veto is the
        whole point of the hook rather than an ornament on it: it is what lets
        a caller watch a run and end it - MRST's own use is a GUI stop button,
        and a monitor that cannot stop the run is just an observer, which this
        class already has.
        """
        states, report = [], SimulationReport()
        simtime: List[float] = []
        elapsed = 0.0
        for st, rep in self.iter_schedule(model, state0, schedule):
            states.append(st)
            report.steps.append(rep)
            elapsed += float(getattr(rep, "dt", 0.0))
            simtime.append(elapsed)
            if after_step_fn is None:
                continue
            model, states, steps, self, ok = after_step_fn(
                model, states, report.steps, self, schedule, simtime)
            report.steps = list(steps)
            if not ok:
                break
        return states, report

    def iter_schedule(self, model, state0: State, schedule):
        """Yield each completed report boundary using native solver semantics.

        Yielding occurs before the next report step begins. Consumers may
        persist the accepted state or stop iteration; the existing
        ``simulate_schedule`` method simply consumes this generator and keeps
        its historical return contract.
        """
        from pymrst.modules.ad_core.utils.schedule import Schedule

        if not isinstance(schedule, Schedule):
            yield from self._iter_timesteps(model, state0, np.atleast_1d(schedule))
            return

        st = state0
        last_control = -1
        simulated_time = 0.0
        for i in range(schedule.n_steps):
            self._observation_context = {
                "report_step_index": int(i),
                "control_index": int(schedule.step_control[i]),
            }
            self._observe(
                "report_step_started", dt=float(schedule.step_val[i]),
            )
            c = int(schedule.step_control[i])
            if c != last_control:              # a new control period begins
                old_wells = list(getattr(getattr(model, "facility", None),
                                         "wells", ()))
                self._apply_control(model, schedule.controls[c])
                new_wells = list(getattr(getattr(model, "facility", None),
                                         "wells", ()))
                # Realign first: the injector seeding indexes the per-well
                # arrays positionally, so it must see arrays that already
                # match the new well set.
                st, aligned_old = self._realign_well_state(
                    st, old_wells, new_wells)
                st = self._seed_injector_control_state(st, aligned_old,
                                                       new_wells)
                last_control = c
            started = perf_counter()
            st, rep = self.solve_timestep(
                model, st, float(schedule.step_val[i]),
                schedule.controls[c],
            )
            simulated_time += float(schedule.step_val[i])
            self._report_step_progress(
                i, schedule.n_steps, float(schedule.step_val[i]),
                simulated_time, rep, perf_counter() - started,
            )
            yield st, rep
            self._observe(
                "step_accepted" if rep.converged else "step_rejected",
                dt=float(schedule.step_val[i]), iterations=int(rep.iterations),
                cuts=int(rep.cuts),
            )
            if not rep.converged:
                if self.verbose:
                    print(f"  step {i} did not converge; stopping")
                break

    def _simulate_timesteps(self, model, state0, dts):
        states, report = [], SimulationReport()
        for st, rep in self._iter_timesteps(model, state0, dts):
            states.append(st)
            report.steps.append(rep)
        return states, report

    def _iter_timesteps(self, model, state0, dts):
        st = state0
        dts = np.atleast_1d(dts)
        simulated_time = 0.0
        for index, dt in enumerate(dts):
            self._observation_context = {"report_step_index": int(index)}
            self._observe("report_step_started", dt=float(dt))
            started = perf_counter()
            st, rep = self.solve_timestep(model, st, float(dt))
            simulated_time += float(dt)
            self._report_step_progress(
                index, len(dts), float(dt), simulated_time, rep,
                perf_counter() - started,
            )
            yield st, rep
            self._observe(
                "step_accepted" if rep.converged else "step_rejected",
                dt=float(dt), iterations=int(rep.iterations), cuts=int(rep.cuts),
            )
            if not rep.converged:
                if self.verbose:
                    print(f"  step of {dt:.3g} s did not converge; stopping")
                break

    @staticmethod
    def _apply_control(model, control) -> None:
        """Set a control period's driving forces on the model.

        Wells are replaced object-for-object (they carry their own well index,
        perforations and geometry, so a control supplies *complete* wells, not
        just new targets).

        The well **count** may change: field schedules routinely drill a new
        producer or abandon one partway through. That used to be refused,
        because the state's per-well arrays are indexed positionally and would
        no longer line up. `_realign_well_state` rebuilds them, which is what
        makes it supportable.
        """
        if control is None:
            return
        if control.bc is not None:
            model.bc = control.bc
        if control.src is not None:
            model.src = control.src
        if control.W is not None and hasattr(model, "facility"):
            new_wells = list(control.W)
            model.facility.wells[:] = new_wells
            model.facility.within_limits = True
            # VFP table bindings live on each well and must follow a schedule
            # control replacement.  This is a no-op for ordinary wells.
            bind_vfp = getattr(model.facility, "_bind_vfp_tables", None)
            if bind_vfp is not None:
                bind_vfp()

    @staticmethod
    def _realign_well_state(state, old_wells, new_wells):
        """Rebuild the per-well arrays when a control changes the well set.

        Wells are matched **by name**, not by position. A schedule that
        abandons the second of three wells leaves the third in slot two, and
        carrying the arrays over positionally would hand the third well the
        second's rates and pressure - a plausible starting guess for the wrong
        well, which Newton then converges away from rather than failing on.

        A well that survives keeps its solution: that is the whole reason to
        match rather than re-initialise everything, since a converged rate is a
        far better start than a default. A new well starts from the field's
        mean pressure and zero rates; `_seed_injector_control_state` then
        refines an injector's rates from its prescribed composition.

        Names are required for this. Without them there is no identity to match
        on, and the only honest fallback when the count has changed is to
        re-initialise every well rather than guess at a correspondence.

        Returns `(state, aligned_old_wells)`. The second is the old well set
        reordered into the *new* slots, with a shut stand-in wherever a well is
        brand new. `_seed_injector_control_state` consumes it and needs that
        shape: it pairs old with new by position, and a new well has to look
        like one that just turned on, which is exactly what a shut predecessor
        makes it look like.
        """
        if len(old_wells) == len(new_wells):
            return state, list(old_wells)

        per_well = [k for k in ("qWs", "qOs", "qGs", "bhp", "qTs")
                    if k in state]
        if not per_well:
            return state, list(old_wells)

        def name_of(well, index):
            found = getattr(well, "name", None)
            return str(found) if found else None

        old_names = [name_of(w, i) for i, w in enumerate(old_wells)]
        new_names = [name_of(w, i) for i, w in enumerate(new_wells)]
        matchable = (all(n is not None for n in old_names)
                     and all(n is not None for n in new_names)
                     and len(set(old_names)) == len(old_names))

        realigned = state.copy()
        for key in per_well:
            old_values = np.asarray(realigned[key], dtype=float).ravel()
            default = (float(np.mean(old_values)) if key == "bhp"
                       and old_values.size else 0.0)
            fresh = np.full(len(new_wells), default)
            if matchable:
                index = {name: i for i, name in enumerate(old_names)}
                for slot, name in enumerate(new_names):
                    if name in index and index[name] < old_values.size:
                        fresh[slot] = old_values[index[name]]
            realigned[key] = fresh

        # A well with no predecessor is presented as having been shut, so the
        # injector seeding sees a status change and supplies its rates.
        shut_stand_in = SimpleNamespace(status=False, type="", val=0.0,
                                        compi=(), name=None)
        if matchable:
            index = {name: i for i, name in enumerate(old_names)}
            aligned = [old_wells[index[name]] if name in index
                       else shut_stand_in for name in new_names]
        else:
            aligned = [shut_stand_in]*len(new_wells)
        return realigned, aligned

    @staticmethod
    def _seed_injector_control_state(state, old_wells, new_wells):
        """Seed a newly changed injector's surface-rate unknowns.

        Field schedules commonly open a high-rate gas or water injector in one
        control jump.  Carrying the preceding zero rate into Newton gives the
        well equations an unnecessarily violent initial residual.  Injector
        composition is prescribed, so its target supplies an exact and safe
        initial guess for the phase rates.  Producer phase splits are not
        prescribed and are intentionally left untouched.
        """
        if not old_wells or len(old_wells) != len(new_wells):
            return state
        names = ("qWs", "qOs", "qGs")
        if not all(name in state for name in names):
            return state
        seeded = state.copy()
        changed = False
        for index, (old, new) in enumerate(zip(old_wells, new_wells)):
            target = float(getattr(new, "val", 0.0))
            control = str(getattr(new, "type", "")).lower()
            old_signature = (
                bool(getattr(old, "status", False)),
                str(getattr(old, "type", "")).lower(),
                float(getattr(old, "val", 0.0)),
            )
            new_signature = (
                bool(getattr(new, "status", False)), control, target
            )
            if (old_signature == new_signature or not new_signature[0]
                    or target <= 0.0
                    or control not in {
                        "rate", "wrat", "orat", "grat", "lrat", "vrat",
                        "nrat", "resv_history",
                    }):
                continue
            compi = np.asarray(getattr(new, "compi", ()), dtype=float).ravel()
            if compi.size < len(names):
                compi = np.pad(compi, (0, len(names)-compi.size))
            for phase, name in enumerate(names):
                seeded[name][index] = target*compi[phase]
            changed = True
        return seeded if changed else state


if __name__ == "__main__":
    from pymrst.utils.adi import ad_exp, init_variables_adi

    class _Scalar:
        """A one-variable model, so the solver's own logic is what is tested.

        `residual(x) = x^2 - a` converges quadratically from a good start
        and needs damping from a bad one - which is what relaxation and
        line search are for.
        """

        def __init__(self, a=4.0):
            self.a = a

        def _residual(self, x):
            return x * x - self.a

        def get_equations(self, state0, state, dt):
            (x,) = init_variables_adi(state["x"])
            return [self._residual(x)]

        def get_convergence(self, eqs, dt):
            r = float(np.max(np.abs(value(eqs[0]))))
            return r < 1e-10, {"residual": r}

        def update_state(self, state, increment):
            new = state.copy()
            new["x"] = state["x"] + increment
            return new

    class _NoRoot(_Scalar):
        """residual(x) = exp(x) + 1, which has no root and never converges.

        The update is limited to a finite range, as every real model limits
        its increments. Without that, Newton runs x off to -1e8, exp(x)
        underflows to 0, and the Jacobian goes exactly singular - so what
        the solver would report is a failed linear solve rather than the
        honest non-convergence this test is about.
        """

        def _residual(self, x):
            return ad_exp(x) + 1.0

        def update_state(self, state, increment):
            new = state.copy()
            new["x"] = np.clip(state["x"] + increment, -20.0, 20.0)
            return new

    def test_newton_converges_quadratically():
        """Plain Newton finds sqrt(4) = 2 in a handful of iterations."""
        st, rep = NonLinearSolver().solve_ministep(
            _Scalar(4.0), State(x=np.array([1.0])), State(x=np.array([1.0])),
            1.0)
        assert rep.converged and rep.iterations <= 8
        assert abs(st["x"][0] - 2.0) < 1e-10
        print(f"ok test_newton_converges_quadratically ({rep.iterations} its)")

    def test_report_records_residual_history():
        """The report carries the residual of every iteration, falling."""
        _, rep = NonLinearSolver().solve_ministep(
            _Scalar(9.0), State(x=np.array([1.0])), State(x=np.array([1.0])),
            1.0)
        res = [r["residual"] for r in rep.residuals]
        assert len(res) == rep.iterations + 1
        assert res[-1] < res[0] and res[-1] < 1e-10
        print("ok test_report_records_residual_history")

    def test_failure_is_reported_not_hidden():
        """A step that cannot converge reports converged=False.

        The whole point of the report: a solver that quietly returns an
        unconverged state is worse than one that fails loudly.
        """
        _, rep = NonLinearSolver(max_iterations=5).solve_ministep(
            _NoRoot(), State(x=np.array([1.0])), State(x=np.array([1.0])), 1.0)
        assert not rep.converged and rep.iterations == 5
        print("ok test_failure_is_reported_not_hidden")

    def test_cutting_gives_up_but_reports_the_cuts():
        """A hopeless step is cut the full number of times, then reported."""
        _, rep = NonLinearSolver(max_iterations=3,
                                 max_timestep_cuts=2).solve_timestep(
            _NoRoot(), State(x=np.array([1.0])), 1.0)
        assert not rep.converged and rep.cuts == 2
        print("ok test_cutting_gives_up_but_reports_the_cuts")

    def test_relaxation_and_line_search_still_converge():
        """Damped Newton reaches the same root from a far starting point."""
        for kwargs in ({"use_relaxation": True},
                       {"use_line_search": True},
                       {"use_relaxation": True, "use_line_search": True}):
            st, rep = NonLinearSolver(**kwargs).solve_ministep(
                _Scalar(4.0), State(x=np.array([10.0])),
                State(x=np.array([10.0])), 1.0)
            assert rep.converged, f"failed with {kwargs}"
            assert abs(st["x"][0] - 2.0) < 1e-8
        print("ok test_relaxation_and_line_search_still_converge")

    def test_line_search_stays_off_when_not_struggling():
        """The SPE1 regression this fix targets: a smoothly-converging solve
        never sets `convergence_issues`, so line search - though enabled -
        is never even *called*. `report.line_searches` alone cannot prove
        this: it counts backoff halvings, and a well-conditioned model can
        be called and immediately accept the full step with zero halvings,
        which would read identically to never being called at all - so this
        spies on `_line_search` itself, the call PyMRST actually pays for
        (`model.get_equations` inside it), not its return value."""
        calls = []
        solver = NonLinearSolver(use_line_search=True)
        original = solver._line_search

        def spy(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        solver._line_search = spy
        st, rep = solver.solve_ministep(
            _Scalar(4.0), State(x=np.array([3.0])),
            State(x=np.array([3.0])), 1.0)
        assert rep.converged
        assert len(calls) == 0, (
            f"_line_search was called {len(calls)} times on a well-behaved solve")
        print("ok test_line_search_stays_off_when_not_struggling")

    def test_convergence_issues_activates_on_synthetic_oscillation():
        """Direct, deterministic test of the detection itself - a residual
        history built to alternate every iteration must set
        `convergence_issues`, independent of whether any particular model
        happens to organically produce one (`x^2 - a`'s own Newton
        iteration is globally monotone and never does)."""
        solver = NonLinearSolver(oscillation_threshold=1.0)
        solver.convergence_issues = False
        residuals = [{"residual": v} for v in (10.0, 1.0, 10.0, 1.0, 10.0)]
        for i in range(len(residuals)):
            solver._update_convergence_issues(residuals[:i + 1], i, _Scalar(4.0))
        assert solver.convergence_issues, "oscillating residual did not activate the gate"
        print("ok test_convergence_issues_activates_on_synthetic_oscillation")

    def test_convergence_issues_resets_between_ministeps():
        """`convergence_issues` must not leak from one ministep attempt into
        the next - MRST resets it at both ends of `solveMinistep`
        (NonLinearSolver.m:358, 447)."""
        solver = NonLinearSolver(use_line_search=True)
        solver.convergence_issues = True  # simulate a prior struggling ministep
        st, rep = solver.solve_ministep(
            _Scalar(4.0), State(x=np.array([2.0])),
            State(x=np.array([2.0])), 1.0)
        assert rep.converged
        assert solver.convergence_issues is False, (
            "convergence_issues leaked past the ministep that should reset it")
        print("ok test_convergence_issues_resets_between_ministeps")

    def test_always_use_stabilization_actually_reaches_line_search():
        """A spy proves `is_struggling` really gates the call to
        `_line_search`, not just that `convergence_issues` can be set in
        isolation - `always_use_stabilization` forces it on from the first
        iteration, matching MRST's own debug switch."""
        calls = []
        solver = NonLinearSolver(use_line_search=True,
                                 always_use_stabilization=True)
        original = solver._line_search

        def spy(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        solver._line_search = spy
        st, rep = solver.solve_ministep(
            _Scalar(4.0), State(x=np.array([10.0])),
            State(x=np.array([10.0])), 1.0)
        assert rep.converged
        assert len(calls) > 0, (
            "always_use_stabilization did not reach _line_search")
        print("ok test_always_use_stabilization_actually_reaches_line_search "
             f"({len(calls)} calls)")

    def test_schedule_runs_every_step():
        """simulate_schedule returns one state per step and a clean report."""
        states, rep = NonLinearSolver().simulate_schedule(
            _Scalar(4.0), State(x=np.array([1.0])), [1.0, 1.0, 1.0])
        assert len(states) == 3 and rep.converged
        assert rep.total_iterations > 0 and rep.total_cuts == 0
        assert all(abs(s["x"][0] - 2.0) < 1e-10 for s in states)
        print("ok test_schedule_runs_every_step")

    def test_adaptive_ministeps_sum_to_the_report_step():
        """A selector may cross a report step however it likes - but it must
        land on it exactly. A schedule that quietly simulates a different
        length of time is worse than no schedule at all."""
        from pymrst.modules.ad_core.timesteps import (
            IterationCountTimeStepSelectorAD as IterationCountTimeStepSelector,
        )

        solver = NonLinearSolver(
            timestep_selector=IterationCountTimeStepSelector(
                target_iteration_count=3, max_timestep=0.3))
        st, rep = solver.solve_timestep(_Scalar(4.0), State(x=np.array([1.0])),
                                        1.0)
        assert rep.converged
        assert len(rep.ministeps) > 1, "max_timestep=0.3 forces subdivision"
        assert abs(sum(rep.ministeps) - 1.0) < 1e-12, sum(rep.ministeps)
        assert abs(st["x"][0] - 2.0) < 1e-10
        print(f"ok test_adaptive_ministeps_sum_to_the_report_step "
              f"({len(rep.ministeps)} ministeps summing to "
              f"{sum(rep.ministeps):.12g})")

    def test_a_selector_gives_up_at_the_floor_rather_than_grinding():
        """A hopeless step must still fail, not subdivide for ever."""
        from pymrst.modules.ad_core.timesteps import SimpleTimeStepSelectorAD as SimpleTimeStepSelector

        solver = NonLinearSolver(max_iterations=3, max_timestep_cuts=3,
                                 timestep_selector=SimpleTimeStepSelector())
        _, rep = solver.solve_timestep(_NoRoot(), State(x=np.array([1.0])), 1.0)
        assert not rep.converged and rep.cuts > 0
        print(f"ok test_a_selector_gives_up_at_the_floor_rather_than_grinding "
              f"({rep.cuts} cuts, then reported failure)")

    test_newton_converges_quadratically()
    test_report_records_residual_history()
    test_failure_is_reported_not_hidden()
    test_cutting_gives_up_but_reports_the_cuts()
    test_relaxation_and_line_search_still_converge()
    test_line_search_stays_off_when_not_struggling()
    test_convergence_issues_activates_on_synthetic_oscillation()
    test_convergence_issues_resets_between_ministeps()
    test_always_use_stabilization_actually_reaches_line_search()
    test_schedule_runs_every_step()
    test_adaptive_ministeps_sum_to_the_report_step()
    test_a_selector_gives_up_at_the_floor_rather_than_grinding()
    print("\nAll tests passed!")
