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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse.linalg as spla

from pymrst.modules.ad_core.statefunctions.state_function import State
from pymrst.utils.adi import combine_equations, value


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
        linear_solver=None,
        timestep_selector=None,
        verbose: bool = False,
    ):
        self.max_iterations = max_iterations
        self.max_timestep_cuts = max_timestep_cuts
        self.use_line_search = use_line_search
        self.use_relaxation = use_relaxation
        self.relaxation_min = relaxation_min
        self.relaxation_increment = relaxation_increment
        self.line_search_max = line_search_max
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

    # -- one Newton loop at a fixed dt --------------------------------------
    def solve_ministep(self, model, state0: State, state: State,
                       dt: float) -> Tuple[State, StepReport]:
        """Newton to convergence at this dt, or report failure."""
        report = StepReport(converged=False, iterations=0, dt=dt)
        st = state
        relax = 1.0
        prev_norm = np.inf

        for it in range(self.max_iterations):
            eqs = model.get_equations(state0, st, dt)
            converged, res = model.get_convergence(eqs, dt)
            report.residuals.append(res)
            report.iterations = it

            if converged:
                report.converged = True
                if hasattr(model, "after_convergence"):
                    st = model.after_convergence(st, state0, dt)
                return st, report

            norm = max(res.values())
            if self.use_relaxation:
                relax = self._update_relaxation(relax, norm, prev_norm)
            prev_norm = norm

            increment = self._solve_linear(eqs, model, st, dt)
            st_new = model.update_state(st, relax * increment)

            if self.use_line_search:
                st_new, n_ls = self._line_search(
                    model, state0, st, increment, relax, dt, norm
                )
                report.line_searches += n_ls

            st = st_new

        eqs = model.get_equations(state0, st, dt)
        converged, res = model.get_convergence(eqs, dt)
        report.residuals.append(res)
        report.converged = converged
        report.iterations = self.max_iterations
        if converged and hasattr(model, "after_convergence"):
            st = model.after_convergence(st, state0, dt)
        return st, report

    def _solve_linear(self, eqs, model=None, state=None,
                      dt: Optional[float] = None) -> np.ndarray:
        eq = combine_equations(*eqs)
        if self.linear_solver is not None:
            return self.linear_solver.solve(eq.jac[0], -eq.val, model, state,
                                            dt)
        return spla.spsolve(eq.jac[0].tocsc(), -eq.val)

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
    def solve_timestep(self, model, state0: State,
                       dt: float) -> Tuple[State, StepReport]:
        """Take a step of dt, cutting it in half on failure and re-trying.

        A cut step is taken as several sub-steps that sum to dt, so the
        schedule's report times are still honoured.
        """
        if self.timestep_selector is not None:
            return self._solve_adaptive(model, state0, dt)

        flow = getattr(model, "FlowDiscretization", None)
        if flow is not None:
            from pymrst.modules.ad_core.statefunctions.flux.flow_state import (
                ImplicitFlowStateBuilder,
            )
            builder = flow.get_flow_state_builder()
            if not isinstance(builder, ImplicitFlowStateBuilder):
                return self._solve_flow_discretized(model, state0, dt, flow)

        st, report = self.solve_ministep(model, state0, state0.copy(), dt)
        if report.converged:
            return st, report

        for cut in range(1, self.max_timestep_cuts + 1):
            n_sub = 2 ** cut
            dt_sub = dt / n_sub
            if self.verbose:
                print(f"  step failed; cutting to {n_sub} x {dt_sub:.3g} s")

            st = state0
            ok = True
            iters = 0
            for _ in range(n_sub):
                st_new, rep = self.solve_ministep(model, st, st.copy(), dt_sub)
                iters += rep.iterations
                if not rep.converged:
                    ok = False
                    break
                st = st_new
            if ok:
                return st, StepReport(converged=True, iterations=iters, dt=dt,
                                      cuts=cut)

        return st, StepReport(converged=False, iterations=report.iterations,
                              dt=dt, cuts=self.max_timestep_cuts)

    def _solve_flow_discretized(self, model, state0, dt, flow):
        """Cross one report step using the flow builder's CFL/ministep policy."""
        st = state0
        elapsed = 0.0
        total_iterations = cuts = 0
        ministeps: List[float] = []
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

            while True:
                guess = st.copy()
                flow, guess = flow.prepare_timestep(
                    model, guess, st, dt_try, forces
                )
                candidate, rep = self.solve_ministep(
                    model, st, guess, dt_try
                )
                total_iterations += rep.iterations
                if rep.converged:
                    st = candidate
                    elapsed += dt_try
                    ministeps.append(dt_try)
                    break
                cuts += 1
                dt_try *= 0.5
                if dt_try < floor:
                    return st, StepReport(
                        converged=False, iterations=total_iterations, dt=dt,
                        cuts=cuts, ministeps=ministeps,
                    )

        return st, StepReport(
            converged=True, iterations=total_iterations, dt=dt,
            cuts=cuts, ministeps=ministeps,
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

        while t_local < dt * (1.0 - 1e-12):
            pick = sel.cut_timestep if failed else sel.pick_timestep
            dt_sel = pick(dt_prev, dt_try, model, self, state_prev, st)

            remaining = dt - t_local
            is_final = dt_sel >= remaining * (1.0 - 1e-12)
            dt_try = remaining if is_final else dt_sel

            st_new, rep = self.solve_ministep(model, st, st.copy(), dt_try)
            total_its += rep.iterations

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
                                          cuts=cuts, ministeps=taken)
                if self.verbose:
                    print(f"  ministep of {dt_try:.3g} s failed; cutting")

        return st, StepReport(converged=True, iterations=total_its, dt=dt,
                              cuts=cuts, ministeps=taken)

    # -- a whole schedule ----------------------------------------------------
    def simulate_schedule(self, model, state0: State,
                          schedule) -> Tuple[List[State], SimulationReport]:
        """Run every step of a schedule; return the states and the report.

        `schedule` is either a bare sequence of timesteps - the wells are then
        whatever the model was built with, fixed for the run - or a structured
        `ad_core.utils.schedule.Schedule`, whose control periods change the
        wells / boundary / sources as time advances. The structured form is
        what lets a well open partway through a run, or a rate step down.
        """
        from pymrst.modules.ad_core.utils.schedule import Schedule

        if not isinstance(schedule, Schedule):
            return self._simulate_timesteps(model, state0,
                                            np.atleast_1d(schedule))

        states, report = [], SimulationReport()
        st = state0
        last_control = -1
        for i in range(schedule.n_steps):
            c = int(schedule.step_control[i])
            if c != last_control:              # a new control period begins
                self._apply_control(model, schedule.controls[c])
                last_control = c
            st, rep = self.solve_timestep(model, st, float(schedule.step_val[i]))
            report.steps.append(rep)
            states.append(st)
            if not rep.converged:
                if self.verbose:
                    print(f"  step {i} did not converge; stopping")
                break
        return states, report

    def _simulate_timesteps(self, model, state0, dts):
        states, report = [], SimulationReport()
        st = state0
        for dt in dts:
            st, rep = self.solve_timestep(model, st, float(dt))
            report.steps.append(rep)
            states.append(st)
            if not rep.converged:
                if self.verbose:
                    print(f"  step of {dt:.3g} s did not converge; stopping")
                break
        return states, report

    @staticmethod
    def _apply_control(model, control) -> None:
        """Set a control period's driving forces on the model.

        Wells are replaced object-for-object (they carry their own well index,
        perforations and geometry, so a control supplies *complete* wells, not
        just new targets). The well *count* may not change mid-run - the state's
        per-well arrays would no longer line up - and a change is refused with a
        clear message rather than a silent shape error.
        """
        if control is None:
            return
        if control.bc is not None:
            model.bc = control.bc
        if control.src is not None:
            model.src = control.src
        if control.W is not None and hasattr(model, "facility"):
            new_wells = list(control.W)
            if len(new_wells) != model.facility.n_wells:
                raise NotImplementedError(
                    "a control step changes the number of wells "
                    f"({model.facility.n_wells} -> {len(new_wells)}); "
                    "opening or shutting a well mid-schedule needs the well "
                    "state re-initialised, which is not yet supported"
                )
            model.facility.wells[:] = new_wells
            model.facility.within_limits = True


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
            IterationCountTimeStepSelector,
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
        from pymrst.modules.ad_core.timesteps import SimpleTimeStepSelector

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
    test_schedule_runs_every_step()
    test_adaptive_ministeps_sum_to_the_report_step()
    test_a_selector_gives_up_at_the_floor_rather_than_grinding()
    print("\nAll tests passed!")
