"""Fixed-stress mechanics/fluid splitting model."""

from __future__ import annotations

from typing import Any, Iterable
import copy

from pymrst.modules.ad_mechanics.models.poromechanics import FixedStressSplitModel


class MechFluidFixedStressSplitModel(FixedStressSplitModel):
    """Native MRST-layout owner for the operational fixed-stress solver."""

    def compute_mech_term(self, state: Any) -> dict[str, Any]:
        """Return ``sTerm`` and ``pTerm`` for constant-total-stress flow."""
        for field in ("xd", "pressure"):
            if field not in state:
                raise KeyError(f"fixed-stress state requires {field!r}")
        s_term, p_term = self.mech.fixed_stress_terms(state["xd"], state["pressure"])
        return {"sTerm": s_term, "pTerm": p_term}

    computeMechTerm = compute_mech_term

    def step_function(
        self,
        state: Any,
        state0: Any,
        dt: float,
        driving_forces: Any = None,
        linsolve: Any = None,
        nonlinsolve: Any = None,
        iteration: int = 0,
        **kwargs: Any,
    ):
        """MRST step-function adapter around PyMRST's fixed-stress step."""
        original = state.copy() if hasattr(state, "copy") else copy.copy(state)
        accepted, converged, outer_iterations = self.step(state0, dt)
        details = getattr(self, "last_split_report", {})
        if not converged:
            # A nonlinear timestep controller may retry with a smaller dt.
            # Never leak a rejected trial solution into that retry.
            accepted = original
        residuals = [entry.get("mechanics_residual")
                     for entry in details.get("outer", [])]
        residuals = [value for value in residuals if value is not None]
        failure_msg = details.get("failure", "")
        if not converged and not failure_msg:
            failure_msg = "fixed-stress iteration did not converge"
        report = {
            "Converged": bool(converged),
            "Failure": not bool(converged),
            "FailureMsg": "" if converged else failure_msg,
            "Iterations": int(details.get("iterations", outer_iterations)),
            "Residuals": residuals,
            "ResidualsConverged": bool(converged),
            "SplittingIterations": details.get("outer", []),
        }
        return accepted, report

    stepFunction = step_function


def simulate_fixed_stress_schedule(
    model: Any,
    state0: Any,
    timesteps: Iterable[float],
    *,
    max_timestep_cuts: int = 6,
    min_timestep: float = 0.0,
    growth_factor: float = 2.0,
    observer=None,
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Advance report steps with rollback-safe fixed-stress cutbacks.

    ``observer`` is an optional, dependency-free observation hook matching the
    one ``NonLinearSolver`` carries. This loop owns its own Newton iteration and
    rollback rather than delegating, so the canonical hook never reaches it and
    a schedule run here produced no events at all. Event names match the
    canonical solver's. It receives copied scalars only, its exceptions are
    swallowed, and with the default ``None`` this function behaves exactly as
    before.

    Every attempted substep is retained in ``reports[*]['Attempts']``.  A
    failed candidate is discarded, its timestep is halved, and the retry is
    started from a deep copy of the last accepted state.  Successful cut
    substeps may grow again, but never beyond the remaining report interval.
    """
    if max_timestep_cuts < 0:
        raise ValueError("max_timestep_cuts must be non-negative")
    if not 0.0 < growth_factor:
        raise ValueError("growth_factor must be positive")

    state = copy.deepcopy(state0)
    states: list[Any] = []
    reports: list[dict[str, Any]] = []
    def _observe(event, **payload):
        if observer is None:
            return
        try:
            observer(event, payload)
        except Exception:
            pass

    for report_index, report_dt in enumerate(timesteps):
        report_dt = float(report_dt)
        if report_dt <= 0.0:
            raise ValueError("all report timesteps must be positive")
        elapsed = 0.0
        trial_dt = report_dt
        cuts = 0
        attempts: list[dict[str, Any]] = []
        accepted_reports: list[dict[str, Any]] = []
        tolerance = max(1.0, report_dt) * 1e-12

        _observe("report_step_started", report_step=int(report_index),
                 dt=float(report_dt))
        while elapsed < report_dt - tolerance:
            trial_dt = min(trial_dt, report_dt - elapsed)
            start = copy.deepcopy(state)
            _observe("ministep_started", report_step=int(report_index),
                     dt=float(trial_dt))
            candidate, converged, outer_iterations = model.step(start, trial_dt)
            split = copy.deepcopy(getattr(model, "last_split_report", {}))
            attempt = {
                "dt": trial_dt,
                "Converged": bool(converged),
                "Iterations": int(outer_iterations),
                "Splitting": split,
            }
            attempts.append(attempt)
            _observe("step_accepted" if converged else "step_rejected",
                     report_step=int(report_index), dt=float(trial_dt),
                     iterations=int(outer_iterations))
            if converged:
                state = candidate
                elapsed += trial_dt
                accepted_reports.append(attempt)
                if elapsed < report_dt - tolerance:
                    trial_dt = min(
                        trial_dt * growth_factor, report_dt - elapsed
                    )
                continue

            # Candidate is intentionally ignored: retry from the last
            # accepted state, which has not been passed to the model directly.
            cuts += 1
            if cuts > max_timestep_cuts:
                raise RuntimeError(
                    f"report step {report_index + 1} failed after "
                    f"{max_timestep_cuts} timestep cuts"
                )
            trial_dt *= 0.5
            if trial_dt < float(min_timestep):
                raise RuntimeError(
                    f"report step {report_index + 1} requires dt={trial_dt:g}, "
                    f"below min_timestep={float(min_timestep):g}"
                )

        states.append(copy.deepcopy(state))
        reports.append({
            "ReportStep": report_index + 1,
            "RequestedTime": report_dt,
            "AcceptedTime": elapsed,
            "Converged": True,
            "Cuts": cuts,
            "Attempts": attempts,
            "AcceptedSubsteps": accepted_reports,
            "Iterations": sum(item["Iterations"] for item in attempts),
        })
    return states, reports


__all__ = [
    "FixedStressSplitModel",
    "MechFluidFixedStressSplitModel",
    "simulate_fixed_stress_schedule",
]
