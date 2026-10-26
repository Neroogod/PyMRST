"""MRST ``ad-core/timesteps/IterationCountTimeStepSelector.m``: time steps
chosen to bring the Newton iteration count to `targetIterationCount`.

With the response `(its + iterationOffset)/(maxIterations + iterationOffset)`,
the next step is the last one scaled by target over response, or, when the
last two steps were both chosen freely and converged, a two-point update
from both. For a model whose step function is linear (a sequential model)
the transport solver's iterations are counted instead.
"""

from __future__ import annotations

from typing import Any, Optional

from pymrst.modules.ad_core.timesteps.simple_time_step_selector import (
    SimpleTimeStepSelector,
    report_field,
)


class IterationCountTimeStepSelector(SimpleTimeStepSelector):
    """MRST's `IterationCountTimeStepSelector`."""

    _OPTIONS = ("targetIterationCount", "iterationOffset")

    def __init__(self, **options: Any) -> None:
        self.targetIterationCount = 5
        self.iterationOffset = 5
        super().__init__(**options)

    def compute_timestep(self, dt: float, dt_prev: Optional[float], model: Any, solver: Any,
                         state_prev: Any, state_curr: Any, forces: Any = None) -> float:
        hist = self.history
        n_hist = len(hist)
        if n_hist == 0 or not report_field(hist[-1], "Converged") or (
                self.controlsChanged and self.resetOnControlsChanged):
            return dt
        if getattr(model, "stepFunctionIsLinear", False):
            if any(c.__name__ == "SequentialPressureTransportModel" for c in type(model).__mro__):
                def get_its(x):
                    nonlinear = report_field(x, "NonlinearReport")
                    return report_field(report_field(nonlinear[-1], "TransportSolver"),
                                        "Iterations")
            else:
                raise TypeError("Step function is linear, but I do not know how to calculate "
                                f"the iterations for models of type {type(model).__name__}")
        else:
            def get_its(x):
                return report_field(x, "Iterations")
        if n_hist > 1:
            restart = self.stepLimitedByHardLimits or not report_field(hist[-2], "Converged")
        else:
            restart = True
        max_its = _max_iterations(solver) + self.iterationOffset
        offset = self.iterationOffset
        tol = (self.targetIterationCount + offset)/max_its
        le1 = (get_its(hist[-1]) + offset)/max_its
        dt1 = report_field(hist[-1], "Timestep")
        if restart:
            return (tol/le1)*dt1
        le0 = (get_its(hist[-2]) + offset)/max_its
        dt0 = report_field(hist[-2], "Timestep")
        return (dt1/dt0)*(tol*le0/le1**2)*dt1

    computeTimestep = compute_timestep


def _max_iterations(solver: Any) -> float:
    for name in ("maxIterations", "max_iterations"):
        if hasattr(solver, name):
            return getattr(solver, name)
    raise AttributeError("the nonlinear solver carries no maxIterations")


__all__ = ["IterationCountTimeStepSelector"]


if __name__ == "__main__":
    from types import SimpleNamespace

    SOLVER = SimpleNamespace(maxIterations=25)

    def test_one_point_update_after_a_restart():
        sel = IterationCountTimeStepSelector(targetIterationCount=5)
        sel.store_timestep({"Converged": True, "Iterations": 10, "Timestep": 100.0})
        dt = sel.compute_timestep(100.0, None, None, SOLVER, None, None)
        assert abs(dt - (10/15)*100.0) < 1e-12
        print("ok test_one_point_update_after_a_restart")

    def test_two_point_update_when_steps_were_free():
        sel = IterationCountTimeStepSelector()
        sel.store_timestep({"Converged": True, "Iterations": 5, "Timestep": 50.0})
        sel.store_timestep({"Converged": True, "Iterations": 10, "Timestep": 100.0})
        sel.stepLimitedByHardLimits = False
        dt = sel.compute_timestep(100.0, None, None, SOLVER, None, None)
        le0, le1, tol = 10/30, 15/30, 10/30
        assert abs(dt - (100/50)*(tol*le0/le1**2)*100) < 1e-12
        print("ok test_two_point_update_when_steps_were_free")

    def test_a_failed_last_step_leaves_dt_alone():
        sel = IterationCountTimeStepSelector()
        sel.store_timestep({"Converged": False, "Iterations": 25, "Timestep": 10.0})
        assert sel.compute_timestep(7.0, None, None, SOLVER, None, None) == 7.0
        print("ok test_a_failed_last_step_leaves_dt_alone")

    test_one_point_update_after_a_restart()
    test_two_point_update_when_steps_were_free()
    test_a_failed_last_step_leaves_dt_alone()
    print("\nAll tests passed!")
