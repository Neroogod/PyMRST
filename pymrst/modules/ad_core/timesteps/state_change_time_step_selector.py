"""MRST ``ad-core/timesteps/StateChangeTimeStepSelector.m``: time steps that
bound the change of chosen state properties per step (Cao, 2002), on top of
the iteration-count rule.

For each of `targetProps` (read with `model.getProp`, several per property if
it holds a list) the largest absolute and relative change over the last step
is compared with `targetChangeAbs` and `targetChangeRel` (infinite by
default), and the previous step is scaled by
`(1 + w) target/(change + w target)` with `w = relaxFactor`; the smallest of
these and the iteration-count step is taken. Without a previous state the
current one is remembered for the next step.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

from pymrst.modules.ad_core.timesteps.iteration_count_time_step_selector import (
    IterationCountTimeStepSelector,
)


class StateChangeTimeStepSelector(IterationCountTimeStepSelector):
    """MRST's `StateChangeTimeStepSelector`."""

    _OPTIONS = ("targetProps", "targetChangeRel", "targetChangeAbs", "relaxFactor")

    def __init__(self, **options: Any) -> None:
        self.targetProps = []
        self.targetChangeRel = []
        self.targetChangeAbs = []
        self.relaxFactor = 1.0
        self.previousState = None
        self.previousDt = math.inf
        super().__init__(**options)
        self.targetProps = list(self.targetProps)
        nprops = len(self.targetProps)
        if len(self.targetChangeRel) == 0:
            self.targetChangeRel = [math.inf]*nprops
        if len(self.targetChangeAbs) == 0:
            self.targetChangeAbs = [math.inf]*nprops
        self.previousDt = math.inf
        assert len(self.targetChangeRel) == nprops and len(self.targetChangeAbs) == nprops, (
            "targetProps must be equal in length to the targets!")

    def compute_timestep(self, dt: float, dt_prev: Optional[float], model: Any, solver: Any,
                         state_prev: Any, state_curr: Any, forces: Any = None) -> float:
        from pymrst.utils.adi import value
        dt0 = dt
        dt = super().compute_timestep(dt, dt_prev, model, solver, state_prev, state_curr, forces)
        if state_prev is None:
            if self.previousState is None:
                self.previousState = state_curr
                self.previousDt = dt0
                return dt
            state_prev = self.previousState
            dt_prev = self.previousDt
        self.previousState = state_curr
        self.previousDt = dt0
        dt_next = math.inf
        for i, fn in enumerate(self.targetProps):
            curr = model.get_prop(state_curr, fn)
            prev = model.get_prop(state_prev, fn)
            curr = curr if isinstance(curr, (list, tuple)) else [curr]
            prev = prev if isinstance(prev, (list, tuple)) else [prev]
            assert len(curr) == len(prev), "Mismatch in property size."
            for jj, (c, p) in enumerate(zip(curr, prev)):
                c = _as_array(c, value)
                p = _as_array(p, value)
                change_abs = np.abs(c - p)
                with np.errstate(divide="ignore", invalid="ignore"):
                    change_rel = change_abs/np.abs(p)
                change_rel[np.isnan(change_rel)] = 0
                max_abs = float(np.max(change_abs)) if change_abs.size else -math.inf
                max_rel = float(np.max(change_rel)) if change_rel.size else -math.inf
                w = self.relaxFactor
                target_abs = self.targetChangeAbs[i]
                target_rel = self.targetChangeRel[i]
                with np.errstate(divide="ignore", invalid="ignore"):
                    f_abs = ((1 + w)*target_abs)/(max_abs + w*target_abs)
                    f_rel = ((1 + w)*target_rel)/(max_rel + w*target_rel)
                if math.isnan(f_rel):
                    f_rel = math.inf
                # MATLAB's min skips NaN.
                dt_next = float(np.nanmin([dt_next, _times(dt_prev, f_rel), _times(dt_prev, f_abs)]))
                if self.verbose:
                    sname = "" if len(curr) == 1 else f" ({jj + 1})"
                    print(f"Property '{fn}{sname}': Rel change {max_rel:1.2f} [{target_rel:1.2f}], "
                          f"abs change: {max_abs:1.2f} [{target_abs:1.2f}]-> adjustment of "
                          f"{min(f_rel, f_abs):1.2f}")
            dt = min(dt, dt_next)
        return dt

    def reset(self) -> None:
        super().reset()
        self.previousState = None

    computeTimestep = compute_timestep


def _as_array(x: Any, value: Any) -> np.ndarray:
    if isinstance(x, (list, tuple)):
        return np.column_stack([np.ravel(value(v)) for v in x])
    return np.asarray(value(x), dtype=float)


def _times(dt_prev: Optional[float], factor: float) -> float:
    """`dt_prev*factor`, NaN where MATLAB's would be empty (no previous step)
    or NaN (0*Inf), both of which its `min` skips."""
    if dt_prev is None:
        return math.nan
    with np.errstate(invalid="ignore"):
        return float(np.float64(dt_prev)*factor)


__all__ = ["StateChangeTimeStepSelector"]


if __name__ == "__main__":
    from types import SimpleNamespace

    MODEL = SimpleNamespace(get_prop=lambda state, name: state[name])
    SOLVER = SimpleNamespace(maxIterations=25)

    def test_the_change_bounds_the_step():
        sel = StateChangeTimeStepSelector(targetProps=["s"], targetChangeAbs=[0.2],
                                          targetIterationCount=math.inf)
        s0, s1 = {"s": np.array([0.1, 0.5])}, {"s": np.array([0.3, 0.9])}
        sel.store_timestep({"Converged": True, "Iterations": 3, "Timestep": 10.0})
        dt = sel.compute_timestep(10.0, 10.0, MODEL, SOLVER, s0, s1)
        # max change 0.4, target 0.2, w = 1: factor 0.4/0.6
        assert abs(dt - 10.0*0.4/0.6) < 1e-12
        print("ok test_the_change_bounds_the_step")

    def test_first_call_remembers_the_state():
        sel = StateChangeTimeStepSelector(targetProps=["s"], targetChangeAbs=[0.2])
        assert sel.compute_timestep(5.0, None, MODEL, SOLVER, None, {"s": np.zeros(2)}) == 5.0
        assert sel.previousState is not None and sel.previousDt == 5.0
        sel.reset()
        assert sel.previousState is None
        print("ok test_first_call_remembers_the_state")

    test_the_change_bounds_the_step()
    test_first_call_remembers_the_state()
    print("\nAll tests passed!")
