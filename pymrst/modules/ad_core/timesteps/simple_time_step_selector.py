"""MRST ``ad-core/timesteps/SimpleTimeStepSelector.m``: the base time-step
selector.

It does no adapting of its own (`computeTimestep` returns `dt`), but holds
what every selector shares: the history of ministep reports (at most
`maxHistoryLength`), hard limits `minTimestep`/`maxTimestep`, a bound on the
change between ministeps (`minRelativeAdjustment`..`maxRelativeAdjustment`,
not applied at the start of a control step), the ramp-up of the first step
after the controls change (`firstRampupStep`, `firstRampupStepRelative`, on
the first step, or on every change with `resetOnControlsChanged`), and the
cut on failure (`dt*minRelativeAdjustment`). A control is recognised as new
by its `controlId`.

Options take MRST's names; the snake-case spelling of each is accepted and
answered to as well. Reports are read as MRST's structs (`Iterations`,
`Converged`, `Timestep`) whether given as dicts or objects.
"""

from __future__ import annotations

import math
import re
from typing import Any, List, Optional


def _snake(name: str) -> str:
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name).lower()


def report_field(report: Any, name: str, default: Any = None) -> Any:
    """`report.(name)` for an MRST report given as a dict or an object."""
    if isinstance(report, dict):
        return report.get(name, default)
    if hasattr(report, name):
        return getattr(report, name)
    if hasattr(report, "get"):
        try:
            return report.get(name, default)
        except TypeError:
            pass
    return getattr(report, _snake(name), default)


def _control_id(control: Any) -> Any:
    return control.get("controlId") if isinstance(control, dict) else getattr(control, "controlId", None)


class SimpleTimeStepSelector:
    """MRST's `SimpleTimeStepSelector`."""

    _OPTIONS = ("maxTimestep", "minTimestep", "verbose", "maxHistoryLength", "firstRampupStep",
                "firstRampupStepRelative", "maxRelativeAdjustment", "minRelativeAdjustment",
                "resetOnControlsChanged")

    def __init__(self, **options: Any) -> None:
        self.maxTimestep = math.inf
        self.minTimestep = 0.0
        self.maxHistoryLength = 50
        self.maxRelativeAdjustment = 2.0
        self.minRelativeAdjustment = 0.5
        self.firstRampupStep = math.inf
        self.firstRampupStepRelative = 1.0
        self.resetOnControlsChanged = False
        self.verbose = False
        self._merge_options(options)
        assert 0 < self.firstRampupStepRelative <= 1
        self.reset()

    def _merge_options(self, options: dict) -> None:
        """MRST's `merge_options` on the selector, case-insensitive, with the
        snake-case spelling of each option accepted."""
        names = {}
        for cls in type(self).__mro__:
            for name in getattr(cls, "_OPTIONS", ()):
                names[name.lower()] = name
                names[_snake(name)] = name
        for key, value in options.items():
            name = names.get(key.lower()) or names.get(key)
            if name is None:
                raise TypeError(f"{type(self).__name__} has no option {key!r}")
            setattr(self, name, value)

    def __getattr__(self, name: str) -> Any:
        # The snake-case spelling of a property, for this port's callers.
        if name.startswith("_"):
            raise AttributeError(name)
        camel = re.sub(r"_([a-z])", lambda m: m.group(1).upper(), name)
        if camel != name and camel in self.__dict__:
            return self.__dict__[camel]
        raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        camel = re.sub(r"_([a-z])", lambda m: m.group(1).upper(), name)
        if camel != name and not name.startswith("_") and camel in self.__dict__:
            name = camel
        object.__setattr__(self, name, value)

    def reset(self) -> None:
        self.history: List[Any] = []
        self.previousControl = None
        self.isStartOfCtrlStep = True
        self.isFirstStep = True
        self.controlsChanged = True
        self.stepLimitedByHardLimits = True

    def store_timestep(self, report: Any) -> None:
        self.history.append(report)
        n = self.maxHistoryLength
        if len(self.history) > n:
            self.history = self.history[-(n + 1):]

    def pick_timestep(self, dt_prev: Optional[float], dt: float, model: Any, solver: Any,
                      state_prev: Any, state_curr: Any, forces: Any = None) -> float:
        if self.controlsChanged and (self.resetOnControlsChanged or self.isFirstStep):
            dt = min(dt, self.firstRampupStepRelative*dt)
            dt = min(dt, self.firstRampupStep)
            self.stepLimitedByHardLimits = True
        dt0 = dt
        dt_new = self.compute_timestep(dt, dt_prev, model, solver, state_prev, state_curr, forces)
        change = dt_new/dt
        if not self.isStartOfCtrlStep:
            change = min(change, self.maxRelativeAdjustment)
            change = max(change, self.minRelativeAdjustment)
        dt = dt*change
        dt = min(self.maxTimestep, dt)
        dt = max(self.minTimestep, dt)
        if self.verbose and dt0 != dt:
            from pymrst.utils.format_time_range import format_time_range
            if self.history:
                print(f"Prev # its: {report_field(self.history[-1], 'Iterations')} -> ", end="")
            print(f"Suggested timestep modification of a factor {dt/dt0:1.2f}. "
                  f"dT: {format_time_range(dt0, 2)} -> {format_time_range(dt, 2)}")
        self.stepLimitedByHardLimits = dt != dt_new
        self.isStartOfCtrlStep = False
        self.isFirstStep = False
        self.controlsChanged = False
        return dt

    def new_control_step(self, control: Any) -> None:
        self.isStartOfCtrlStep = True
        prev = self.previousControl
        if prev is None or _control_id(prev) != _control_id(control):
            if self.resetOnControlsChanged:
                self.reset()
            self.controlsChanged = True
            self.previousControl = control
        else:
            self.controlsChanged = False

    def cut_timestep(self, dt_prev: Optional[float], dt: float, model: Any, solver: Any,
                     state_prev: Any, state_curr: Any, forces: Any = None) -> float:
        return dt*self.minRelativeAdjustment

    def compute_timestep(self, dt: float, dt_prev: Optional[float], model: Any, solver: Any,
                         state_prev: Any, state_curr: Any, forces: Any = None) -> float:
        return dt

    storeTimestep = store_timestep
    pickTimestep = pick_timestep
    newControlStep = new_control_step
    cutTimestep = cut_timestep
    computeTimestep = compute_timestep


__all__ = ["SimpleTimeStepSelector", "report_field"]


if __name__ == "__main__":
    def test_ramp_up_limits_and_leash():
        sel = SimpleTimeStepSelector(firstRampupStepRelative=0.1, maxTimestep=50.0)
        sel.new_control_step({"controlId": 1})
        assert sel.pick_timestep(None, 100.0, None, None, None, None) == 10.0
        assert sel.pick_timestep(10.0, 100.0, None, None, None, None) == 50.0
        sel.minTimestep = 80.0
        assert sel.pickTimestep(10.0, 60.0, None, None, None, None) == 80.0
        print("ok test_ramp_up_limits_and_leash")

    def test_snake_case_options_and_controls():
        sel = SimpleTimeStepSelector(first_rampup_step=5.0, resetOnControlsChanged=True)
        assert sel.firstRampupStep == 5.0 and sel.first_rampup_step == 5.0
        sel.first_rampup_step = 7.0
        assert sel.firstRampupStep == 7.0
        sel.new_control_step({"controlId": 1})
        sel.pick_timestep(None, 10.0, None, None, None, None)
        sel.new_control_step({"controlId": 1})
        assert not sel.controlsChanged
        sel.new_control_step({"controlId": 2})
        assert sel.controlsChanged and sel.cut_timestep(None, 8.0, None, None, None, None) == 4.0
        print("ok test_snake_case_options_and_controls")

    def test_history_is_bounded_like_mrst():
        sel = SimpleTimeStepSelector(maxHistoryLength=3)
        for k in range(10):
            sel.store_timestep({"Iterations": k})
        assert [r["Iterations"] for r in sel.history] == [6, 7, 8, 9]
        print("ok test_history_is_bounded_like_mrst")

    test_ramp_up_limits_and_leash()
    test_snake_case_options_and_controls()
    test_history_is_bounded_like_mrst()
    print("\nAll tests passed!")
