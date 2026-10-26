"""The schedule structure MRST's ad-core functions pass around:
`schedule.step.val`, `schedule.step.control` and `schedule.control`, as
`Schedule.step_val`, `Schedule.step_control` (0-based) and
`Schedule.controls`, a list of `Control`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence

import numpy as np


@dataclass
class Control:
    """One control period's driving forces (MRST's `schedule.control(i)`)."""

    W: Optional[Sequence[Any]] = None        # wells
    bc: Any = None                           # boundary conditions
    src: Any = None                          # sources


@dataclass
class Schedule:
    """A simulation schedule: timesteps grouped into control periods.

    `step_val[i]` is step `i`'s length and `step_control[i]` is the index into
    `controls` it runs under. Mirrors MRST's `schedule.step.val` /
    `schedule.step.control` / `schedule.control`, with 0-based indices.
    """

    step_val: np.ndarray
    step_control: np.ndarray
    controls: List[Control] = field(default_factory=list)

    def __post_init__(self):
        self.step_val = np.asarray(self.step_val, dtype=float).ravel()
        self.step_control = np.asarray(self.step_control, dtype=np.int64).ravel()
        if self.step_val.shape != self.step_control.shape:
            raise ValueError("step_val and step_control must have equal length")
        if self.step_control.size and (
                self.step_control.min() < 0
                or self.step_control.max() >= len(self.controls)):
            raise ValueError("a step references a control that does not exist")

    @property
    def n_steps(self) -> int:
        return self.step_val.size

    @property
    def total_time(self) -> float:
        return float(self.step_val.sum())

    def control_for(self, step: int) -> Control:
        """The control governing step `step`."""
        return self.controls[int(self.step_control[step])]


__all__ = ["Control", "Schedule"]


if __name__ == "__main__":
    def test_steps_must_point_at_existing_controls():
        s = Schedule(np.array([1.0, 2.0]), np.array([0, 1]), [Control(), Control(W=["w"])])
        assert s.n_steps == 2 and s.total_time == 3.0 and s.control_for(1).W == ["w"]
        try:
            Schedule(np.array([1.0]), np.array([1]), [Control()])
        except ValueError:
            print("ok test_steps_must_point_at_existing_controls")
            return
        raise AssertionError("expected a ValueError")

    test_steps_must_point_at_existing_controls()
    print("\nAll tests passed!")
