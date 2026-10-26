"""MRST ``ad-core/utils/filterSchedule.m``: the controls no step uses
removed, and the step controls renumbered (MATLAB's `unique` order)."""

import copy

import numpy as np

from pymrst.modules.ad_core.utils.schedule_types import Schedule


def filter_schedule(s: Schedule) -> Schedule:
    """MRST `s = filterSchedule(s)`."""
    subs, renum = np.unique(s.step_control, return_inverse=True)
    out = copy.copy(s)
    out.step_control = renum.astype(np.int64).ravel()
    out.controls = [s.controls[k] for k in subs]
    return out


filterSchedule = filter_schedule
__all__ = ["filterSchedule", "filter_schedule"]


if __name__ == "__main__":
    from pymrst.modules.ad_core.utils.schedule_types import Control

    def test_unused_controls_go_and_steps_are_renumbered():
        s = Schedule(np.ones(3), np.array([2, 0, 2]), [Control(W=[k]) for k in range(3)])
        f = filter_schedule(s)
        assert np.array_equal(f.step_control, [1, 0, 1]) and [c.W for c in f.controls] == [[0], [2]]
        assert len(s.controls) == 3
        print("ok test_unused_controls_go_and_steps_are_renumbered")

    test_unused_controls_go_and_steps_are_renumbered()
    print("\nAll tests passed!")
