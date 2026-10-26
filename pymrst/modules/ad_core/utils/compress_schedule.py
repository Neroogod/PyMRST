"""MRST ``ad-core/utils/compressSchedule.m``: each run of consecutive steps
under one control merged into a single step of their summed length."""

import copy

import numpy as np

from pymrst.modules.ad_core.utils.schedule_types import Schedule


def compress_schedule(schedule: Schedule) -> Schedule:
    """MRST `scheduleCompressed = compressSchedule(schedule)`."""
    nstep = schedule.step_val.size
    newsteps = np.full(nstep, np.nan)
    newctrls = np.full(nstep, np.nan)
    oldctrl = schedule.step_control.astype(float)
    oldvals = schedule.step_val
    finished = nstep == 0
    i = stepno = 0
    while not finished:
        ctrl = oldctrl[i]
        later = np.flatnonzero((oldctrl != ctrl) & ~np.isnan(oldctrl))
        if later.size == 0:
            nextctrl = nstep
            finished = True
        else:
            nextctrl = int(later[0])
        newsteps[stepno] = oldvals[i:nextctrl].sum()
        newctrls[stepno] = ctrl
        oldctrl[i:nextctrl] = np.nan
        i = nextctrl
        stepno += 1
    out = copy.copy(schedule)
    out.step_val = newsteps[~np.isnan(newsteps)]
    out.step_control = newctrls[~np.isnan(newctrls)].astype(np.int64)
    return out


compressSchedule = compress_schedule
__all__ = ["compressSchedule", "compress_schedule"]


if __name__ == "__main__":
    from pymrst.modules.ad_core.utils.schedule_types import Control

    def test_runs_are_merged_and_order_kept():
        s = Schedule(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), np.array([0, 0, 1, 1, 0]),
                     [Control(), Control()])
        c = compress_schedule(s)
        assert np.array_equal(c.step_val, [3.0, 7.0, 5.0]) and np.array_equal(c.step_control, [0, 1, 0])
        assert c.controls is s.controls and s.step_val.size == 5
        print("ok test_runs_are_merged_and_order_kept")

    test_runs_are_merged_and_order_kept()
    print("\nAll tests passed!")
