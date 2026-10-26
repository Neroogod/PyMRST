"""MRST ``ad-core/utils/refineSchedule.m``: a schedule's time grid merged with
a second one.

The report times of `schedule` and the times of `new_time_steps` (both
starting at `init_time`) are merged and sorted; a new time within one second
of a schedule time is dropped. Each resulting step takes the control of the
schedule step it ends in, and is marked as a report step (`repStep`). The
schedule's controls are kept. Step controls are 0-based.
"""

from typing import Any

import numpy as np

from pymrst.modules.ad_core.utils.schedule_types import Schedule

EPSI = 1.0  # one second


def refine_schedule(init_time: float, new_time_steps: Any, schedule: Schedule) -> Schedule:
    """MRST `new_schedule = refineSchedule(init_time, new_time_steps, schedule)`."""
    ref_steps = np.asarray(schedule.step_val, dtype=float).ravel()
    ref_control = np.asarray(schedule.step_control, dtype=np.int64).ravel()
    new_time_steps = np.asarray(new_time_steps, dtype=float).ravel()
    t_new = init_time + np.concatenate([[0.0], np.cumsum(new_time_steps)])
    t_ref = init_time + np.concatenate([[0.0], np.cumsum(ref_steps)])
    cum_time = np.vstack([np.column_stack([t_new, np.ones(t_new.size)]),
                          np.column_stack([t_ref, 2*np.ones(t_ref.size)])])
    cum_time = cum_time[np.lexsort((cum_time[:, 1], cum_time[:, 0]))]
    n = cum_time.shape[0]
    remove = []
    for i in range(n):
        if cum_time[i, 1] == 2:
            t = cum_time[i, 0]
            if i > 0 and cum_time[i - 1, 1] == 1 and abs(cum_time[i - 1, 0] - t) < EPSI:
                remove.append(i - 1)
            if i < n - 1 and cum_time[i + 1, 1] == 1 and abs(cum_time[i + 1, 0] - t) < EPSI:
                remove.append(i + 1)
    keep = np.ones(n, dtype=bool)
    keep[remove] = False
    steps = np.diff(cum_time[keep, 0])
    t = init_time
    ref_time = init_time + ref_steps[0]
    ref_ind = 0
    control = np.zeros(steps.size, dtype=np.int64)
    for ind, step in enumerate(steps):
        t = t + step
        while t - ref_time > EPSI:
            ref_ind += 1
            ref_time = ref_time + ref_steps[ref_ind]
        control[ind] = ref_control[ref_ind]
    new = Schedule(step_val=steps, step_control=control, controls=list(schedule.controls))
    for name in ("W", "type"):
        if hasattr(schedule, name):
            setattr(new, name, getattr(schedule, name))
    new.repStep = np.ones(steps.size, dtype=bool)
    return new


refineSchedule = refine_schedule
__all__ = ["refineSchedule", "refine_schedule"]


if __name__ == "__main__":
    from pymrst.modules.ad_core.utils.schedule_types import Control

    def test_two_time_grids_merge():
        s = Schedule(np.array([10.0, 10.0]), np.array([0, 1]), [Control(), Control()])
        r = refine_schedule(0.0, np.array([5.0, 10.0]), s)
        # times 0, 5, 10(old), 15, 20 -> the new 15 within a second of nothing; steps 5,5,5,5
        assert np.allclose(r.step_val, [5, 5, 5, 5]) and np.array_equal(r.step_control, [0, 0, 1, 1])
        assert r.repStep.all() and r.controls == s.controls
        print("ok test_two_time_grids_merge")

    def test_a_new_time_within_a_second_of_an_old_one_is_dropped():
        s = Schedule(np.array([10.0]), np.array([0]), [Control()])
        r = refine_schedule(0.0, np.array([9.5]), s)
        assert np.allclose(r.step_val, [10.0])
        print("ok test_a_new_time_within_a_second_of_an_old_one_is_dropped")

    test_two_time_grids_merge()
    test_a_new_time_within_a_second_of_an_old_one_is_dropped()
    print("\nAll tests passed!")
