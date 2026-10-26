"""MRST ``ad-core/utils/combineSchedules.m``: schedules run one after the
other. Each schedule's controls are appended and its step controls shifted
to point at them. With `makeConsistent` (on by default when a grid `G` is
given) the result is passed through `makeScheduleConsistent`."""

from typing import Any, Optional

import numpy as np

from pymrst.modules.ad_core.utils.schedule_types import Schedule


def combine_schedules(schedule: Schedule, *schedules: Schedule,
                      makeConsistent: Optional[bool] = None, G: Any = None) -> Schedule:
    """MRST `schedule = combineSchedules(s1, s2, ..., 'makeConsistent', b, 'G', G)`."""
    for s2 in schedules:
        schedule = _combine(schedule, s2)
    if makeConsistent is None:
        makeConsistent = G is not None
    if makeConsistent:
        from pymrst.modules.ad_core.utils.make_schedule_consistent import (
            make_schedule_consistent,
        )
        schedule = make_schedule_consistent(schedule, G=G)
    return schedule


def _combine(s: Schedule, s2: Schedule) -> Schedule:
    offset = len(s.controls)
    return Schedule(step_val=np.concatenate([s.step_val, s2.step_val]),
                    step_control=np.concatenate([s.step_control, s2.step_control + offset]),
                    controls=list(s.controls) + list(s2.controls))


combineSchedules = combine_schedules
__all__ = ["combineSchedules", "combine_schedules"]


if __name__ == "__main__":
    from pymrst.modules.ad_core.utils.schedule_types import Control

    def _schedule(vals, ctrls, n):
        return Schedule(np.array(vals, dtype=float), np.array(ctrls),
                        [Control(W=[f"c{k}"]) for k in range(n)])

    def test_controls_are_appended_and_steps_shifted():
        s = combine_schedules(_schedule([1, 2], [0, 1], 2), _schedule([3], [0], 1),
                              _schedule([4, 5], [1, 0], 2))
        assert np.array_equal(s.step_val, [1, 2, 3, 4, 5])
        assert np.array_equal(s.step_control, [0, 1, 2, 4, 3]) and len(s.controls) == 5
        print("ok test_controls_are_appended_and_steps_shifted")

    def test_a_single_schedule_is_returned_unchanged():
        s = _schedule([1], [0], 1)
        assert combine_schedules(s) is s
        print("ok test_a_single_schedule_is_returned_unchanged")

    test_controls_are_appended_and_steps_shifted()
    test_a_single_schedule_is_returned_unchanged()
    print("\nAll tests passed!")
