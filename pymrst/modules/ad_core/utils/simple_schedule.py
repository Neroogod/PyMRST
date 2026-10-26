"""MRST ``ad-core/utils/simpleSchedule.m``: a schedule of the given time
steps, all under one control holding the wells `W`, boundary conditions `bc`
and sources `src`. Step controls are 0-based."""

from typing import Any

import numpy as np

from pymrst.modules.ad_core.utils.schedule_types import Control, Schedule


def simple_schedule(dt: Any, W: Any = None, bc: Any = None, src: Any = None) -> Schedule:
    """MRST `schedule = simpleSchedule(dt, 'W', W, 'bc', bc, 'src', src)`."""
    shape = np.shape(dt)
    assert len(shape) <= 1 or shape[0] == 1 or shape[1] == 1, (
        "Timesteps must be given as a row or column vector")
    dt = np.asarray(dt, dtype=float).ravel()
    return Schedule(step_val=dt, step_control=np.zeros(dt.size, dtype=np.int64),
                    controls=[Control(W=W, bc=bc, src=src)])


simpleSchedule = simple_schedule
__all__ = ["simpleSchedule", "simple_schedule"]


if __name__ == "__main__":
    def test_one_control_for_every_step():
        s = simple_schedule(np.array([[1.0], [2.0], [3.0]]), W=["w"])
        assert np.array_equal(s.step_val, [1, 2, 3]) and np.array_equal(s.step_control, [0, 0, 0])
        assert len(s.controls) == 1 and s.controls[0].W == ["w"] and s.controls[0].bc is None
        print("ok test_one_control_for_every_step")

    def test_a_matrix_of_steps_is_rejected():
        try:
            simple_schedule(np.ones((2, 2)))
        except AssertionError:
            print("ok test_a_matrix_of_steps_is_rejected")
            return
        raise AssertionError("expected an AssertionError")

    test_one_control_for_every_step()
    test_a_matrix_of_steps_is_rejected()
    print("\nAll tests passed!")
