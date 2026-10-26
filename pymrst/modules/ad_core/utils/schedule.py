"""Simulation schedules (MRST ad-core `simpleSchedule` and friends).

MRST's schedule functions are each ported in full in their own file -
`simple_schedule.py`, `combine_schedules.py`, `compress_schedule.py`,
`filter_schedule.py`, `refine_schedule.py` - and the structure in
`schedule_types.py`. This module gathers them under the import path the port
has always used, with two PyMRST helpers: `subset_schedule` (a schedule
restricted to some of its steps) and `step_times`.

`refine_schedule_ad`, below, is PyMRST's earlier `refine_schedule`, kept for
its callers: it splits every step longer than `max_dt` into equal pieces,
where MRST's `refineSchedule` merges a second time grid into the schedule's.
The earlier notes follow.

Simulation schedules (MRST ad-core `simpleSchedule` and friends).

Until now a "schedule" in this port was a bare list of timesteps, and the wells
were whatever the model was built with - fixed for the whole run. A real field
schedule is not that. A well opens in year 3, a rate steps down in year 7, an
injector switches from water to gas: the *controls change with time*, and the
timesteps are grouped into control periods.

MRST captures that in one structure, and this ports it:

    schedule.step.val      - the timestep of each report step
    schedule.step.control  - which control each step runs under (an index)
    schedule.control       - the controls themselves: a list of `Control`,
                             each carrying its own wells / bc / sources

So `step.control[i]` selects `control[step.control[i]]` for step `i`. A single
fixed control is just the length-one case (`simple_schedule`), and everything
else - compressing a fine schedule into its control periods, refining a coarse
one for accuracy, gluing two together - is an operation on this structure.

The editing operations are checked by *invariants* rather than a MATLAB
reference, because that is what they are: the total simulated time is preserved
by every one of them; `compress` leaves exactly one step per maximal run of a
single control; `refine` never lengthens a step and preserves its control; and
`compress(refine(s))` recovers the control periods of `s`.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from pymrst.modules.ad_core.utils.schedule_types import Control, Schedule
from pymrst.modules.ad_core.utils.combine_schedules import combine_schedules
from pymrst.modules.ad_core.utils.compress_schedule import compress_schedule
from pymrst.modules.ad_core.utils.filter_schedule import filter_schedule
from pymrst.modules.ad_core.utils.refine_schedule import refine_schedule
from pymrst.modules.ad_core.utils.simple_schedule import simple_schedule


def subset_schedule(schedule: Schedule, steps: Sequence[int]) -> Schedule:
    """The schedule restricted to `steps` (MRST-style `filterSchedule`).

    Controls that no surviving step references are dropped, and the remaining
    step indices are renumbered to match - so the result is self-consistent and
    carries no dead controls.
    """
    steps = np.asarray(steps, dtype=np.int64).ravel()
    val = schedule.step_val[steps]
    ctrl = schedule.step_control[steps]
    used = sorted(set(int(c) for c in ctrl))
    remap = {c: k for k, c in enumerate(used)}
    return Schedule(step_val=val,
                    step_control=np.array([remap[int(c)] for c in ctrl],
                                          dtype=np.int64),
                    controls=[schedule.controls[c] for c in used])


def step_times(schedule: Schedule) -> np.ndarray:
    """Cumulative time at the *end* of each step - the report times."""
    return np.cumsum(schedule.step_val)



# ---------------------------------------------------------------------------
# PyMRST's earlier version.

def refine_schedule_ad(schedule: Schedule, max_dt: float) -> Schedule:
    """Subdivide any step longer than `max_dt` into equal pieces, keeping its
    control (a uniform-grid `refineSchedule`).

    A step of length `L > max_dt` becomes `ceil(L/max_dt)` equal steps of
    `L/ceil(...)`, all under the original control. The total time and the
    control history are unchanged; every resulting step is `<= max_dt`. This is
    what you reach for when a control period is one big step and the solver
    needs a finer grid through it.
    """
    if max_dt <= 0:
        raise ValueError("max_dt must be positive")
    new_val: List[float] = []
    new_ctrl: List[int] = []
    for length, c in zip(schedule.step_val, schedule.step_control):
        n_sub = max(1, int(np.ceil(length / max_dt - 1e-12)))
        new_val.extend([length / n_sub] * n_sub)
        new_ctrl.extend([int(c)] * n_sub)
    return Schedule(step_val=np.array(new_val),
                    step_control=np.array(new_ctrl, dtype=np.int64),
                    controls=list(schedule.controls))


combineSchedules = combine_schedules
compressSchedule = compress_schedule
filterSchedule = filter_schedule
refineSchedule = refine_schedule
simpleSchedule = simple_schedule

__all__ = ["Control", "Schedule", "combineSchedules", "combine_schedules", "compressSchedule",
           "compress_schedule", "filterSchedule", "filter_schedule", "refineSchedule",
           "refine_schedule", "refine_schedule_ad", "simpleSchedule", "simple_schedule",
           "step_times", "subset_schedule"]


if __name__ == "__main__":
    from pymrst.utils.units import DAY, YEAR

    def test_simple_schedule():
        s = simple_schedule(np.full(5, 30.0 * DAY), W=["a", "b"])
        assert s.n_steps == 5
        assert np.all(s.step_control == 0)
        assert len(s.controls) == 1 and s.controls[0].W == ["a", "b"]
        assert np.isclose(s.total_time, 150.0 * DAY)
        print("ok test_simple_schedule")

    def _two_control_schedule() -> Schedule:
        # 4 steps under control 0, then 3 under control 1
        return Schedule(
            step_val=np.array([10., 10., 10., 10., 20., 20., 20.]) * DAY,
            step_control=np.array([0, 0, 0, 0, 1, 1, 1]),
            controls=[Control(W=["open"]), Control(W=["shut"])],
        )

    def test_compress_leaves_one_step_per_control_run():
        s = _two_control_schedule()
        c = compress_schedule(s)
        assert c.n_steps == 2, c.n_steps
        assert list(c.step_control) == [0, 1]
        assert np.isclose(c.step_val[0], 40. * DAY)   # 4 x 10
        assert np.isclose(c.step_val[1], 60. * DAY)   # 3 x 20
        assert np.isclose(c.total_time, s.total_time)
        print("ok test_compress_leaves_one_step_per_control_run")

    def test_compress_merges_only_consecutive_runs():
        # control pattern 0,1,0 must stay three blocks, not two
        s = Schedule(step_val=np.array([1., 2., 3., 4.]) * DAY,
                     step_control=np.array([0, 1, 1, 0]),
                     controls=[Control(), Control()])
        c = compress_schedule(s)
        assert list(c.step_control) == [0, 1, 0]
        assert np.allclose(c.step_val / DAY, [1., 5., 4.])
        print("ok test_compress_merges_only_consecutive_runs")

    def test_refine_never_lengthens_and_keeps_control():
        s = _two_control_schedule()
        r = refine_schedule_ad(s, 5.0 * DAY)
        assert np.all(r.step_val <= 5.0 * DAY + 1e-9)
        assert np.isclose(r.total_time, s.total_time)
        # every refined step inherits a real control from its parent
        assert set(r.step_control.tolist()) == {0, 1}
        # the 10-day steps split in 2, the 20-day in 4: 4*2 + 3*4 = 20 steps
        assert r.n_steps == 4 * 2 + 3 * 4
        print("ok test_refine_never_lengthens_and_keeps_control")

    def test_compress_after_refine_recovers_the_control_periods():
        """The round-trip invariant: refining then compressing returns the
        control history you started from - same blocks, same durations."""
        s = _two_control_schedule()
        base = compress_schedule(s)
        back = compress_schedule(refine_schedule_ad(s, 3.0 * DAY))
        assert list(back.step_control) == list(base.step_control)
        assert np.allclose(back.step_val, base.step_val)
        assert np.isclose(back.total_time, s.total_time)
        print("ok test_compress_after_refine_recovers_the_control_periods")

    def test_combine_shifts_control_indices():
        a = simple_schedule(np.full(2, 1.0 * YEAR), W=["A"])
        b = simple_schedule(np.full(3, 0.5 * YEAR), W=["B"])
        c = combine_schedules(a, b)
        assert c.n_steps == 5
        assert list(c.step_control) == [0, 0, 1, 1, 1]
        assert c.controls[0].W == ["A"] and c.controls[1].W == ["B"]
        assert np.isclose(c.total_time, 2 * YEAR + 1.5 * YEAR)
        print("ok test_combine_shifts_control_indices")

    def test_subset_drops_dead_controls_and_renumbers():
        s = _two_control_schedule()
        # keep only the control-1 steps
        sub = subset_schedule(s, [4, 5, 6])
        assert sub.n_steps == 3
        assert len(sub.controls) == 1                 # control 0 dropped
        assert np.all(sub.step_control == 0)          # control 1 renumbered to 0
        assert sub.controls[0].W == ["shut"]
        print("ok test_subset_drops_dead_controls_and_renumbers")

    def test_filter_drops_dead_controls_but_keeps_every_step():
        """MRST's `filterSchedule`: unlike `subset_schedule`, no step is lost.

        The two-control schedule references both, so nothing is dropped; a
        schedule carrying an unreferenced third control loses exactly that one
        and keeps its step count.
        """
        s = _two_control_schedule()
        same = filter_schedule(s)
        assert same.n_steps == s.n_steps
        assert len(same.controls) == len(s.controls)

        padded = Schedule(step_val=s.step_val, step_control=s.step_control,
                          controls=list(s.controls) + [Control(W=["unused"])])
        trimmed = filter_schedule(padded)
        assert trimmed.n_steps == padded.n_steps, "a step was lost"
        assert len(trimmed.controls) == len(s.controls), "dead control kept"
        assert all(c.W != ["unused"] for c in trimmed.controls)
        print("ok test_filter_drops_dead_controls_but_keeps_every_step")

    def test_bad_reference_is_rejected():
        try:
            Schedule(step_val=[1.0], step_control=[3], controls=[Control()])
        except ValueError:
            print("ok test_bad_reference_is_rejected")
            return
        raise AssertionError("a step pointing at a missing control must raise")

    test_simple_schedule()
    test_compress_leaves_one_step_per_control_run()
    test_compress_merges_only_consecutive_runs()
    test_refine_never_lengthens_and_keeps_control()
    test_compress_after_refine_recovers_the_control_periods()
    test_combine_shifts_control_indices()
    test_subset_drops_dead_controls_and_renumbers()
    test_filter_drops_dead_controls_but_keeps_every_step()
    test_bad_reference_is_rejected()
    print("\nAll tests passed!")
