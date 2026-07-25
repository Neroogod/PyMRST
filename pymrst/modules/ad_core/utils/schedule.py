"""Simulation schedules (MRST ad-core `simpleSchedule` and friends).

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


def simple_schedule(dt, W=None, bc=None, src=None) -> Schedule:
    """A schedule with a single, unchanging control (MRST `simpleSchedule`).

    Every step runs under control 0, whose wells/bc/sources are the ones passed
    here. This is the common case, and the base every richer schedule is built
    from by adding controls and pointing steps at them.
    """
    dt = np.asarray(dt, dtype=float).ravel()
    return Schedule(step_val=dt,
                    step_control=np.zeros(dt.size, dtype=np.int64),
                    controls=[Control(W=W, bc=bc, src=src)])


def compress_schedule(schedule: Schedule) -> Schedule:
    """Merge each maximal run of one control into a single step (MRST
    `compressSchedule`): the timesteps of a control period are summed, so the
    result has one step per control period and the same total time.

    This is the coarsest schedule with the same control history - a report of
    *what was controlled when*, not how finely it was stepped.
    """
    ctrl = schedule.step_control
    val = schedule.step_val
    new_val: List[float] = []
    new_ctrl: List[int] = []
    i, n = 0, ctrl.size
    while i < n:
        c = int(ctrl[i])
        j = i
        while j < n and int(ctrl[j]) == c:
            j += 1
        new_val.append(float(val[i:j].sum()))
        new_ctrl.append(c)
        i = j
    return Schedule(step_val=np.array(new_val),
                    step_control=np.array(new_ctrl, dtype=np.int64),
                    controls=list(schedule.controls))


def refine_schedule(schedule: Schedule, max_dt: float) -> Schedule:
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


def combine_schedules(*schedules: Schedule) -> Schedule:
    """Concatenate schedules end to end (MRST `combineSchedules`).

    The steps run in order; each schedule's controls are kept and its step
    indices shifted so they still point at their own controls in the combined
    list. The total time is the sum of the parts.
    """
    if not schedules:
        raise ValueError("nothing to combine")
    all_controls: List[Control] = []
    all_val: List[np.ndarray] = []
    all_ctrl: List[np.ndarray] = []
    offset = 0
    for s in schedules:
        all_val.append(s.step_val)
        all_ctrl.append(s.step_control + offset)
        all_controls.extend(s.controls)
        offset += len(s.controls)
    return Schedule(step_val=np.concatenate(all_val),
                    step_control=np.concatenate(all_ctrl).astype(np.int64),
                    controls=all_controls)


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
        r = refine_schedule(s, 5.0 * DAY)
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
        back = compress_schedule(refine_schedule(s, 3.0 * DAY))
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
    test_bad_reference_is_rejected()
    print("\nAll tests passed!")
