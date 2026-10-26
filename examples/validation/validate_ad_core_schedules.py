"""Validate MRST's schedule utilities: `simpleSchedule`, `combineSchedules`,
`compressSchedule`, `filterSchedule` and `refineSchedule`, ported in
`ad_core/utils/` one file each.

The reference is MRST 2024a (`dump_ad_core_schedules_reference.m`): three
schedules combined (one with two controls, stepped out of order), a schedule
with repeated and split control runs compressed, one with unused controls
filtered, and a three-step, two-control schedule refined twice - once from
time 0 with a step ending a tenth of a microsecond past an old report time,
and once from day 2. Step lengths, step controls (MRST's less one), the order
of the controls and `repStep` are compared.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

HERE = Path(__file__).resolve().parent
RTOL = 1e-15


def report(ok: bool, text: str) -> bool:
    print(f"{'ok  ' if ok else 'FAIL'} {text}")
    return ok


def _check(tag: str, s, ref, names: bool = True) -> bool:
    val, ctrl = np.atleast_1d(ref.val), np.atleast_1d(ref.control)
    ok = report(s.step_val.shape == val.shape
                and np.abs(s.step_val - val).max() <= RTOL*np.abs(val).max(),
                f"{tag}: step lengths ({s.step_val.size})")
    ok &= report(np.array_equal(s.step_control + 1, ctrl), f"{tag}: step controls")
    if names:
        ok &= report([c.W[0] for c in s.controls] == [str(n) for n in np.atleast_1d(ref.names)],
                     f"{tag}: controls in MRST's order")
    return ok


def main() -> int:
    from pymrst.modules.ad_core.utils.combine_schedules import combine_schedules
    from pymrst.modules.ad_core.utils.compress_schedule import compress_schedule
    from pymrst.modules.ad_core.utils.filter_schedule import filter_schedule
    from pymrst.modules.ad_core.utils.refine_schedule import refine_schedule
    from pymrst.modules.ad_core.utils.schedule_types import Control, Schedule
    from pymrst.modules.ad_core.utils.simple_schedule import simple_schedule
    from pymrst.utils.units import DAY

    ref = loadmat(HERE/"ref_ad_core_schedules.mat", squeeze_me=True, struct_as_record=False)
    s1 = simple_schedule(np.array([1, 2, 3])*DAY, W=["c1"])
    ok = _check("simpleSchedule", s1, ref["simple"], names=False)
    s2 = Schedule(np.array([4.0, 5.0])*DAY, np.array([1, 0]), [Control(W=["c2"]), Control(W=["c3"])])
    s3 = simple_schedule(6*DAY, W=["c4"])
    comb = combine_schedules(s1, s2, s3)
    ok &= _check("combineSchedules", comb, ref["c"])
    s4 = Schedule(np.arange(1, 7)*DAY, np.array([1, 1, 3, 3, 1, 2]) - 1, list(comb.controls))
    ok &= _check("compressSchedule", compress_schedule(s4), ref["cmp"])
    s5 = Schedule(comb.step_val, np.array([4, 2, 4, 4, 2, 4]) - 1, list(comb.controls))
    ok &= _check("filterSchedule", filter_schedule(s5), ref["flt"])
    base = Schedule(np.array([10.0, 10.0, 10.0])*DAY, np.array([0, 1, 1]),
                    [Control(W=["c1"]), Control(W=["c2"])])
    for tag, t0, steps, r in (("refineSchedule from 0", 0.0, [5, 5, 3, 7.0000001, 4], ref["ref1"]),
                              ("refineSchedule from day 2", 2*DAY, [3, 4.5], ref["ref2"])):
        out = refine_schedule(t0, np.array(steps)*DAY, base)
        ok &= _check(tag, out, r, names=False)
        ok &= report(np.array_equal(out.repStep, np.atleast_1d(r.repStep).astype(bool)),
                     f"{tag}: repStep")
    print("\nThe schedule utilities match MRST" if ok else "\nThe schedule utilities do NOT match MRST")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
