"""MRST ``ad-core/timesteps/rampupTimesteps.m``: time steps of `dt` over
`time`, starting with `n` steps that double up to `dt` (the first of them
repeated), and ending with the remainder.

The sums are MATLAB's (`matlab_sum`): whether a remainder step is added
depends on `time - sum(dt)` being positive, and with numpy's pairwise sum a
ten-year ramp gained a step of 6e-8 s that MRST does not take."""

from __future__ import annotations

import numpy as np

from pymrst.utils.matlab_sum import matlab_sum


def rampup_timesteps(time: float, dt: float, n: int = 8) -> np.ndarray:
    """MRST `dT = rampupTimesteps(time, dt, n)`."""
    if time == 0:
        return np.zeros(0)
    dt_init = dt/2.0**np.array([n] + list(range(n, 0, -1)), dtype=float)
    cs_time = np.cumsum(dt_init)
    if np.any(cs_time > time):
        dt_init = dt_init[cs_time < time]
    dt_left = time - matlab_sum(dt_init)
    dt_rem = np.full(int(np.floor(dt_left/dt)), float(dt))
    dt_final = time - matlab_sum(dt_init) - matlab_sum(dt_rem)
    final = np.array([dt_final]) if dt_final > 0 else np.zeros(0)
    return np.concatenate([dt_init, dt_rem, final])


rampupTimesteps = rampup_timesteps
__all__ = ["rampupTimesteps", "rampup_timesteps"]


if __name__ == "__main__":
    def test_ramp_then_full_steps_then_remainder():
        dT = rampup_timesteps(10.5, 1.0, 2)
        assert np.allclose(dT[:3], [0.25, 0.25, 0.5]) and np.allclose(dT[3:-1], 1.0)
        assert np.isclose(dT.sum(), 10.5) and np.isclose(dT[-1], 0.5)
        print("ok test_ramp_then_full_steps_then_remainder")

    def test_short_total_time_truncates_the_ramp():
        dT = rampup_timesteps(0.3, 1.0, 2)
        assert np.allclose(dT, [0.25, 0.05])
        assert rampup_timesteps(0.0, 1.0).size == 0
        print("ok test_short_total_time_truncates_the_ramp")

    def test_ten_years_has_no_round_off_step():
        from pymrst.utils.units import YEAR
        dT = rampup_timesteps(10*YEAR, YEAR/10)
        assert dT.size == 108, dT.size
        print("ok test_ten_years_has_no_round_off_step")

    test_ramp_then_full_steps_then_remainder()
    test_ten_years_has_no_round_off_step()
    test_short_total_time_truncates_the_ramp()
    print("\nAll tests passed!")
