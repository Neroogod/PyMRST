"""Backward-compatible import path for the AD-core schedule driver.

The implementation lives in `ad_core.simulators.simulate_schedule_ad`,
mirroring MRST's `ad-core/simulators/simulateScheduleAD.m`.
"""

from pymrst.modules.ad_core.simulators.simulate_schedule_ad import simulate_schedule_ad

__all__ = ["simulate_schedule_ad"]
