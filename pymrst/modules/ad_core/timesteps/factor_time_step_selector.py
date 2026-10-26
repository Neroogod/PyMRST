"""MRST ``ad-core/timesteps/FactorTimeStepSelector.m``: every ministep grows
by `maxRelativeAdjustment` and every cut shrinks by `minRelativeAdjustment`."""

from __future__ import annotations

from typing import Any, Optional

from pymrst.modules.ad_core.timesteps.simple_time_step_selector import SimpleTimeStepSelector


class FactorTimeStepSelector(SimpleTimeStepSelector):
    """MRST's `FactorTimeStepSelector`."""

    def cut_timestep(self, dt_prev: Optional[float], dt: float, model: Any, solver: Any,
                     state_prev: Any, state_curr: Any, forces: Any = None) -> float:
        return dt*self.minRelativeAdjustment

    def compute_timestep(self, dt: float, dt_prev: Optional[float], model: Any, solver: Any,
                         state_prev: Any, state_curr: Any, forces: Any = None) -> float:
        return dt*self.maxRelativeAdjustment

    cutTimestep = cut_timestep
    computeTimestep = compute_timestep


__all__ = ["FactorTimeStepSelector"]


if __name__ == "__main__":
    def test_grow_and_cut_by_the_factors():
        sel = FactorTimeStepSelector(maxRelativeAdjustment=3.0)
        assert sel.compute_timestep(2.0, None, None, None, None, None) == 6.0
        assert sel.cut_timestep(None, 2.0, None, None, None, None) == 1.0
        print("ok test_grow_and_cut_by_the_factors")

    test_grow_and_cut_by_the_factors()
    print("\nAll tests passed!")
