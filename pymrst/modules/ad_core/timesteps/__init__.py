"""MRST ad-core adaptive timestep selection (`ad-core/timesteps/`).

The implementation lives in `timestep_selectors.py`; this package surface
mirrors MRST's `timesteps/` directory (`SimpleTimeStepSelector.m`,
`FactorTimeStepSelector.m`, `IterationCountTimeStepSelector.m`,
`StateChangeTimeStepSelector.m`, `rampupTimesteps.m`). The submodule is
named `timestep_selectors` rather than `selectors` to avoid shadowing the
Python standard-library `selectors` module when run as a script.
"""

from pymrst.modules.ad_core.timesteps.timestep_selectors import (
    FactorTimeStepSelector,
    IterationCountTimeStepSelector,
    SimpleTimeStepSelector,
    StateChangeTimeStepSelector,
    rampup_timesteps,
)

__all__ = [
    "FactorTimeStepSelector",
    "IterationCountTimeStepSelector",
    "SimpleTimeStepSelector",
    "StateChangeTimeStepSelector",
    "rampup_timesteps",
]
