"""MRST ad-core adaptive timestep selection (`ad-core/timesteps/`).

Each of MRST's files is ported in its own module: `simple_time_step_selector`,
`factor_time_step_selector`, `iteration_count_time_step_selector`,
`state_change_time_step_selector` and `rampup_timesteps`. The `*AD` classes
are PyMRST's earlier selectors (`timestep_selectors.py`), kept for the legacy
nonlinear solver.
"""

from pymrst.modules.ad_core.timesteps.factor_time_step_selector import FactorTimeStepSelector
from pymrst.modules.ad_core.timesteps.iteration_count_time_step_selector import (
    IterationCountTimeStepSelector,
)
from pymrst.modules.ad_core.timesteps.rampup_timesteps import rampup_timesteps, rampupTimesteps
from pymrst.modules.ad_core.timesteps.simple_time_step_selector import SimpleTimeStepSelector
from pymrst.modules.ad_core.timesteps.state_change_time_step_selector import (
    StateChangeTimeStepSelector,
)
from pymrst.modules.ad_core.timesteps.timestep_selectors import (
    FactorTimeStepSelectorAD,
    IterationCountTimeStepSelectorAD,
    SimpleTimeStepSelectorAD,
    StateChangeTimeStepSelectorAD,
)

__all__ = [
    "FactorTimeStepSelector",
    "FactorTimeStepSelectorAD",
    "IterationCountTimeStepSelector",
    "IterationCountTimeStepSelectorAD",
    "SimpleTimeStepSelector",
    "SimpleTimeStepSelectorAD",
    "StateChangeTimeStepSelector",
    "StateChangeTimeStepSelectorAD",
    "rampupTimesteps",
    "rampup_timesteps",
]
