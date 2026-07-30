"""Validation for MRST timestep APIs.

Covers ``SimpleTimeStepSelector``, ``FactorTimeStepSelector``, and
``rampupTimesteps``.
"""

from types import SimpleNamespace

import numpy as np

from pymrst.modules.ad_core.statefunctions import State
from pymrst.modules.ad_core.timesteps import (
    FactorTimeStepSelector,
    SimpleTimeStepSelector,
    rampup_timesteps,
)


def test_simple_selector_enforces_limits_and_relative_leash():
    selector = SimpleTimeStepSelector(
        min_timestep=2.0,
        max_timestep=20.0,
        min_relative_adjustment=0.5,
        max_relative_adjustment=1.5,
    )
    state = State(x=np.ones(1))
    assert selector.pick_timestep(None, 30.0, None, None, None, state) == 20.0
    # Past the first control-step boundary, relative adjustment is constrained.
    selector.compute_timestep = lambda *args: 100.0
    assert selector.pick_timestep(20.0, 10.0, None, None, state, state) == 15.0
    assert selector.cut_timestep(10.0, 10.0, None, None, state, state) == 5.0


def test_factor_selector_grows_then_respects_hard_maximum():
    selector = FactorTimeStepSelector(
        max_relative_adjustment=2.0, max_timestep=12.0
    )
    state = State(x=np.ones(1))
    assert selector.pick_timestep(None, 5.0, None, None, None, state) == 10.0
    assert selector.pick_timestep(10.0, 10.0, None, None, state, state) == 12.0


def test_rampup_timesteps_are_positive_and_close_exact_schedule():
    steps = rampup_timesteps(100.0, 20.0, n=3)
    assert np.all(steps > 0.0)
    assert np.isclose(steps.sum(), 100.0)
    assert np.all(steps <= 20.0)
    assert steps[0] == steps[1] == 2.5
    assert rampup_timesteps(0.0, 20.0).size == 0
