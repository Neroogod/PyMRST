"""MRST ``ad-core/statefunctions/flux/flow_state/FlowStateBuilder.m``.

The flow state is the state the fluxes are evaluated in. The base uses the
current iterate unchanged and sets no timestep limit; explicit and
adaptive-implicit builders mix in values from the start of the step.
"""

from __future__ import annotations

from typing import Any

from pymrst.modules.ad_core.ad_oo.physical_model import merge_options


class FlowStateBuilder:
    """Base: no timestep limit, the flow state is the state."""

    def __init__(self, **options: Any):
        self.verbose = None
        merge_options(self, options)
        if self.verbose is None:
            self.verbose = False

    def get_maximum_timestep(self, fd: Any, model: Any, state: Any,
                             state0: Any, dt: float, forces: Any) -> float:
        return float("inf")

    def build(self, fd: Any, model: Any, state: Any, state0: Any,
              dt: float) -> Any:
        return state

    def prepare_timestep(self, fd: Any, model: Any, state: Any, state0: Any,
                         dt: float, driving_forces: Any):
        return self, state

    getMaximumTimestep = get_maximum_timestep
    prepareTimestep = prepare_timestep


__all__ = ["FlowStateBuilder"]


if __name__ == "__main__":
    def test_the_base_flow_state_is_the_state_itself():
        state = {"pressure": 1.0}
        builder = FlowStateBuilder()
        assert builder.build(None, None, state, None, 1.0) is state
        assert builder.get_maximum_timestep(None, None, state, None, 1.0, None) == float("inf")
        print("ok test_the_base_flow_state_is_the_state_itself")

    def test_unknown_options_are_refused():
        try:
            FlowStateBuilder(noSuchOption=1)
            raise SystemExit("should have raised")
        except AttributeError:
            pass
        print("ok test_unknown_options_are_refused")

    test_the_base_flow_state_is_the_state_itself()
    test_unknown_options_are_refused()
    print("\nAll tests passed!")
