"""MRST ``ad-core/statefunctions/flux/flow_state/ImplicitFlowStateBuilder.m``.

Fully implicit: the fluxes are evaluated at the current Newton iterate. MRST's
class adds nothing to its base, and neither does this one.
"""

from __future__ import annotations

from pymrst.modules.ad_core.ad_oo.statefunctions.flux.flow_state_builder import (
    FlowStateBuilder,
)


class ImplicitFlowStateBuilder(FlowStateBuilder):
    """Fluxes at the current iterate."""


__all__ = ["ImplicitFlowStateBuilder"]


if __name__ == "__main__":
    def test_implicit_flow_state_is_the_state_itself():
        state = {"pressure": 1.0}
        assert ImplicitFlowStateBuilder().build(None, None, state, None, 1.0) is state
        print("ok test_implicit_flow_state_is_the_state_itself")

    test_implicit_flow_state_is_the_state_itself()
    print("\nAll tests passed!")
