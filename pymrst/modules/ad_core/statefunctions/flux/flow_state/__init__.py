"""MRST ad-core flow-state builders (`.../statefunctions/flux/flow_state/`).

Builders that decide, per property, whether a flux term is taken at the new
(implicit) or old (explicit) time level - the switch between fully-implicit,
IMPES, and adaptive-implicit (AIM) time integration.

- `FlowStateBuilder` / `ImplicitFlowStateBuilder` - fully implicit (default);
- `ExplicitFlowStateBuilder` - freeze the flux face-terms at the old state;
- `AdaptiveImplicitFlowStateBuilder` - freeze the cell mobilities only in the
  cells where the step is stable, per a per-cell implicit mask.

The CFL estimate that drives the automatic explicit step / AIM cell mask is
AC21 (`estimateSaturationCFL` / `estimateCompositionCFL`) and not yet ported;
the builders accept it injected or take the mask directly. The hybridizing
mechanism itself is complete.
"""

from pymrst.modules.ad_core.statefunctions.flux.flow_state.flow_state_builder import (  # noqa: E501
    FlowStateBuilder,
    ImplicitFlowStateBuilder,
    owning_grouping,
    seed_property,
)
from pymrst.modules.ad_core.statefunctions.flux.flow_state.explicit_flow_state_builder import (  # noqa: E501
    ExplicitFlowStateBuilder,
)
from pymrst.modules.ad_core.statefunctions.flux.flow_state.adaptive_implicit_flow_state_builder import (  # noqa: E501
    AdaptiveImplicitFlowStateBuilder,
)

__all__ = [
    "AdaptiveImplicitFlowStateBuilder",
    "ExplicitFlowStateBuilder",
    "FlowStateBuilder",
    "ImplicitFlowStateBuilder",
    "owning_grouping",
    "seed_property",
]
