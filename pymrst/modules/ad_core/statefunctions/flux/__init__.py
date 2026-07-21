"""MRST ad-core flux state functions (`ad-core/statefunctions/flux/`).

The Darcy-flux half of the state-function graph. The generic flux nodes
(`PhaseFlux`, `PhasePotentialDifference`, `Transmissibility`, ...) currently
live in the sibling `flow_discretization` module; this package holds the MRST
`flux/` additions that make the discretization pluggable:

- `PermeabilityGradientDiscretization` / `TwoPointFluxApproximation` - the
  ``K grad`` stencil, abstracted so MPFA or mimetic schemes can replace TPFA;
- `PermeabilityPotentialGradient` - the state function that applies it;
- `PhasePotentialDifferenceThresholded` - the potential difference with
  ECLIPSE `THPRES` threshold pressures.

`FaceComponentMobility` (MRST files it here too) is deferred until the upwind
hierarchy (`flux/upwind`, AC2) and `ComponentMobility` (`flowprops`, AC6) land,
since it depends on both.
"""

from pymrst.modules.ad_core.statefunctions.flux.permeability_gradient_discretization import (  # noqa: E501
    PermeabilityGradientDiscretization,
    TwoPointFluxApproximation,
)
from pymrst.modules.ad_core.statefunctions.flux.permeability_potential_gradient import (  # noqa: E501
    PermeabilityPotentialGradient,
)
from pymrst.modules.ad_core.statefunctions.flux.phase_potential_difference_thresholded import (  # noqa: E501
    PhasePotentialDifferenceThresholded,
)
from pymrst.modules.ad_core.statefunctions.flux.upwind import (
    UpwindDiscretization,
    UpwindFunctionWrapperDiscretization,
    UpwindProperty,
    get_boundary_faces_merged,
)
from pymrst.modules.ad_core.statefunctions.flux.flow_state import (
    AdaptiveImplicitFlowStateBuilder,
    ExplicitFlowStateBuilder,
    FlowStateBuilder,
    ImplicitFlowStateBuilder,
)

__all__ = [
    "AdaptiveImplicitFlowStateBuilder",
    "ExplicitFlowStateBuilder",
    "FlowStateBuilder",
    "ImplicitFlowStateBuilder",
    "PermeabilityGradientDiscretization",
    "PermeabilityPotentialGradient",
    "PhasePotentialDifferenceThresholded",
    "TwoPointFluxApproximation",
    "UpwindDiscretization",
    "UpwindFunctionWrapperDiscretization",
    "UpwindProperty",
    "get_boundary_faces_merged",
]
