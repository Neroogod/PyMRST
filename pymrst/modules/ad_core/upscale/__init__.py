"""AD-side upscaling: coarsen a model, its schedule and its state (MRST
`ad-core/upscale`)."""

from pymrst.modules.ad_core.upscale.upscale_state import (
    coarsen_state,
    upscale_state,
    upscale_state_ad,
)
from pymrst.modules.ad_core.upscale.upscale_model_tpfa import (
    upscale_model_tpfa,
    upscale_model_tpfa_ad,
)
from pymrst.modules.ad_core.upscale.upscale_schedule import (
    upscale_schedule,
    upscale_schedule_ad,
)

__all__ = [
    "coarsen_state",
    "upscale_state",
    "upscale_state_ad",
    "upscale_model_tpfa",
    "upscale_model_tpfa_ad",
    "upscale_schedule",
    "upscale_schedule_ad",
]
