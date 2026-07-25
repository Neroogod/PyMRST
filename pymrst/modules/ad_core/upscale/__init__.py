"""AD-side upscaling: coarsen a model, its schedule and its state (MRST
`ad-core/upscale`)."""

from pymrst.modules.ad_core.upscale.upscale_state import (
    coarsen_state,
    upscale_state,
)
from pymrst.modules.ad_core.upscale.upscale_model_tpfa import upscale_model_tpfa
from pymrst.modules.ad_core.upscale.upscale_schedule import upscale_schedule

__all__ = [
    "coarsen_state",
    "upscale_state",
    "upscale_model_tpfa",
    "upscale_schedule",
]
