"""Runnable ad-core examples (MRST `ad-core/examples`)."""

from pymrst.modules.ad_core.examples.ad_buckley_leverett_1d import (
    run_ad_buckley_leverett_1d,
)
from pymrst.modules.ad_core.examples.tutorial_state_functions import (
    TutorialAdd,
    TutorialNumber,
    TutorialProduct,
)

__all__ = [
    "run_ad_buckley_leverett_1d",
    "TutorialAdd",
    "TutorialNumber",
    "TutorialProduct",
]
