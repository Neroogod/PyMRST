"""SPE9 ad-blackoil example layout."""

from pymrst.modules.ad_blackoil.examples.spe9.simulate_spe9 import (
    SPE9Result, setup_spe9_ad, simulateSPE9, simulate_spe9, simulate_spe9_ad,
)
from pymrst.modules.ad_blackoil.examples.spe9.blackoil_tutorial_spe9 import (
    blackoilTutorialSPE9, blackoil_tutorial_spe9,
)

__all__ = ["SPE9Result", "blackoilTutorialSPE9", "blackoil_tutorial_spe9", "setup_spe9_ad",
           "simulateSPE9", "simulate_spe9", "simulate_spe9_ad"]
