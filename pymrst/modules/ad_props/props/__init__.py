"""MRST-shaped `ad-props/props` namespace."""

from pymrst.modules.ad_props.blackoil_fluid import BlackOilFluid, init_deck_blackoil_fluid
from pymrst.modules.ad_props.simple_adi_fluid import ADFluid, init_simple_adi_fluid

init_deck_adi_fluid = init_deck_blackoil_fluid

__all__ = [
    "ADFluid",
    "BlackOilFluid",
    "init_deck_adi_fluid",
    "init_deck_blackoil_fluid",
    "init_simple_adi_fluid",
]
