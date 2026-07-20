"""MRST ``initDeckADIFluid.m`` black-oil deck-fluid constructor."""

from typing import Dict

from pymrst.modules.ad_props.blackoil_fluid import (
    BlackOilFluid,
    init_deck_blackoil_fluid,
)


def init_deck_adi_fluid(deck: Dict) -> BlackOilFluid:
    """Assemble an AD-ready black-oil fluid from a parsed deck."""
    return init_deck_blackoil_fluid(deck)

__all__ = ["BlackOilFluid", "init_deck_adi_fluid", "init_deck_blackoil_fluid"]
