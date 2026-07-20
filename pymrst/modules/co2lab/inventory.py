"""Backward-compatible import path for CO2 trapping inventory calculations."""

from pymrst.modules.co2lab_mit.inventory import (
    CATEGORIES,
    Inventory,
    inventory_history,
    mass_trapping_distribution,
    trap_depth,
)

__all__ = [
    "CATEGORIES",
    "Inventory",
    "inventory_history",
    "mass_trapping_distribution",
    "trap_depth",
]
