"""Backward-compatible import path for CO2 Atlas grid helpers."""

from pymrst.modules.co2lab_mit.atlas import (
    atlas_grid,
    available_formations,
    convert_atlas_to_3d,
    read_aai_grid,
    read_atlas_dataset,
)

__all__ = [
    "atlas_grid",
    "available_formations",
    "convert_atlas_to_3d",
    "read_aai_grid",
    "read_atlas_dataset",
]
