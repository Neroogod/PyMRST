"""Backward-compatible import path for CO2 Atlas grid helpers."""

from pymrst.modules.co2lab_mit.atlas import (
    _atlas_dir,
    atlas_grid,
    available_formations,
    convert_atlas_to_3d,
    read_aai_grid,
    read_atlas_dataset,
)

# `_atlas_dir` is private by convention, and is re-exported anyway: this module
# exists to preserve the *old* import path, and that path included it. Callers
# that located the Atlas data through it keep working.
__all__ = [
    "_atlas_dir",
    "atlas_grid",
    "available_formations",
    "convert_atlas_to_3d",
    "read_aai_grid",
    "read_atlas_dataset",
]
