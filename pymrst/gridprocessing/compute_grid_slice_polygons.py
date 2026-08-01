"""Moved to `pymrst.utils.gridtools.compute_grid_slice_polygons`.

MRST keeps this in `utils/gridtools/` and does not duplicate it in its
`gridprocessing` module, so that is where the implementation now lives.

This module re-exports it so existing imports keep working; new code
should import from `pymrst.utils.gridtools`.
"""

from pymrst.utils.gridtools.compute_grid_slice_polygons import (  # noqa: F401
    compute_grid_slice_polygons,
)

__all__ = ['compute_grid_slice_polygons']
