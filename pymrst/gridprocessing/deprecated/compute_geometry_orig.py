"""Compatibility entry for deprecated MRST ``computeGeometryOrig``."""

import warnings

from pymrst.gridprocessing.compute_geometry import compute_geometry


def compute_geometry_orig(grid, *_, **__):
    """Compute geometry using the maintained implementation.

    MRST retained ``computeGeometryOrig`` only as a deprecated predecessor to
    ``computeGeometry``. PyMRST exposes the old name for layout and migration
    compatibility while deliberately using the corrected maintained kernel.
    """
    warnings.warn(
        "computeGeometryOrig is deprecated; use compute_geometry",
        DeprecationWarning,
        stacklevel=2,
    )
    return compute_geometry(grid)


computeGeometryOrig = compute_geometry_orig
