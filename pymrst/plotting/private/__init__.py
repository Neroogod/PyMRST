"""Private topology/rendering helpers matching MRST ``plotting/private``."""

from pymrst.plotting.private.plot_face_outline import (
    face_outline_geometry,
    plot_face_outline,
)
from pymrst.plotting.private.plot_patches import (
    build_patch_geometry,
    normalize_color_data,
    plot_patches,
)
from pymrst.plotting.private.coarse_faces import (
    coarse_grid_parts,
    expand_coarse_faces,
)

__all__ = [
    "build_patch_geometry",
    "coarse_grid_parts",
    "expand_coarse_faces",
    "face_outline_geometry",
    "normalize_color_data",
    "plot_face_outline",
    "plot_patches",
]
