"""MRST ``plotting/Contents.m`` compatibility catalog.

The MRST-shaped core is being implemented according to
``plotting_implementation.md``. Public functions are exported from
``pymrst.plotting`` as each phase is completed.
"""

PUBLIC_CORE = (
    "boundary_faces",
    "plot_faces",
    "plot_face_data",
    "plot_grid",
    "plot_cell_data",
    "plot_node_data",
    "plot_well",
    "plot_faults",
    "outline_coarse_grid",
    "plot_block_and_neighbors",
    "plot_slice",
    "interpolate_data",
    "unstructured_surf",
    "unstructured_contour",
    "plot_contours",
    "colorbar_hist",
    "mrst_colorbar",
    "plot_grid_volumes",
)

__all__ = ["PUBLIC_CORE"]
