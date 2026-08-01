# PyMRST plotting

MRST-shaped grid and field visualization with backend-neutral topology and
lazy rendering adapters. PyVista is the primary 2-D/3-D grid renderer.

Implemented core entries currently include `boundary_faces`, `plot_faces`,
`plot_face_data`, `plot_cell_data`, `plot_node_data`, and the 2-D/3-D
topology paths of `plot_grid`. The Phase 3 core also includes `plot_well`
and `plot_faults`, `outline_coarse_grid`, and `plot_block_and_neighbors`.
Cartesian `plot_slice` completes the Phase 3 public surface. Both snake-case
and MATLAB-style aliases are exported without eagerly importing PyVista.
Phase 4 begins with backend-free `interpolate_data` for grid entity and
explicit scattered coordinates, plus `unstructured_surf` for lifted 2-D
surface geometry and `unstructured_contour` for structured contour sampling
and headless Matplotlib rendering. Discrete 3-D `plot_contours` completes the
Phase 4 public surface.
Phase 5 includes headless `colorbar_hist` and `mrst_colorbar`; volume
isosurface rendering is provided by `plot_grid_volumes` with reusable sampled
fields and patch-budget control.

The end-to-end implementation plan is documented at the repository root in
`plotting_implementation.md`.
