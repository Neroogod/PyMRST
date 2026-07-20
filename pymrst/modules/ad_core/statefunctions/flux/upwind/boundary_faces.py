"""getBoundaryFacesMerged (MRST ad-core/statefunctions/flux/upwind).

Ports MRST's `getBoundaryFacesMerged.m` - the geometry a higher-order upwind
reconstruction (WENO) needs at the domain boundary. A WENO stencil reaches into
neighbouring cells; at a boundary face there is no neighbour, so MRST invents a
mirror ("ghost") point by reflecting the owning cell's centroid through the
boundary-face centroid: ``x_ghost = x_cell + 2 (x_face - x_cell)``. That places
the ghost the same distance beyond the face as the cell is inside it, which is
the standard zero-gradient mirror.

Returns the boundary faces, their owning cells, and the mirrored face
centroids. The reconstruction itself (`WENOUpwindDiscretization`) is not yet
ported (it needs a MATLAB reference to validate); this pure-geometry helper is,
because its correctness is an invariant, not a comparison.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np


def get_boundary_faces_merged(g: Any, dim: Optional[int] = None
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Boundary faces, their cells, and mirrored (ghost) face centroids.

    Parameters
    ----------
    g : Grid
        A grid with ``faces.neighbors`` (``-1`` on the outside, PyMRST's
        convention), ``faces.centroids`` and ``cells.centroids``.
    dim : int, optional
        Spatial dimension to keep in the centroids (default ``g.grid_dim``).

    Returns
    -------
    bfaces : (nbf,) int
        Indices of the boundary faces.
    bcells : (nbf,) int
        The single interior cell owning each boundary face.
    bfc : (nbf, dim) float
        Mirrored face centroids ``x_cell + 2 (x_face - x_cell)``.
    """
    if dim is None:
        dim = getattr(g, "grid_dim", g.cells.centroids.shape[1])

    neighbors = g.faces.neighbors
    on_boundary = np.any(neighbors < 0, axis=1)
    bfaces = np.flatnonzero(on_boundary)

    pair = neighbors[bfaces]
    # the owning cell is whichever neighbour is not the -1 outside marker
    bcells = np.where(pair[:, 0] >= 0, pair[:, 0], pair[:, 1])

    bfc = g.faces.centroids[bfaces, :dim]
    bcc = g.cells.centroids[bcells, :dim]
    bfc = bcc + 2.0 * (bfc - bcc)
    return bfaces, bcells, bfc


__all__ = ["get_boundary_faces_merged"]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry

    def test_counts_and_cells_on_cartesian():
        """A 3x3x1 grid has 3*2*(3+3)=... boundary faces; each maps to a cell."""
        g = compute_geometry(cart_grid([3, 3, 1], [3.0, 3.0, 1.0]))
        bfaces, bcells, bfc = get_boundary_faces_merged(g)
        # every returned face is genuinely on the boundary
        assert np.all(np.any(g.faces.neighbors[bfaces] < 0, axis=1))
        # every owning cell is a real interior cell
        assert np.all((bcells >= 0) & (bcells < g.cells.num))
        # a 3x3x1 box: 4 sides x 3 + top/bottom 9 each = 12 + 18 = 30 faces
        assert bfaces.size == 30
        print("ok test_counts_and_cells_on_cartesian")

    def test_ghost_is_reflection_through_face():
        """The ghost centroid is the cell centroid reflected through the face:
        the face sits exactly halfway between cell and ghost."""
        g = compute_geometry(cart_grid([4, 2, 1], [8.0, 4.0, 2.0]))
        bfaces, bcells, bfc = get_boundary_faces_merged(g)
        fc = g.faces.centroids[bfaces, :3]
        cc = g.cells.centroids[bcells, :3]
        assert np.allclose(fc, 0.5 * (cc + bfc))
        # ghost lies outside: farther from the cell than the face is
        assert np.all(np.linalg.norm(bfc - cc, axis=1)
                      >= np.linalg.norm(fc - cc, axis=1) - 1e-12)
        print("ok test_ghost_is_reflection_through_face")

    def test_dim_argument_trims_centroids():
        """Passing dim=2 keeps only the first two coordinates."""
        g = compute_geometry(cart_grid([2, 2, 1], [2.0, 2.0, 1.0]))
        _, _, bfc = get_boundary_faces_merged(g, dim=2)
        assert bfc.shape[1] == 2
        print("ok test_dim_argument_trims_centroids")

    test_counts_and_cells_on_cartesian()
    test_ghost_is_reflection_through_face()
    test_dim_argument_trims_centroids()
    print("\nAll tests passed!")
