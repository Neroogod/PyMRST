"""MRST ``ad-core/statefunctions/flux/upwind/getBoundaryFacesMerged.m``.

The boundary faces of a grid (`boundaryFaces`), their cells, and each face's
centroid reflected through it from the cell centroid,
`bcc + 2*(bfc - bcc)`, in the first `dim` coordinates (default the grid
dimension). `WENOUpwindDiscretization` uses the reflected points as ghost
cells. MRST's merging of a cell's boundary faces with the same direction tag
follows an early `return` and never runs, so it is not ported.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

from pymrst.utils.gridtools.boundary_faces import boundary_faces


def get_boundary_faces_merged(G: Any, dim: Optional[int] = None
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MRST `getBoundaryFacesMerged`: `(bfaces, bcells, bfc)`."""
    if dim is None:
        dim = G.griddim
    bfaces, bcells = boundary_faces(G, with_cells=True)
    bfc = np.asarray(G.faces.centroids, dtype=float)[bfaces, :dim]
    bcc = np.asarray(G.cells.centroids, dtype=float)[bcells, :dim]
    bfc = bcc + 2*(bfc - bcc)
    return bfaces, bcells, bfc


getBoundaryFacesMerged = get_boundary_faces_merged

__all__ = ["getBoundaryFacesMerged", "get_boundary_faces_merged"]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry

    def test_faces_cells_and_reflected_centroids():
        G = compute_geometry(cart_grid([3, 2], [3.0, 2.0]))
        bfaces, bcells, bfc = get_boundary_faces_merged(G)
        assert bfaces.size == 10 and np.all(np.diff(bfaces) > 0)
        cc = G.cells.centroids[bcells]
        fc = G.faces.centroids[bfaces]
        np.testing.assert_allclose(bfc, cc + 2*(fc - cc))
        xmin = bfaces[np.isclose(fc[:, 0], 0.0)]
        assert np.allclose(bfc[np.isin(bfaces, xmin), 0], -0.5)
        print("ok test_faces_cells_and_reflected_centroids")

    def test_dim_selects_coordinates():
        G = compute_geometry(cart_grid([2, 2, 2]))
        _, _, bfc = get_boundary_faces_merged(G, 2)
        assert bfc.shape == (24, 2)
        print("ok test_dim_selects_coordinates")

    test_faces_cells_and_reflected_centroids()
    test_dim_selects_coordinates()
    print("\nAll tests passed!")
