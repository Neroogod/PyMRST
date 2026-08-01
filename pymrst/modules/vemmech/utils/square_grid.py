"""Square test grids for vemmech (MRST `squareGrid`).

One entry point that builds the family of 2-D and 3-D grids the vemmech tests
and examples run on: Cartesian, triangulated, PEBI, and the glued
``boxes*``/``mixed*`` layouts that put cells of different shape and size next
to each other. The point of the awkward ones is to exercise VEM on grids no
finite-volume scheme would enjoy.

Every non-Cartesian grid comes back with boundary face tags in
``cells.tags``: 1 and 2 for the ``x`` extremes, 3 and 4 for ``y`` (and for
``z`` in 3-D, following MRST).
"""

from typing import Tuple

import numpy as np

from pymrst.gridprocessing.cart_grid import cart_grid
from pymrst.gridprocessing.compute_geometry import compute_geometry
from pymrst.gridprocessing.glue_2d_grid import glue_2d_grid
# MRST's sortEdges is consolidated into the PEBI construction; the parity
# ledger records it there, so this is the documented entry point for it
from pymrst.gridprocessing.pebi import _sort_cell_faces as sort_cell_faces
from pymrst.gridprocessing.pebi import pebi
from pymrst.gridprocessing.remove_short_edges import remove_short_edges
from pymrst.gridprocessing.testgrids.twister import twister
from pymrst.gridprocessing.translate_grid import translate_grid
from pymrst.gridprocessing.triangle_grid import triangle_grid

GRID_TYPES = (
    "cartgrid", "triangle", "pebi",
    "boxes1", "boxes2", "boxes3", "boxes4",
    "mixed1", "mixed2", "mixed3", "mixed4",
)


def square_grid(cart_dims, L, *, grid_type: str = "cartgrid",
                disturb: float = 0.0):
    """Return the square test grid named by ``grid_type``.

    ``cart_dims`` and ``L`` are the logical size and physical extent.
    ``disturb`` perturbs interior nodes through MRST's ``twister``, which is
    what makes the cells irregular; it leaves the boundary in place so the
    domain stays square.
    """
    cart_dims = [int(n) for n in np.atleast_1d(cart_dims).ravel()]
    L = [float(v) for v in np.atleast_1d(L).ravel()]
    if grid_type not in GRID_TYPES:
        raise ValueError(
            f"no such grid type: {grid_type!r}; expected one of {GRID_TYPES}"
        )
    if len(cart_dims) != 2 and grid_type != "cartgrid":
        raise ValueError("only 'cartgrid' is defined outside two dimensions")

    if grid_type == "cartgrid":
        grid = cart_grid(cart_dims, L)
        if disturb > 0:
            grid = twister(grid, disturb)
    elif grid_type in ("triangle", "pebi"):
        grid = _delaunay_grid(cart_dims, L, disturb, grid_type)
    else:
        grid = _glued_grid(cart_dims, L, disturb, grid_type)

    if grid_type != "cartgrid":
        grid = _tag_boundary(grid, cart_dims, L)
    return grid


def _delaunay_grid(cart_dims, L, disturb: float, grid_type: str):
    """Triangulate the Cartesian node cloud, optionally taking its dual."""
    from scipy.spatial import Delaunay

    base = cart_grid(cart_dims, L)
    if disturb > 0:
        base = twister(base, disturb)
    points = np.asarray(base.nodes.coords, dtype=float)[:, :2]
    grid = triangle_grid(points, Delaunay(points).simplices)
    if grid_type == "triangle":
        return grid
    grid = pebi(grid)
    grid = remove_short_edges(grid, min(L) / (max(cart_dims) * 10.0))
    return sort_cell_faces(grid)


def _glued_grid(cart_dims, L, disturb: float, grid_type: str):
    """Glue blocks of differing resolution, and for `mixed*` differing type.

    The trailing digit selects how far the construction goes: 2 adds a third
    block, 3 adds the thin strip and the wide blocks above and below, and 4
    adds the tall blocks on either side.
    """
    level = int(grid_type[-1])
    mixed = grid_type.startswith("mixed")
    nx, ny = cart_dims
    Lx, Ly = L

    if mixed:
        first = square_grid(cart_dims, L, grid_type="cartgrid", disturb=disturb)
        second = square_grid([nx + 1, ny + 1], L, grid_type="pebi",
                             disturb=disturb)
    else:
        first = cart_grid(cart_dims, L)
        second = cart_grid([2 * nx + 1, 2 * ny + 1], L)
    grid = glue_2d_grid(first, translate_grid(second, [Lx, 0.0]))

    if level > 1:
        if mixed:
            third = square_grid([nx + 2, ny + 2], L, grid_type="triangle",
                                disturb=0.02)
        else:
            third = cart_grid([nx + 2, ny + 2], L)
        grid = glue_2d_grid(grid, translate_grid(third, [2.0 * Lx, 0.0]))

    if level > 2:
        wide = cart_grid([nx + 7, ny + 1], [3.0 * Lx, Ly])
        thickness = 0.05
        strip = cart_grid([nx * 5, 1], [3.0 * Lx, thickness])
        grid = glue_2d_grid(grid, translate_grid(strip, [0.0, Ly]))
        grid = glue_2d_grid(grid, translate_grid(wide, [0.0, Ly + thickness]))
        grid = glue_2d_grid(grid, translate_grid(wide, [0.0, -Ly]))

        if level > 3:
            tall_size = [Lx, 3.0 * Ly + thickness]
            if mixed:
                left = square_grid([3, 10], tall_size, grid_type="pebi",
                                   disturb=disturb)
                right = square_grid([3, 10], tall_size, grid_type="triangle",
                                    disturb=disturb)
            else:
                left = cart_grid([3, 10], tall_size)
                right = left
            grid = glue_2d_grid(grid, translate_grid(left, [-Lx, -Ly]))
            grid = glue_2d_grid(grid, translate_grid(right, [3.0 * Lx, -Ly]))

    grid.cells.indexMap = None
    if disturb > 0:
        grid = twister(grid, disturb)
    grid = _rescale(grid, L)
    return sort_cell_faces(grid)


def _rescale(grid, L):
    """Move the glued patchwork back onto the requested rectangle."""
    coords = np.asarray(grid.nodes.coords, dtype=float)
    coords = coords - coords.min(axis=0)
    extent = coords.max(axis=0)
    scale = np.where(extent > 0, np.asarray(L[:coords.shape[1]]) / extent, 1.0)
    grid.nodes.coords = coords * scale
    return grid


def _tag_boundary(grid, cart_dims, L):
    """Tag the outer faces 1-4 by side, as MRST does for non-Cartesian grids."""
    grid = compute_geometry(grid)
    coords = np.asarray(grid.nodes.coords, dtype=float)
    centroids = np.asarray(grid.faces.centroids, dtype=float)
    tol = 1e-3 / max(cart_dims) * min(L)

    tag = np.zeros(grid.faces.num, dtype=np.int64)
    tag[np.abs(centroids[:, 0]) < tol] = 1
    tag[np.abs(centroids[:, 0] - coords[:, 0].max()) < tol] = 2
    tag[np.abs(centroids[:, 1]) < tol] = 3
    tag[np.abs(centroids[:, 1] - coords[:, 1].max()) < tol] = 4
    if grid.griddim == 3:
        # MRST compares the z extremes against the y extent; kept verbatim so
        # a 3-D tagging difference shows up as a mismatch rather than silently
        tag[np.abs(centroids[:, 2]) < tol] = 3
        tag[np.abs(centroids[:, 2] - coords[:, 1].max()) < tol] = 4
    grid.cells.tags = tag[np.asarray(grid.cells.faces, dtype=np.int64)]
    return grid


squareGrid = square_grid


if __name__ == "__main__":

    def test_cartgrid_matches_cart_grid() -> None:
        grid = square_grid([4, 3], [2.0, 1.5])
        assert grid.cells.num == 12
        assert np.isclose(grid.nodes.coords.max(axis=0)[0], 2.0)
        print("ok test_cartgrid_matches_cart_grid")

    def test_cartgrid_supports_three_dimensions() -> None:
        grid = compute_geometry(square_grid([3, 3, 2], [3.0, 3.0, 2.0]))
        assert grid.cells.num == 18 and grid.griddim == 3
        assert np.isclose(grid.cells.volumes.sum(), 18.0)
        print("ok test_cartgrid_supports_three_dimensions")

    def test_disturb_moves_interior_nodes_only() -> None:
        plain = square_grid([4, 4], [1.0, 1.0])
        shaken = square_grid([4, 4], [1.0, 1.0], disturb=0.1)
        assert not np.allclose(plain.nodes.coords, shaken.nodes.coords)
        for axis in (0, 1):
            assert np.isclose(shaken.nodes.coords[:, axis].min(), 0.0)
            assert np.isclose(shaken.nodes.coords[:, axis].max(), 1.0)
        print("ok test_disturb_moves_interior_nodes_only")

    def test_triangle_grid_is_all_triangles() -> None:
        grid = compute_geometry(square_grid([4, 4], [1.0, 1.0],
                                            grid_type="triangle"))
        assert np.all(np.diff(grid.cells.facePos) == 3)
        assert np.isclose(grid.cells.volumes.sum(), 1.0)
        print("ok test_triangle_grid_is_all_triangles")

    def test_pebi_grid_covers_the_domain() -> None:
        grid = compute_geometry(square_grid([5, 5], [1.0, 1.0],
                                            grid_type="pebi"))
        assert grid.cells.num > 0
        assert np.all(grid.cells.volumes > 0)
        print(f"ok test_pebi_grid_covers_the_domain ({grid.cells.num} cells)")

    def test_boundary_tags_cover_all_four_sides() -> None:
        grid = square_grid([4, 4], [1.0, 1.0], grid_type="triangle")
        assert grid.cells.tags is not None
        assert set(np.unique(grid.cells.tags)) >= {0, 1, 2, 3, 4}
        print("ok test_boundary_tags_cover_all_four_sides")

    def test_glued_grids_build_and_fill_the_rectangle() -> None:
        for name in ("boxes1", "boxes2", "mixed1"):
            grid = compute_geometry(square_grid([4, 4], [3.0, 3.0],
                                                grid_type=name))
            assert grid.cells.num > 0, name
            assert np.all(grid.cells.volumes > 0), name
            assert np.isclose(grid.nodes.coords[:, 0].max(), 3.0), name
            assert np.isclose(grid.nodes.coords[:, 1].max(), 3.0), name
            print(f"ok test_glued_grids_build_and_fill_the_rectangle [{name}] "
                  f"({grid.cells.num} cells)")

    def test_unknown_grid_type_is_rejected() -> None:
        try:
            square_grid([2, 2], [1.0, 1.0], grid_type="nope")
        except ValueError:
            print("ok test_unknown_grid_type_is_rejected")
            return
        raise AssertionError("unknown grid type should be rejected")

    def test_three_dimensions_rejects_non_cartesian() -> None:
        try:
            square_grid([2, 2, 2], [1.0, 1.0, 1.0], grid_type="triangle")
        except ValueError:
            print("ok test_three_dimensions_rejects_non_cartesian")
            return
        raise AssertionError("3D non-Cartesian should be rejected")

    test_cartgrid_matches_cart_grid()
    test_cartgrid_supports_three_dimensions()
    test_disturb_moves_interior_nodes_only()
    test_triangle_grid_is_all_triangles()
    test_pebi_grid_covers_the_domain()
    test_boundary_tags_cover_all_four_sides()
    test_glued_grids_build_and_fill_the_rectangle()
    test_unknown_grid_type_is_rejected()
    test_three_dimensions_rejects_non_cartesian()
    print("\nAll tests passed!")
