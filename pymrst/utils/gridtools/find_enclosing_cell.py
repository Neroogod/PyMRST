"""Which cell contains a point (MRST ``utils/gridtools/findEnclosingCell.m``).

A point is inside a cell when it lies on the inward side of every one of that
cell's faces. The test is per-face and vectorised over cells, so it needs no
triangulation and works on a general polyhedron.

**It assumes convex cells**, because "inside every face plane" and "inside the
cell" are the same statement only then. A non-convex cell's re-entrant corner
puts points outside the cell on the inward side of all its planes, so the
answer is wrong rather than approximate. MRST has the same assumption and does
not state it.

A point on a shared face satisfies both neighbours; the **lowest** cell index
wins, which is arbitrary but deterministic. A point outside every cell returns
-1 - MRST returns 0, which here would name a real cell.
"""

import numpy as np

from pymrst.utils.mcolon import mcolon
from pymrst.utils.rldecode import rldecode


def find_enclosing_cell(g, points, cells=None) -> np.ndarray:
    """Return the enclosing cell of each point, or -1 where there is none."""
    if cells is None:
        cells = np.arange(g.cells.num)
    else:
        cells = np.asarray(cells)
        cells = (np.flatnonzero(cells) if cells.dtype == bool
                 else np.atleast_1d(cells.astype(np.int64)).ravel())
        if cells.size and (cells.min() < 0 or cells.max() >= g.cells.num):
            raise ValueError(
                f"cell subset must lie in [0, {g.cells.num - 1}]")

    points = np.atleast_2d(np.asarray(points, dtype=float))
    dim = g.griddim

    face_pos = np.asarray(g.cells.facePos, dtype=np.int64)
    all_faces = np.asarray(g.cells.faces, dtype=np.int64)
    all_faces = all_faces[:, 0] if all_faces.ndim > 1 else all_faces

    faces = all_faces[mcolon(face_pos[cells], face_pos[cells + 1])]
    owner = rldecode(cells, face_pos[cells + 1] - face_pos[cells])

    normals = np.asarray(g.faces.normals, dtype=float)[faces, :dim]
    face_centroids = np.asarray(g.faces.centroids, dtype=float)[faces, :dim]
    cell_centroids = np.asarray(g.cells.centroids, dtype=float)[owner, :dim]

    # Orient every normal outward from its own cell.
    sign = np.sign(np.sum((face_centroids - cell_centroids) * normals, axis=1))
    normals = normals * sign[:, None]

    out = np.full(points.shape[0], -1, dtype=np.int64)
    for k, point in enumerate(points[:, :dim]):
        inside = np.sum(normals * (face_centroids - point), axis=1) >= 0.0
        per_cell = np.bincount(owner, weights=inside, minlength=g.cells.num)
        needed = np.bincount(owner, minlength=g.cells.num)
        enclosing = np.flatnonzero((needed > 0) & (per_cell == needed))
        if enclosing.size:
            out[k] = enclosing.min()
    return out


findEnclosingCell = find_enclosing_cell


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry

    G = compute_geometry(cart_grid([3, 3], [3.0, 3.0]))
    G3 = compute_geometry(cart_grid([2, 2, 2], [2.0, 2.0, 2.0]))

    def test_a_cell_centroid_lands_in_its_own_cell() -> None:
        centroids = np.asarray(G.cells.centroids)[:, :2]
        found = find_enclosing_cell(G, centroids)
        assert np.array_equal(found, np.arange(G.cells.num)), found
        print("ok test_a_cell_centroid_lands_in_its_own_cell")

    def test_it_works_in_three_dimensions() -> None:
        centroids = np.asarray(G3.cells.centroids)
        assert np.array_equal(find_enclosing_cell(G3, centroids),
                              np.arange(G3.cells.num))
        print("ok test_it_works_in_three_dimensions")

    def test_a_point_outside_the_grid_returns_minus_one() -> None:
        """MRST returns 0, which here is a real cell."""
        assert find_enclosing_cell(G, [[-5.0, -5.0]])[0] == -1
        assert find_enclosing_cell(G, [[99.0, 99.0]])[0] == -1
        print("ok test_a_point_outside_the_grid_returns_minus_one")

    def test_an_interior_point_lands_in_the_expected_cell() -> None:
        """Cell 4 of a 3x3 unit grid spans [1,2] x [1,2]."""
        assert find_enclosing_cell(G, [[1.5, 1.5]])[0] == 4
        print("ok test_an_interior_point_lands_in_the_expected_cell")

    def test_a_point_on_a_shared_face_takes_the_lower_index() -> None:
        """Both neighbours satisfy the test; the tie is broken deterministically."""
        found = find_enclosing_cell(G, [[1.0, 0.5]])[0]
        assert found == 0, found
        print("ok test_a_point_on_a_shared_face_takes_the_lower_index")

    def test_several_points_are_answered_at_once() -> None:
        found = find_enclosing_cell(G, [[0.5, 0.5], [2.5, 2.5], [-1.0, 0.0]])
        assert found.tolist() == [0, 8, -1], found
        print("ok test_several_points_are_answered_at_once")

    def test_a_subset_limits_the_search() -> None:
        """The point is in cell 4, but cell 4 is not offered."""
        assert find_enclosing_cell(G, [[1.5, 1.5]], cells=[0, 1, 2])[0] == -1
        assert find_enclosing_cell(G, [[1.5, 1.5]], cells=[3, 4])[0] == 4
        print("ok test_a_subset_limits_the_search")

    def test_a_logical_subset_is_accepted() -> None:
        mask = np.zeros(G.cells.num, dtype=bool)
        mask[4] = True
        assert find_enclosing_cell(G, [[1.5, 1.5]], cells=mask)[0] == 4
        print("ok test_a_logical_subset_is_accepted")

    def test_it_works_on_a_twisted_grid() -> None:
        """Convex cells, but not axis-aligned - the planes really are used."""
        g = cart_grid([4, 4], [4.0, 4.0])
        coords = np.asarray(g.nodes.coords, dtype=float)
        x, y = coords[:, 0].copy(), coords[:, 1].copy()
        coords[:, 0] += 0.2 * np.sin(np.pi * x / 4) * np.sin(np.pi * y / 4)
        g.nodes.coords = coords
        g = compute_geometry(g)
        centroids = np.asarray(g.cells.centroids)[:, :2]
        assert np.array_equal(find_enclosing_cell(g, centroids),
                              np.arange(g.cells.num))
        print("ok test_it_works_on_a_twisted_grid")

    def test_an_out_of_range_subset_is_rejected() -> None:
        try:
            find_enclosing_cell(G, [[0.5, 0.5]], cells=[999])
        except ValueError as error:
            assert "must lie in" in str(error)
            print("ok test_an_out_of_range_subset_is_rejected")
            return
        raise AssertionError("an out-of-range cell must be rejected")

    test_a_cell_centroid_lands_in_its_own_cell()
    test_it_works_in_three_dimensions()
    test_a_point_outside_the_grid_returns_minus_one()
    test_an_interior_point_lands_in_the_expected_cell()
    test_a_point_on_a_shared_face_takes_the_lower_index()
    test_several_points_are_answered_at_once()
    test_a_subset_limits_the_search()
    test_a_logical_subset_is_accepted()
    test_it_works_on_a_twisted_grid()
    test_an_out_of_range_subset_is_rejected()
    print("\nAll tests passed!")
