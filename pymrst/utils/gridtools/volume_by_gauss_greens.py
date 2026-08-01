"""Cell volumes by the divergence theorem (MRST ``utils/gridtools/volumeByGaussGreens.m``).

An *independent* volume computation, which is the point of it: `compute_geometry`
gets volumes by splitting each cell into tetrahedra, and this gets them by
integrating over the boundary instead. Agreement between two unrelated routes
is real evidence the grid is sound; disagreement localises the bad cell.

Applying Gauss-Green to the field `(x, 0, 0)`, whose divergence is 1:

    V = ∫_C div(x,0,0) dV = ∮_∂C x·n dS = Σ_faces R_x · N_x

with `R` the face centroid and `N` the area-weighted outward normal. The same
holds for `(0, y, 0)` and `(0, 0, z)`, so **each coordinate direction gives an
independent estimate** and the result has one column per direction. That
redundancy is the diagnostic - on a well-formed grid all columns agree, and a
column that disagrees says the boundary does not close in that direction.

**It is exact only for planar faces, and that is the whole caveat.** The
boundary integral is evaluated as one centroid times one normal per face, which
is the true integral only if the face is flat. Cartesian, triangular and
tetrahedral grids are fine. A *twisted* hexahedral grid - which is most real
reservoir geometry - has non-planar quadrilateral faces, and there this
disagrees with `compute_geometry` cell by cell: on the twisted grid in the
tests below, by around 0.4%.

The totals still agree to machine precision, and the reason is worth knowing.
Every interior face is shared by two cells and enters their sums with opposite
signs, so whatever error it carries cancels exactly between them. **The error
is a redistribution of volume between neighbours, not a net loss.** So a total
that matches is no evidence the per-cell volumes do, and comparing totals is
the check most likely to be reached for.

That makes the disagreement useful rather than merely tolerable: the spread
across the columns measures how far the faces are from flat. `compute_geometry`
triangulates and gets the volume right regardless; this one is the instrument
that says the faces are warped.

MRST builds two block-diagonal sparse matrices and reads the volumes off the
diagonal of their product. That computes the off-diagonal entries only to throw
them away; here it is the segmented sum it actually is, which is `O(cell-faces)`
rather than `O(cells × cell-faces)`.

**Requires `compute_geometry` to have run** - it reads face centroids and
normals. It cannot check the volumes of a grid that has none.
"""

import numpy as np

from pymrst.utils.rldecode import rldecode


def volume_by_gauss_greens(g) -> np.ndarray:
    """Return ``(num_cells, griddim)`` volume estimates, one per direction."""
    for holder, name in ((g.faces, "centroids"), (g.faces, "normals")):
        if getattr(holder, name, None) is None:
            raise ValueError(
                f"faces.{name} is missing; run compute_geometry before "
                f"volume_by_gauss_greens")

    face_pos = np.asarray(g.cells.facePos, dtype=np.int64)
    cell_faces = np.asarray(g.cells.faces, dtype=np.int64)
    cell_faces = cell_faces[:, 0] if cell_faces.ndim > 1 else cell_faces
    cell_no = rldecode(np.arange(g.cells.num), np.diff(face_pos))

    # A face's stored normal points out of neighbours[0]; flip it for the other.
    neighbours = np.asarray(g.faces.neighbors, dtype=np.int64)
    sign = np.where(neighbours[cell_faces, 1] == cell_no, -1.0, 1.0)

    centroids = np.asarray(g.faces.centroids, dtype=float)[cell_faces]
    normals = np.asarray(g.faces.normals, dtype=float)[cell_faces]
    contribution = centroids * normals * sign[:, None]

    return np.column_stack([
        np.bincount(cell_no, weights=contribution[:, d], minlength=g.cells.num)
        for d in range(g.griddim)])


volumeByGaussGreens = volume_by_gauss_greens


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry

    def test_a_unit_cube_has_unit_volume() -> None:
        g = compute_geometry(cart_grid([3, 2, 2], [3.0, 2.0, 2.0]))
        assert np.allclose(volume_by_gauss_greens(g), 1.0)
        print("ok test_a_unit_cube_has_unit_volume")

    def test_every_direction_gives_the_same_answer() -> None:
        """The redundancy that makes this a check rather than a computation."""
        g = compute_geometry(cart_grid([3, 2, 2], [6.0, 4.0, 2.0]))
        got = volume_by_gauss_greens(g)
        assert np.allclose(got, got[:, :1]), got[:3]
        print("ok test_every_direction_gives_the_same_answer")

    def test_it_agrees_with_compute_geometry() -> None:
        """Two unrelated routes to the same number; the reason to have both."""
        for dims, sizes in (([3, 2, 2], [6.0, 4.0, 2.0]),
                            ([4, 4], [4.0, 4.0]),
                            ([2, 3, 4], [1.0, 1.0, 1.0])):
            g = compute_geometry(cart_grid(dims, sizes))
            got = volume_by_gauss_greens(g)[:, 0]
            assert np.allclose(got, np.asarray(g.cells.volumes)), dims
        print("ok test_it_agrees_with_compute_geometry")

    def twisted(seed=3):
        """A grid whose interior faces are no longer flat."""
        g = cart_grid([4, 4, 2], [4.0, 4.0, 2.0])
        coords = np.asarray(g.nodes.coords, dtype=float).copy()
        rng = np.random.default_rng(seed)
        interior = np.all((coords > 1e-9)
                          & (coords < np.array([4.0, 4.0, 2.0]) - 1e-9), axis=1)
        coords[interior] += rng.uniform(-0.2, 0.2, coords[interior].shape)
        g.nodes.coords = coords
        return compute_geometry(g)

    def test_non_planar_faces_break_the_per_cell_agreement() -> None:
        """The caveat, asserted rather than described; see the docstring."""
        g = twisted()
        got = volume_by_gauss_greens(g)[:, 0]
        volumes = np.asarray(g.cells.volumes, dtype=float)
        worst = np.max(np.abs(got - volumes) / volumes)
        assert worst > 1e-6, \
            f"expected a visible per-cell disagreement, got {worst:.2e}"
        assert worst < 0.05, f"but not a wild one: {worst:.2e}"
        print(f"ok test_non_planar_faces_break_the_per_cell_agreement "
              f"({worst:.2%})")

    def test_the_total_survives_because_the_errors_cancel() -> None:
        """Why a total-only check would miss the problem above."""
        g = twisted()
        got = volume_by_gauss_greens(g)[:, 0]
        volumes = np.asarray(g.cells.volumes, dtype=float)
        assert np.isclose(np.sum(got), np.sum(volumes), rtol=1e-12)
        assert np.isclose(np.sum(got), 32.0, rtol=1e-12)
        print("ok test_the_total_survives_because_the_errors_cancel")

    def test_the_column_spread_measures_non_planarity() -> None:
        """Flat faces agree across directions; warped ones do not."""
        flat = compute_geometry(cart_grid([4, 4, 2], [4.0, 4.0, 2.0]))
        for g, want in ((flat, False), (twisted(), True)):
            got = volume_by_gauss_greens(g)
            spread = np.max(np.abs(got - got[:, :1]))
            assert (spread > 1e-6) == want, (spread, want)
        print("ok test_the_column_spread_measures_non_planarity")

    def test_it_agrees_in_two_dimensions() -> None:
        g = compute_geometry(cart_grid([3, 3], [3.0, 3.0]))
        got = volume_by_gauss_greens(g)
        assert got.shape == (g.cells.num, 2)
        assert np.allclose(got[:, 0], np.asarray(g.cells.volumes))
        print("ok test_it_agrees_in_two_dimensions")

    def test_it_agrees_on_a_triangle_grid() -> None:
        from pymrst.gridprocessing import triangle_grid
        points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
                           [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
        triangles = np.array([[0, 1, 3], [1, 4, 3], [1, 2, 4], [2, 5, 4]])
        g = compute_geometry(triangle_grid(points, triangles))
        got = volume_by_gauss_greens(g)
        assert np.allclose(got[:, 0], np.asarray(g.cells.volumes))
        print("ok test_it_agrees_on_a_triangle_grid")

    def test_the_total_is_the_domain_volume() -> None:
        g = compute_geometry(cart_grid([5, 4, 3], [10.0, 8.0, 6.0]))
        assert np.isclose(np.sum(volume_by_gauss_greens(g)[:, 0]), 10 * 8 * 6)
        print("ok test_the_total_is_the_domain_volume")

    def test_a_broken_normal_is_caught() -> None:
        """Guard: the agreement tests must be able to fail.

        The face is chosen by its contribution, not taken as face 0. The
        `x`-column is blind to any face lying in the plane `x = 0`, where the
        centroid's `x` is zero and multiplies the normal away - and on a grid
        with a corner at the origin, face 0 is exactly such a face.
        """
        g = compute_geometry(cart_grid([2, 2, 2], [2.0, 2.0, 2.0]))
        centroids = np.asarray(g.faces.centroids, dtype=float)
        normals = np.asarray(g.faces.normals, dtype=float).copy()
        worst = int(np.argmax(np.abs(centroids[:, 0] * normals[:, 0])))
        assert abs(centroids[worst, 0] * normals[worst, 0]) > 1e-9
        normals[worst] *= 3.0
        g.faces.normals = normals
        assert not np.allclose(volume_by_gauss_greens(g)[:, 0],
                               np.asarray(g.cells.volumes))
        print("ok test_a_broken_normal_is_caught")

    def test_a_face_in_the_coordinate_plane_is_invisible_to_that_column() -> None:
        """Stated because the test above depends on it, and because it is the
        reason to read all the columns rather than just the first."""
        g = compute_geometry(cart_grid([2, 2, 2], [2.0, 2.0, 2.0]))
        centroids = np.asarray(g.faces.centroids, dtype=float)
        normals = np.asarray(g.faces.normals, dtype=float).copy()
        blind = np.flatnonzero(np.abs(centroids[:, 0]) < 1e-12)
        assert blind.size, "expected faces in the plane x = 0"
        before = volume_by_gauss_greens(g)[:, 0]
        normals[blind[0]] *= 3.0
        g.faces.normals = normals
        assert np.allclose(volume_by_gauss_greens(g)[:, 0], before)
        print("ok test_a_face_in_the_coordinate_plane_is_invisible_to_that_column")

    def test_a_grid_without_geometry_is_rejected() -> None:
        g = cart_grid([2, 2], [2.0, 2.0])
        try:
            volume_by_gauss_greens(g)
        except ValueError as error:
            assert "compute_geometry" in str(error)
            print("ok test_a_grid_without_geometry_is_rejected")
            return
        raise AssertionError("a grid without face geometry must be rejected")

    test_a_unit_cube_has_unit_volume()
    test_every_direction_gives_the_same_answer()
    test_it_agrees_with_compute_geometry()
    test_non_planar_faces_break_the_per_cell_agreement()
    test_the_total_survives_because_the_errors_cancel()
    test_the_column_spread_measures_non_planarity()
    test_it_agrees_in_two_dimensions()
    test_it_agrees_on_a_triangle_grid()
    test_the_total_is_the_domain_volume()
    test_a_broken_normal_is_caught()
    test_a_face_in_the_coordinate_plane_is_invisible_to_that_column()
    test_a_grid_without_geometry_is_rejected()
    print("\nAll tests passed!")
