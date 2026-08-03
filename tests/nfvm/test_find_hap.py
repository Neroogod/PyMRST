"""Harmonic averaging points (MRST ``findHAP``).

Written because F0.4's risk ranking put this module high with no focused test.
It is the geometric heart of the NFVM scheme: every flux the method assembles
is interpolated at these points, so a wrong coordinate does not raise, it
quietly changes the discretization.
"""

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.nfvm.solvers.utils.find_hap import find_hap
from pymrst.params.rock import make_rock


def _grid(dims=(3, 3), size=(3.0, 3.0)):
    return compute_geometry(cart_grid(list(dims), list(size)))


def _rock(g, perm):
    return make_rock(g, perm=np.asarray(perm, dtype=float),
                     poro=np.full(g.cells.num, 0.2))


def _internal(g):
    return np.all(np.asarray(g.faces.neighbors) >= 0, axis=1)


def test_isotropic_permeability_puts_every_point_on_its_face_centroid():
    """The one configuration with an answer known without the algorithm.

    With K = kI on a Cartesian grid, Kn is parallel to n and the construction
    collapses: x_A and x_B both land on the face centroid, from either side.
    Any error in the projection - a transposed tensor, a normal that is not
    area-weighted consistently, a sign - moves the point off it.
    """
    g = _grid()
    hap = find_hap(g, _rock(g, np.full(g.cells.num, 1e-13)))
    np.testing.assert_allclose(
        hap.coords, np.asarray(g.faces.centroids, dtype=float), atol=1e-14,
    )


def test_what_actually_moves_a_point_off_its_face_centroid():
    """The control for the test above, and two facts worth stating.

    A function returning face centroids unconditionally passes the isotropic
    case perfectly, so something has to move the points. Two plausible
    candidates do not, and finding that out is most of the value here:

    **Diagonal anisotropy moves nothing.** On an axis-aligned grid a face
    normal is a unit axis vector, so with `kxy = 0` the product `Kn` stays
    parallel to `n` and both projections land back on the centroid. kxx = 1e-13
    against kyy = 4e-13 gives displacement 2e-16.

    **Off-diagonal terms move nothing either, on the interior.** With the same
    tensor either side, x_A and x_B are mirror images about the face and carry
    equal weights, so their average is the centroid again. Only the boundary
    faces move, because they have a single one-sided projection and nothing to
    cancel against.

    Interior points move under **heterogeneity** — a different tensor on each
    side — which is the case NFVM exists for and the one a two-point scheme
    gets wrong. Interior and boundary are asserted separately because separate
    branches compute them: taking the maximum over all faces let a stub that
    returned centroids for every interior face pass on boundary movement alone.
    """
    g = _grid()
    centroids = np.asarray(g.faces.centroids, dtype=float)
    interior = _internal(g)

    def displacement(perm):
        coords = find_hap(g, _rock(g, perm)).coords
        return np.linalg.norm(coords - centroids, axis=1)

    diagonal = np.tile([1e-13, 0.0, 4e-13], (g.cells.num, 1))    # kxx, kxy, kyy
    np.testing.assert_allclose(displacement(diagonal), 0.0, atol=1e-14)

    uniform_tilt = np.tile([1e-13, 5e-14, 4e-13], (g.cells.num, 1))
    np.testing.assert_allclose(
        displacement(uniform_tilt)[interior], 0.0, atol=1e-14,
    )
    assert displacement(uniform_tilt)[~interior].max() > 1e-3, (
        "a one-sided boundary projection should move under off-diagonal K"
    )

    heterogeneous = np.tile([1e-13, 5e-14, 4e-13], (g.cells.num, 1))
    heterogeneous[::2] = [4e-13, -5e-14, 1e-13]
    assert displacement(heterogeneous)[interior].max() > 1e-3, (
        "a heterogeneous anisotropic medium left every interior point on its "
        "centroid"
    )


def test_interior_weights_are_a_partition_and_symmetric_under_symmetry():
    g = _grid()
    hap = find_hap(g, _rock(g, np.full(g.cells.num, 1e-13)))
    interior = _internal(g)
    np.testing.assert_allclose(hap.weights[interior].sum(axis=1), 1.0)
    # Identical cells either side: neither may be preferred.
    np.testing.assert_allclose(hap.weights[interior], 0.5)


def test_a_boundary_face_puts_all_its_weight_on_the_cell_that_exists():
    """Which column carries the 1 depends on which neighbour is absent.

    Writing the weight to a fixed column would silently credit the missing
    cell on half the boundary.
    """
    g = _grid()
    hap = find_hap(g, _rock(g, np.full(g.cells.num, 1e-13)))
    neighbors = np.asarray(g.faces.neighbors)
    boundary = ~_internal(g)
    owner = np.where(neighbors[boundary, 0] >= 0, 0, 1)
    rows = np.flatnonzero(boundary)
    np.testing.assert_allclose(hap.weights[rows, owner], 1.0)
    np.testing.assert_allclose(hap.weights[rows, 1 - owner], 0.0)
    assert np.any(owner == 0) and np.any(owner == 1), (
        "this grid does not exercise both boundary orientations"
    )


def test_counter_and_fraction_report_cells_whose_centroid_escapes_its_points():
    """`fraction` is the share of cells whose centroid is *outside* the hull.

    The sense is easy to invert, and an inverted one reads as a healthy grid
    being entirely degenerate. On a Cartesian grid with isotropic K the points
    are the face centroids, so every cell centroid is strictly inside and the
    fraction is zero.
    """
    g = _grid()
    hap = find_hap(g, _rock(g, np.full(g.cells.num, 1e-13)))
    assert hap.counter.dtype == bool
    assert hap.counter.shape == (g.cells.num,)
    assert hap.counter.all(), "a centroid escaped the hull of its face centroids"
    assert hap.fraction == 0.0
    np.testing.assert_allclose(hap.fraction, 1.0 - hap.counter.mean())


def test_shapes_follow_the_grid_in_three_dimensions_too():
    g = compute_geometry(cart_grid([2, 2, 2], [2.0, 2.0, 2.0]))
    hap = find_hap(g, _rock(g, np.full(g.cells.num, 1e-13)))
    assert hap.coords.shape == (g.faces.num, 3)
    assert hap.weights.shape == (g.faces.num, 2)
    np.testing.assert_allclose(
        hap.coords, np.asarray(g.faces.centroids, dtype=float), atol=1e-14,
    )
    np.testing.assert_allclose(hap.weights[_internal(g)].sum(axis=1), 1.0)
