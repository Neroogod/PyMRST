"""MRST's 2-D/3-D WENO reconstruction (`WENOUpwindDiscretizationAD`), ported.

There is no directional stencil to walk on an unstructured grid, so MRST builds
a local triangulation around each cell - the cell centroid plus its face
neighbours' centroids - and each simplex touching the cell carries one linear
reconstruction. The candidates are blended by smoothness times simplex volume.

The anchor test is **linear exactness**: the reconstruction is linear on each
simplex, so any convex blend of exact candidates is exact, and a linear field
must be reproduced to machine precision. That single property pins the gradient
basis, the local coordinate frame, the scaling, the offset `dx`, and the weight
normalisation all at once - if any one of them is wrong the field bends.

It holds on *interior* faces only, and the exception is the interesting part.
A boundary ghost point carries its interior cell's value, so a simplex reaching
across the boundary sees a **flat** field, and `beta = (|grad q|^2 + 1e-12)^-2`
gives that flat simplex a weight ~1e23 times its neighbours'. WENO picks the
smoothest stencil, and at a boundary the smoothest stencil is the constant one.
So the scheme degrades to first order there, by construction rather than by
accident - `test_the_scheme_degrades_to_first_order_at_a_boundary` pins that,
because a future change that "fixed" it would be changing the scheme.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_core.statefunctions.flux.upwind.weno import (
    WENOUpwindDiscretizationAD, get_basis, project_points,
)
from pymrst.modules.ad_core.utils.operators import setup_operators_tpfa
from pymrst.params.rock import make_rock
from pymrst.utils.adi import init_variables_adi, value


def _model(dims, extent):
    g = compute_geometry(cart_grid(list(dims), [float(e) for e in extent]))
    return SimpleNamespace(G=g,
                           operators=setup_operators_tpfa(g, make_rock(g, 1e-13, 0.2)))


def _faces(model, dim):
    internal = np.all(np.asarray(model.G.faces.neighbors) >= 0, axis=1)
    return np.asarray(model.G.faces.centroids)[internal, :dim]


def _skewed_model(n=5, grade=1.6, theta=0.4):
    """A graded, rotated grid - the one that makes three properties testable.

    On a Cartesian grid the volume weight is a constant that cancels in the
    normalisation, and the SVD frame is a signed permutation, which is its own
    inverse. Both make real defects invisible. Grading spreads the simplex
    volumes (~2x here) and rotating makes the frame a genuine rotation.
    """
    g = cart_grid([n, n, 1], [float(n), float(n), 1.0])
    g.nodes.coords[:, 0] = g.nodes.coords[:, 0]**grade
    c, s = np.cos(theta), np.sin(theta)
    g.nodes.coords[:, :2] = g.nodes.coords[:, :2] @ np.array([[c, s], [-s, c]])
    g = compute_geometry(g)
    return SimpleNamespace(G=g,
                           operators=setup_operators_tpfa(g, make_rock(g, 1e-13, 0.2)))


def _skewed_3d_model(n=4, grade=1.5, shear=0.25, theta=0.4):
    """A sheared, rotated 3-D grid: the frames here are genuinely
    non-symmetric, which 2-D reflections are not."""
    g = cart_grid([n, n, n], [float(n)]*3)
    g.nodes.coords[:, 0] = g.nodes.coords[:, 0]**grade
    g.nodes.coords[:, 1] += shear*g.nodes.coords[:, 0]
    c, s = np.cos(theta), np.sin(theta)
    g.nodes.coords[:] = g.nodes.coords @ np.array([[c, -s, 0.0],
                                                   [s, c, 0.0],
                                                   [0.0, 0.0, 1.0]]).T
    g = compute_geometry(g)
    return SimpleNamespace(G=g,
                           operators=setup_operators_tpfa(g, make_rock(g, 1e-13, 0.2)))


def _upstream_on_boundary(model, upstream, axis):
    """Cells whose ghost point flattens the field along `axis`."""
    c = np.asarray(model.G.cells.centroids)[:, axis]
    return (c[upstream] <= c.min() + 1e-12) | (c[upstream] >= c.max() - 1e-12)


# -- the anchor -------------------------------------------------------------

@pytest.mark.parametrize("dim, dims, extent", [
    (2, [6, 6, 1], [6.0, 6.0, 1.0]),
    (3, [4, 4, 4], [4.0, 4.0, 4.0]),
])
def test_a_linear_field_is_reproduced_exactly_on_interior_faces(dim, dims, extent):
    """Machine precision, not a tolerance.

    The reconstruction is linear on each simplex and the blend is convex, so
    there is no approximation left to make. A tolerance here would hide a
    genuinely wrong basis or a mis-scaled offset.
    """
    model = _model(dims, extent)
    weno = WENOUpwindDiscretizationAD(model, dim=dim)
    n = np.asarray(model.operators.N)
    flag = np.ones(n.shape[0], dtype=bool)
    upstream = n[:, 0]
    fcs = _faces(model, dim)
    centroids = np.asarray(model.G.cells.centroids)

    for axis in range(dim):
        q = centroids[:, axis] + 10.0
        got = np.asarray(weno.face_upstream(model, None, flag, q))
        interior = ~_upstream_on_boundary(model, upstream, axis)
        np.testing.assert_allclose(got[interior], fcs[interior, axis] + 10.0,
                                   rtol=0, atol=1e-12)


def test_without_ghost_points_every_face_is_exact():
    """Confirms the boundary exception is the ghost points and nothing else."""
    model = _model([6, 6, 1], [6.0, 6.0, 1.0])
    weno = WENOUpwindDiscretizationAD(model, dim=2, include_boundary=False)
    n = np.asarray(model.operators.N)
    q = np.asarray(model.G.cells.centroids)[:, 0] + 10.0

    got = np.asarray(weno.face_upstream(model, None,
                                        np.ones(n.shape[0], dtype=bool), q))
    np.testing.assert_allclose(got, _faces(model, 2)[:, 0] + 10.0,
                               rtol=0, atol=1e-12)


def test_the_scheme_degrades_to_first_order_at_a_boundary():
    """Pinned deliberately: this is the scheme, not a defect.

    The ghost point carries its cell's own value, so a simplex crossing the
    boundary sees a flat field and wins the smoothness contest outright. A
    change that made this exact would be changing WENO into something else.
    """
    model = _model([6, 6, 1], [6.0, 6.0, 1.0])
    weno = WENOUpwindDiscretizationAD(model, dim=2)
    n = np.asarray(model.operators.N)
    q = np.asarray(model.G.cells.centroids)[:, 0] + 10.0

    got = np.asarray(weno.face_upstream(model, None,
                                        np.ones(n.shape[0], dtype=bool), q))
    on_boundary = _upstream_on_boundary(model, n[:, 0], 0)
    assert on_boundary.any()
    # First order means: the upstream cell value, unmodified.
    np.testing.assert_allclose(got[on_boundary], q[n[on_boundary, 0]],
                               rtol=0, atol=1e-12)


# -- monotonicity, the reason WENO exists -----------------------------------

def test_a_step_is_not_overshot():
    """The point of the scheme. A second-order reconstruction that overshoots
    a saturation front puts a value outside the table it came from."""
    model = _model([10, 10, 1], [10.0, 10.0, 1.0])
    weno = WENOUpwindDiscretizationAD(model, dim=2)
    n = np.asarray(model.operators.N)
    x = np.asarray(model.G.cells.centroids)[:, 0]
    q = np.where(x < 5.0, 1.0, 0.1)

    got = np.asarray(weno.face_upstream(model, None,
                                        np.ones(n.shape[0], dtype=bool), q))
    assert got.min() >= q.min() - 1e-12, "undershoot below the data range"
    assert got.max() <= q.max() + 1e-12, "overshoot above the data range"


def test_the_cap_falls_back_rather_than_clipping():
    """A non-positive reconstruction becomes the upstream *cell* value, not
    zero. Clipping would invent a number the scheme never produced."""
    model = _model([6, 6, 1], [6.0, 6.0, 1.0])
    weno = WENOUpwindDiscretizationAD(model, dim=2)
    n = np.asarray(model.operators.N)
    flag = np.ones(n.shape[0], dtype=bool)
    q = np.asarray(model.G.cells.centroids)[:, 0] - 3.0   # straddles zero

    got = np.asarray(weno.face_upstream(model, None, flag, q))
    capped = got <= 0
    if capped.any():
        np.testing.assert_allclose(got[capped], q[n[capped, 0]],
                                   rtol=0, atol=1e-12)


# -- AD ---------------------------------------------------------------------

def test_the_reconstruction_carries_a_jacobian():
    model = _model([5, 5, 1], [5.0, 5.0, 1.0])
    weno = WENOUpwindDiscretizationAD(model, dim=2)
    n = np.asarray(model.operators.N)
    flag = np.ones(n.shape[0], dtype=bool)
    q0 = np.asarray(model.G.cells.centroids)[:, 0] + 10.0

    q_ad, = init_variables_adi(q0.copy())
    out = weno.face_upstream(model, None, flag, q_ad)

    assert hasattr(out, "jac"), "AD did not survive the reconstruction"
    assert out.jac[0].shape == (n.shape[0], model.G.cells.num)
    np.testing.assert_allclose(
        value(out), np.asarray(weno.face_upstream(model, None, flag, q0)))


def test_the_jacobian_matches_a_finite_difference():
    """The smoothness weights depend on `q` and are differentiated, as in MRST.

    The probe field is **random**, and that is not incidental. On a linear
    field the boundary simplices are exactly flat, so `sigma = 0` and
    `beta = (sigma^2 + E)^-2` sits at its peak: the derivative there is zero,
    but the curvature scale is `sqrt(E) = 1e-6`, so a 1e-6 finite-difference
    step leaves the linear regime entirely and disagrees with *any* correct
    Jacobian by ~13%. A random field keeps every stencil non-degenerate and
    the comparison meaningful.

    That distinction cost a wrong diagnosis once already: the disagreement
    looked like a missing weight derivative and was the step size.
    """
    model = _model([5, 5, 1], [5.0, 5.0, 1.0])
    weno = WENOUpwindDiscretizationAD(model, dim=2)
    n = np.asarray(model.operators.N)
    flag = np.ones(n.shape[0], dtype=bool)

    rng = np.random.default_rng(0)
    q0 = 10.0 + rng.random(model.G.cells.num)

    q_ad, = init_variables_adi(q0.copy())
    J = weno.face_upstream(model, None, flag, q_ad).jac[0].toarray()

    dq = rng.normal(size=q0.size)*1e-6
    central = (np.asarray(weno.face_upstream(model, None, flag, q0 + dq))
               - np.asarray(weno.face_upstream(model, None, flag, q0 - dq)))/2
    np.testing.assert_allclose(J @ dq, central, rtol=1e-5,
                               atol=1e-6*np.abs(central).max())


def test_the_weight_derivative_is_part_of_the_jacobian():
    """Freezing the smoothness weights would leave an analytic Jacobian that
    disagrees with its own function - invisible in the residual, paid for in
    Newton iterations. This is the term that catches it."""
    model = _model([5, 5, 1], [5.0, 5.0, 1.0])
    weno = WENOUpwindDiscretizationAD(model, dim=2)
    n = np.asarray(model.operators.N)
    flag = np.ones(n.shape[0], dtype=bool)
    rng = np.random.default_rng(3)
    q0 = 10.0 + rng.random(model.G.cells.num)

    q_ad, = init_variables_adi(q0.copy())
    J = weno.face_upstream(model, None, flag, q_ad).jac[0].toarray()

    # The reconstruction is q_c + sum(w * grad.dx). With the weights frozen
    # the Jacobian would be exactly the same expression evaluated at fixed w,
    # so a row would sum to 1 in both cases - but the individual entries move.
    # Compare against a directional derivative the weights genuinely affect.
    dq = rng.normal(size=q0.size)*1e-7
    central = (np.asarray(weno.face_upstream(model, None, flag, q0 + dq))
               - np.asarray(weno.face_upstream(model, None, flag, q0 - dq)))/2
    err = np.abs(J @ dq - central).max()/np.abs(central).max()
    assert err < 1e-4, f"Jacobian inconsistent with its function ({err:.2e})"


# -- the pieces -------------------------------------------------------------

def test_the_gradient_basis_reproduces_a_known_gradient():
    """`basis[d][i, l]` is the coefficient of vertex value `l` in the
    gradient along `d`. For the unit triangle the gradient of a linear field
    is recovered exactly."""
    weno = WENOUpwindDiscretizationAD(None, dim=2)
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    basis = weno.get_gradient_basis(np.array([[0, 1, 2]]), pts)

    q = 3.0*pts[:, 0] - 7.0*pts[:, 1] + 2.0
    assert basis[0][0] @ q == pytest.approx(3.0)
    assert basis[1][0] @ q == pytest.approx(-7.0)


def test_a_degenerate_simplex_contributes_no_gradient():
    """Collinear vertices carry no directional information. MATLAB's `inv`
    gives Inf and MRST zeroes it; a zero row is the same answer."""
    weno = WENOUpwindDiscretizationAD(None, dim=2)
    collinear = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    basis = weno.get_gradient_basis(np.array([[0, 1, 2]]), collinear)
    assert np.all(np.isfinite(basis[0])) and np.all(basis[0] == 0)


def test_a_repeated_vertex_is_treated_as_an_upwind_triangle():
    weno = WENOUpwindDiscretizationAD(None, dim=2)
    basis = weno.get_gradient_basis(np.array([[2, 2, 2]]),
                                    np.zeros((3, 2)))
    assert np.all(basis[0] == 0) and np.all(basis[1] == 0)


def test_simplex_volumes_are_the_familiar_ones():
    weno2 = WENOUpwindDiscretizationAD(None, dim=2)
    tri = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 3.0]])
    vol, cent = weno2.get_tri_volume(np.array([[0, 1, 2]]), tri)
    assert vol[0] == pytest.approx(3.0)                       # 1/2 * 2 * 3
    np.testing.assert_allclose(cent[0], [2/3, 1.0])

    weno3 = WENOUpwindDiscretizationAD(None, dim=3)
    tet = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    vol3, _ = weno3.get_tri_volume(np.array([[0, 1, 2, 3]]), tet)
    assert vol3[0] == pytest.approx(1/6)


def test_volumes_are_never_negative():
    """The weight must not depend on vertex ordering: Delaunay does not
    promise an orientation, and a negative weight would subtract a stencil."""
    weno = WENOUpwindDiscretizationAD(None, dim=3)
    tet = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    forward, _ = weno.get_tri_volume(np.array([[0, 1, 2, 3]]), tet)
    flipped, _ = weno.get_tri_volume(np.array([[0, 2, 1, 3]]), tet)
    assert forward[0] == pytest.approx(flipped[0]) and forward[0] > 0


def test_projection_shifts_then_rotates_then_scales():
    """Order matters: scaling first would stretch the axes the rotation is
    about to mix, and the frame would no longer be the SVD frame."""
    pts = np.array([[2.0, 1.0]])
    center = np.array([1.0, 1.0])
    M = np.array([[0.0, 1.0], [-1.0, 0.0]])      # rotate by -90 degrees
    out = project_points(pts, center, M, np.array([1.0, 2.0]))
    np.testing.assert_allclose(out, [[0.0, -0.5]])


def test_get_basis_matches_the_gradient_it_encodes():
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    basis = get_basis(pts, 3)
    assert basis.shape == (2, 3)
    q = 5.0*pts[:, 0] + 2.0
    assert basis[0] @ q == pytest.approx(5.0)


def test_scale_points_maps_into_the_unit_box():
    pts = np.array([[1.0, 10.0], [3.0, 20.0]])
    scaled, rng, mv = WENOUpwindDiscretizationAD.scale_points(pts)
    np.testing.assert_allclose(scaled, [[0.0, 0.0], [1.0, 1.0]])
    np.testing.assert_allclose(rng, [2.0, 10.0])
    np.testing.assert_allclose(mv, [1.0, 10.0])


def test_a_zero_range_does_not_divide_by_zero():
    """A flat direction is a real grid, not a bug - a single-layer 3-D model
    has one."""
    scaled, rng, _ = WENOUpwindDiscretizationAD.scale_points(
        np.array([[1.0, 5.0], [3.0, 5.0]]))
    assert np.all(np.isfinite(scaled)) and rng[1] == 1.0


# -- setup and structure ----------------------------------------------------

def test_every_cell_gets_at_least_one_simplex():
    """A cell with no support silently reconstructs to first order."""
    model = _model([5, 5, 2], [5.0, 5.0, 2.0])
    weno = WENOUpwindDiscretizationAD(model, dim=3)
    assert np.all(weno.interp_setup.cell_support_count > 0)


def test_the_basis_matrices_agree_with_the_per_simplex_bases():
    """`tri_basis_matrices` is the same operator as the indexed accumulation
    it replaces - MRST keeps both branches and this is what makes them one."""
    model = _model([4, 4, 1], [4.0, 4.0, 1.0])
    weno = WENOUpwindDiscretizationAD(model, dim=2)
    s = weno.interp_setup
    q = np.arange(float(model.G.cells.num)) + 1.0

    for d in range(2):
        fast = s.tri_basis_matrices[d] @ q
        slow = np.zeros(s.C.shape[0])
        for l in range(s.C.shape[1]):
            slow += s.tri_basis[d][:, l]*q[s.tri_cells[s.C[:, l]]]
        np.testing.assert_allclose(fast, slow, rtol=1e-12)


def test_the_dimension_defaults_to_the_grids():
    """A 3-D model given no dimension must not silently get the 1-D scheme,
    which reads the x-coordinate alone."""
    model = _model([3, 3, 3], [3.0, 3.0, 3.0])
    assert WENOUpwindDiscretizationAD(model).dim == 3


def test_four_dimensions_is_refused():
    with pytest.raises(NotImplementedError, match="4"):
        WENOUpwindDiscretizationAD(None, dim=4)


def test_the_one_dimensional_path_is_untouched():
    """The 1-D scheme is validated against a MATLAB dump; adding the
    multi-dimensional one must not have moved it.

    Face 0's upstream cell is the leftmost, whose absent neighbour is filled
    by constant extrapolation - the same first-order boundary degradation the
    triangulated path shows, arriving by a different route.
    """
    model = _model([20, 1, 1], [20.0, 1.0, 1.0])
    weno = WENOUpwindDiscretizationAD(model, dim=1)
    n = np.asarray(model.operators.N)
    x = np.asarray(model.G.cells.centroids)[:, 0]

    got = np.asarray(weno.face_upstream(model, None,
                                        np.ones(n.shape[0], dtype=bool), x))
    internal = np.all(np.asarray(model.G.faces.neighbors) >= 0, axis=1)
    want = np.asarray(model.G.faces.centroids)[internal, 0]
    np.testing.assert_allclose(got[1:], want[1:], atol=1e-12)
    assert got[0] == pytest.approx(x[n[0, 0]])


def test_a_linear_field_is_exact_on_a_rotated_graded_grid():
    """Linear exactness on a grid whose frame is a genuine rotation.

    This is what makes the local coordinate system testable at all. On an
    axis-aligned grid the SVD frame is a signed permutation - equal to its own
    inverse - so transposing it changes nothing and a real defect passes.
    """
    model = _skewed_model()
    weno = WENOUpwindDiscretizationAD(model, dim=2)
    n = np.asarray(model.operators.N)
    flag = np.ones(n.shape[0], dtype=bool)
    centroids = np.asarray(model.G.cells.centroids)
    fcs = _faces(model, 2)

    # An interior cell here is one whose stencil holds no ghost point.
    interior = _no_ghost_support(weno, model)[n[:, 0]]
    assert interior.any()

    for coeffs in ([1.0, 0.0], [0.0, 1.0], [0.7, -1.3]):
        q = centroids[:, :2] @ np.array(coeffs) + 10.0
        got = np.asarray(weno.face_upstream(model, None, flag, q))
        want = fcs @ np.array(coeffs) + 10.0
        np.testing.assert_allclose(got[interior], want[interior],
                                   rtol=0, atol=1e-10)


def test_the_simplex_volume_actually_weights_the_blend():
    """On a Cartesian grid every simplex has the same volume, so the weight is
    a constant that cancels in `w = beta/sum(beta)` and dropping it changes
    nothing. Graded cells give a ~2x spread, which is where it bites."""
    model = _skewed_model()
    weno = WENOUpwindDiscretizationAD(model, dim=2)
    volumes = np.concatenate(weno.interp_setup.linear_weights)
    assert volumes.max()/volumes.min() > 1.5, "grid is not skewed enough to test"

    n = np.asarray(model.operators.N)
    flag = np.ones(n.shape[0], dtype=bool)
    rng = np.random.default_rng(1)
    q = 10.0 + rng.random(model.G.cells.num)
    with_volume = np.asarray(weno.face_upstream(model, None, flag, q))

    weno.interp_setup.linear_weights = [np.ones_like(v) for v
                                        in weno.interp_setup.linear_weights]
    without = np.asarray(weno.face_upstream(model, None, flag, q))
    assert not np.allclose(with_volume, without), (
        "the simplex volume does not affect the blend")


def test_every_simplex_in_a_cells_support_touches_that_cell():
    """Pins the invariant, and cannot pin the filter that enforces it.

    Measured: across graded, sheared and rotated grids in 2-D and 3-D, with
    and without ghost points, Qhull never produced a simplex excluding the
    centre - the cell centroid is interior to its own face neighbours' hull.
    So `keep = any(t == 0)` is inert for this stencil and removing it changes
    nothing; the invariant is still worth stating, and `get_triangulation`
    records why the filter stays.
    """
    model = _skewed_model()
    weno = WENOUpwindDiscretizationAD(model, dim=2, include_boundary=False)
    s = weno.interp_setup

    for cell in range(model.G.cells.num):
        for tri in s.cell_support[cell]:
            assert cell in s.tri_cells[s.C[tri, :]], (
                f"simplex {tri} in cell {cell}'s support does not touch it")


def test_the_local_frame_orientation_is_used_consistently():
    """The gradient basis is built in the projected frame and `dx` is rotated
    into it. Using the transpose in one of the two places is an inconsistency
    that a linear field detects immediately - but only where the frame is
    genuinely non-symmetric.

    In 2-D the SVD frequently returns a *reflection*, which equals its own
    transpose, so the defect is invisible there. A sheared 3-D grid gives
    non-symmetric frames in 60 of 64 cells.
    """
    model = _skewed_3d_model()
    weno = WENOUpwindDiscretizationAD(model, dim=3)
    M = weno.interp_setup.scaling.mapping
    asymmetry = np.abs(M - np.transpose(M, (0, 2, 1))).max(axis=(1, 2))
    assert (asymmetry > 0.1).sum() > 0.5*asymmetry.size, (
        "frames are near-symmetric; this grid cannot detect a transpose")

    n = np.asarray(model.operators.N)
    flag = np.ones(n.shape[0], dtype=bool)
    centroids = np.asarray(model.G.cells.centroids)
    interior = _no_ghost_support(weno, model)[n[:, 0]]
    assert interior.any()

    coeffs = np.array([0.7, -1.3, 0.4])
    q = centroids[:, :3] @ coeffs + 10.0
    got = np.asarray(weno.face_upstream(model, None, flag, q))
    want = _faces(model, 3) @ coeffs + 10.0
    np.testing.assert_allclose(got[interior], want[interior],
                               rtol=0, atol=1e-9)


def _no_ghost_support(weno, model):
    """Cells whose stencil is made only of real cells."""
    s = weno.interp_setup
    nc = int(model.G.cells.num)
    ok = np.zeros(nc, dtype=bool)
    for c in range(nc):
        pts = s.C[s.cell_support[c], :]
        ok[c] = pts.size > 0 and np.all(pts < nc)
    return ok


def test_a_subset_discretization_keeps_the_support_cells():
    """Dropping a reconstruction's neighbours would change the answer on the
    kept cells, not merely the cost."""
    model = _model([5, 5, 1], [5.0, 5.0, 1.0])
    weno = WENOUpwindDiscretizationAD(model, dim=2)
    subset = weno.get_subset_discretization(model, np.array([12]))

    assert len(subset.interp_setup.cell_support) == 1
    assert subset.interp_setup.cell_support_count[0] > 0
    kept = subset.interp_setup.tri_basis_matrices[0].shape[1]
    assert kept > 1, "the support cells were dropped along with the rest"
