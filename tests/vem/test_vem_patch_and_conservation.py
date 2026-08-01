"""Patch tests and conservation, as assertions rather than printed numbers.

Two properties a VEM must have, and one it must have *after* postprocessing:

- **exactness**: k=1 reproduces any linear field the PDE admits, k=2 any
  quadratic, in 2-D and 3-D. If it does not, the local projection is wrong and
  nothing downstream is trustworthy.
- **conservation**: the divergence of the flux equals the source, cell by
  cell. Without it a transport step invents saturation.

A patch test only tests what the PDE admits, and that trips people up: `x^2`
is not harmonic, so with `K = I` a k=2 solve will *not* reproduce it - it
needs a source. `x^2 - y^2` is. Every field below is checked against the
operator it is solved with.
"""

import numpy as np
import pytest

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.vem.compute_virtual_ip import compute_virtual_ip
from pymrst.modules.vem.incomp_vem import incomp_vem
from pymrst.modules.vem.utils.add_bc_vem import add_bc_vem
from pymrst.modules.vem.utils.compute_vem_geometry import compute_vem_geometry
from pymrst.modules.vem.utils.conserve_flux import conserve_flux
from pymrst.params.rock import make_rock
from pymrst.params.wells_and_bc.add_bc import add_bc
from pymrst.params.wells_and_bc.add_source import add_source
from pymrst.utils.rldecode import rldecode


def twisted(nx=5, amp=0.3):
    """A non-orthogonal grid: exactness here is not exactness on a box."""
    g = cart_grid([nx, nx], [float(nx), float(nx)])
    coords = np.asarray(g.nodes.coords, dtype=float)
    x, y, L = coords[:, 0].copy(), coords[:, 1].copy(), float(nx)
    coords[:, 0] += amp * np.sin(np.pi * x / L) * np.sin(np.pi * y / L)
    coords[:, 1] += 0.6 * amp * np.sin(2 * np.pi * x / L) * np.sin(np.pi * y / L)
    g.nodes.coords = coords
    return compute_vem_geometry(compute_geometry(g))


def boundary_faces_of(g):
    return np.flatnonzero(np.any(np.asarray(g.faces.neighbors) < 0, axis=1))


def divergence(g, flux):
    face_pos = np.asarray(g.cells.facePos, dtype=np.int64)
    cell_faces = np.asarray(g.cells.faces, dtype=np.int64)
    cell_faces = cell_faces[:, 0] if cell_faces.ndim > 1 else cell_faces
    owner = rldecode(np.arange(g.cells.num), np.diff(face_pos))
    neighbours = np.asarray(g.faces.neighbors)
    signs = np.where(neighbours[cell_faces, 0] == owner, 1.0, -1.0)
    out = np.zeros(g.cells.num)
    np.add.at(out, owner, flux[cell_faces] * signs)
    return out


def solve_with(g, order, field):
    S = compute_virtual_ip(g, make_rock(g, 1.0, 1.0), order)
    bc = add_bc_vem(None, boundary_faces_of(g), "pressure", field)
    return incomp_vem(g, S, bc=bc, reconstruct_flux=False)


# --------------------------------------------------------------- exactness

def test_first_order_reproduces_a_linear_field_in_two_dimensions():
    g = twisted()
    field = lambda P: 1 + 2 * P[:, 0] - 3 * P[:, 1]
    state = solve_with(g, 1, field)
    error = np.abs(state["node_pressure"]
                   - field(np.asarray(g.nodes.coords)[:, :2])).max()
    assert error < 1e-12, error


def test_first_order_reproduces_a_linear_field_in_three_dimensions():
    g = compute_vem_geometry(compute_geometry(
        cart_grid([3, 3, 3], [3.0, 3.0, 3.0])))
    field = lambda P: 2 - P[:, 0] + 0.5 * P[:, 1] + P[:, 2]
    state = solve_with(g, 1, field)
    error = np.abs(state["node_pressure"]
                   - field(np.asarray(g.nodes.coords))).max()
    assert error < 1e-12, error


def test_second_order_reproduces_a_harmonic_quadratic_in_two_dimensions():
    g = twisted()
    field = lambda P: 1 + P[:, 0] - 2 * P[:, 1] + P[:, 0] ** 2 - P[:, 1] ** 2
    state = solve_with(g, 2, field)
    error = np.abs(state["node_pressure"]
                   - field(np.asarray(g.nodes.coords)[:, :2])).max()
    assert error < 1e-9, error


def test_second_order_reproduces_a_harmonic_quadratic_in_three_dimensions():
    g = compute_vem_geometry(compute_geometry(
        cart_grid([2, 2, 2], [2.0, 2.0, 2.0])))
    field = lambda P: 1 + P[:, 0] - P[:, 2] + P[:, 0] ** 2 - P[:, 1] ** 2
    state = solve_with(g, 2, field)
    error = np.abs(state["node_pressure"]
                   - field(np.asarray(g.nodes.coords))).max()
    assert error < 1e-9, error


def test_first_order_does_not_reproduce_a_quadratic():
    """The patch tests above would pass for a method that was accidentally
    higher order; this shows k=1 really is first order."""
    g = twisted()
    field = lambda P: P[:, 0] ** 2 - P[:, 1] ** 2
    state = solve_with(g, 1, field)
    error = np.abs(state["node_pressure"]
                   - field(np.asarray(g.nodes.coords)[:, :2])).max()
    assert error > 1e-6, error


def test_a_non_harmonic_quadratic_is_not_reproduced_without_a_source():
    """`x^2` has Laplacian 2, so it does not solve the homogeneous equation.

    This is the mistake a patch test invites: the field must be one the PDE
    admits, not merely one the space contains.
    """
    g = twisted()
    state = solve_with(g, 2, lambda P: P[:, 0] ** 2)
    exact = (np.asarray(g.nodes.coords)[:, 0]) ** 2
    assert np.abs(state["node_pressure"] - exact).max() > 1e-6


# ------------------------------------------------------------ conservation

def test_the_default_flux_reconstruction_is_already_conservative():
    g = twisted()
    S = compute_virtual_ip(g, make_rock(g, 1.0, 1.0), 1)
    centroids = np.asarray(g.faces.centroids)
    outer = np.any(np.asarray(g.faces.neighbors) < 0, axis=1)
    left = np.flatnonzero((np.abs(centroids[:, 0]
                                  - centroids[:, 0].min()) < 1e-9) & outer)
    right = np.flatnonzero((np.abs(centroids[:, 0]
                                   - centroids[:, 0].max()) < 1e-9) & outer)
    bc = add_bc(None, left, "pressure", np.ones(left.size))
    bc = add_bc(bc, right, "pressure", np.zeros(right.size))
    state = incomp_vem(g, S, bc=bc)
    assert np.abs(divergence(g, state["flux"])).max() < 1e-12


def test_the_tpfa_flux_is_not_conservative_until_corrected():
    g = twisted()
    rock = make_rock(g, 1.0, 1.0)
    S = compute_virtual_ip(g, rock, 1, trans="tpfa")
    centroids = np.asarray(g.faces.centroids)
    outer = np.any(np.asarray(g.faces.neighbors) < 0, axis=1)
    left = np.flatnonzero((np.abs(centroids[:, 0]
                                  - centroids[:, 0].min()) < 1e-9) & outer)
    right = np.flatnonzero((np.abs(centroids[:, 0]
                                   - centroids[:, 0].max()) < 1e-9) & outer)
    bc = add_bc(None, left, "pressure", np.ones(left.size))
    bc = add_bc(bc, right, "pressure", np.zeros(right.size))
    state = incomp_vem(g, S, bc=bc)

    before = np.abs(divergence(g, state["flux"])).max()
    fixed, _ = conserve_flux(state, g, rock, bc=bc)
    after = np.abs(divergence(g, fixed["flux"])).max()
    assert before > 1e-4, before
    assert after < 1e-12, after


def test_a_source_is_matched_cell_by_cell_after_correction():
    g = twisted()
    rock = make_rock(g, 1.0, 1.0)
    S = compute_virtual_ip(g, rock, 1, trans="tpfa")
    centroids = np.asarray(g.faces.centroids)
    outer = np.any(np.asarray(g.faces.neighbors) < 0, axis=1)
    right = np.flatnonzero((np.abs(centroids[:, 0]
                                   - centroids[:, 0].max()) < 1e-9) & outer)
    bc = add_bc(None, right, "pressure", np.zeros(right.size))
    src = add_source(None, [0], [1.0])
    state = incomp_vem(g, S, bc=bc, src=src)

    rhs = np.zeros(g.cells.num)
    rhs[0] = 1.0
    fixed, _ = conserve_flux(state, g, rock, bc=bc, src=src)
    assert np.abs(rhs - divergence(g, fixed["flux"])).max() < 1e-10


@pytest.mark.parametrize("order", [1, 2])
def test_what_enters_leaves_a_closed_domain(order):
    """Source and sink in a sealed box: the totals cancel exactly.

    The domain has to be *told* it is sealed - see the next test.
    """
    g = twisted()
    S = compute_virtual_ip(g, make_rock(g, 1.0, 1.0), order)
    src = add_source(None, [0, g.cells.num - 1], [1.0, -1.0])
    outer = boundary_faces_of(g)
    bc = add_bc(None, outer, "flux", np.zeros(outer.size))
    state = incomp_vem(g, S, src=src, bc=bc)
    assert abs(divergence(g, state["flux"]).sum()) < 1e-9
    assert np.abs(state["flux"][outer]).max() < 1e-12


@pytest.mark.parametrize("order", [1, 2])
def test_no_boundary_condition_means_an_open_boundary(order):
    """`bc=None` is not the same as a sealed domain, and the difference is
    easy to miss.

    Without a boundary condition the reconstruction leaves each boundary face
    at whatever flux it computed - here about 0.23 against a source of 1.0 -
    so the interior totals do not cancel. A user solving a source-and-sink
    problem without a `bc` gets a leaky domain and no warning. This pins the
    behaviour so it is a decision rather than a surprise.
    """
    g = twisted()
    S = compute_virtual_ip(g, make_rock(g, 1.0, 1.0), order)
    src = add_source(None, [0, g.cells.num - 1], [1.0, -1.0])
    state = incomp_vem(g, S, src=src)
    outer = boundary_faces_of(g)
    assert np.abs(state["flux"][outer]).max() > 1e-3
    assert abs(divergence(g, state["flux"]).sum()) > 1e-6
