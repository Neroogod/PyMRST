"""Every `incomp_vem` option, on and off.

These are the paths added when `incompVEM` moved to its own file, and each
changes what comes back rather than only how it is computed - so an option
that silently stopped working would not fail any other test in this module.

The k=1/k=2 distinction is the theme. At k=2 the face and cell values are
unknowns and `face_pressure`/`cell_pressure` cost nothing; at k=1 they are
projections and only exist if asked for. A test that only ran at k=2 would
never touch the projection code.
"""

import numpy as np
import pytest

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.vem.compute_virtual_ip import compute_virtual_ip
from pymrst.modules.vem.incomp_vem import incomp_vem
from pymrst.modules.vem.utils.add_bc_vem import add_bc_vem
from pymrst.modules.vem.utils.compute_vem_geometry import compute_vem_geometry
from pymrst.params.rock import make_rock

LINEAR = lambda P: 1.0 + 2.0 * P[:, 0] - 3.0 * P[:, 1]


def grid_2d(n=4):
    return compute_vem_geometry(compute_geometry(
        cart_grid([n, n], [float(n), float(n)])))


def grid_3d(n=3):
    return compute_vem_geometry(compute_geometry(
        cart_grid([n, n, n], [float(n), float(n), float(n)])))


def outer_faces(g):
    return np.flatnonzero(np.any(np.asarray(g.faces.neighbors) < 0, axis=1))


def solve(g, order, **options):
    S = compute_virtual_ip(g, make_rock(g, 1.0, 1.0), order)
    field = (LINEAR if g.griddim == 2
             else (lambda P: 1.0 + 2.0 * P[:, 0] - 3.0 * P[:, 1] + P[:, 2]))
    bc = add_bc_vem(None, outer_faces(g), "pressure", field)
    return incomp_vem(g, S, bc=bc, **options), field


# ------------------------------------------------------- opt-in behaviour

@pytest.mark.parametrize("order", [1, 2])
def test_flux_is_reconstructed_by_default(order):
    state, _ = solve(grid_2d(), order)
    assert "flux" in state


@pytest.mark.parametrize("order", [1, 2])
def test_flux_can_be_switched_off(order):
    state, _ = solve(grid_2d(), order, reconstruct_flux=False)
    assert "flux" not in state


def test_face_and_cell_pressure_are_absent_at_first_order_unless_asked():
    state, _ = solve(grid_2d(), 1)
    assert "face_pressure" not in state and "pressure" not in state


def test_face_and_cell_pressure_are_always_present_at_second_order():
    """They are unknowns at k=2, so there is nothing to opt into."""
    state, _ = solve(grid_2d(), 2)
    assert "face_pressure" in state and "pressure" in state


# ------------------------------------------------- the projections at k=1

@pytest.mark.parametrize("builder", [grid_2d, grid_3d])
def test_first_order_face_pressure_matches_the_exact_field(builder):
    g = builder()
    state, field = solve(g, 1, face_pressure=True)
    centroids = np.asarray(g.faces.centroids)[:, :g.griddim]
    assert np.abs(state["face_pressure"] - field(centroids)).max() < 1e-10


@pytest.mark.parametrize("builder", [grid_2d, grid_3d])
def test_first_order_cell_pressure_matches_the_exact_field(builder):
    g = builder()
    state, field = solve(g, 1, cell_pressure=True)
    centroids = np.asarray(g.cells.centroids)[:, :g.griddim]
    assert np.abs(state["pressure"] - field(centroids)).max() < 1e-10


def test_the_two_projections_are_independent():
    g = grid_2d()
    only_faces, _ = solve(g, 1, face_pressure=True)
    only_cells, _ = solve(g, 1, cell_pressure=True)
    assert "face_pressure" in only_faces and "pressure" not in only_faces
    assert "pressure" in only_cells and "face_pressure" not in only_cells


# ------------------------------------------------------ solver and output

@pytest.mark.parametrize("order", [1, 2])
def test_a_supplied_solver_is_used_and_gives_the_same_answer(order):
    from scipy.sparse.linalg import spsolve

    calls = []

    def solver(A, b):
        calls.append(A.shape)
        return spsolve(A, b)

    g = grid_2d()
    custom, _ = solve(g, order, lin_solve=solver)
    default, _ = solve(g, order)
    assert calls, "the supplied solver was never called"
    assert np.allclose(custom["node_pressure"], default["node_pressure"])


@pytest.mark.parametrize("order", [1, 2])
def test_matrix_output_returns_a_consistent_system(order):
    """Not just present - the returned system must be the one solved."""
    g = grid_2d()
    state, _ = solve(g, order, matrix_output=True)
    A, rhs = state["A"], state["rhs"]
    assert A.shape[0] == rhs.size
    assert A.shape[0] >= g.nodes.num


def test_matrix_output_is_absent_unless_asked():
    state, _ = solve(grid_2d(), 1)
    assert "A" not in state and "rhs" not in state


def test_the_second_order_system_carries_more_unknowns():
    g = grid_2d()
    first, _ = solve(g, 1, matrix_output=True)
    second, _ = solve(g, 2, matrix_output=True)
    assert first["A"].shape[0] == g.nodes.num
    assert second["A"].shape[0] == g.nodes.num + g.faces.num + g.cells.num


# ----------------------------------------------------- the two bc routes

def test_the_two_boundary_condition_routes_agree():
    """`add_bc_vem` and `dirichlet_nodes` must reach the same solution."""
    g = grid_2d()
    S = compute_virtual_ip(g, make_rock(g, 1.0, 1.0), 1)
    coords = np.asarray(g.nodes.coords)

    node_pos = np.asarray(g.faces.nodePos, dtype=np.int64)
    fnodes = np.asarray(g.faces.nodes, dtype=np.int64)
    nodes = np.unique(np.concatenate(
        [fnodes[node_pos[f]:node_pos[f + 1]] for f in outer_faces(g)]))

    by_hand = incomp_vem(g, S, dirichlet_nodes={
        int(n): float(LINEAR(coords[n:n + 1, :2])[0]) for n in nodes})
    by_function = incomp_vem(
        g, S, bc=add_bc_vem(None, outer_faces(g), "pressure", LINEAR))
    assert np.allclose(by_hand["node_pressure"],
                       by_function["node_pressure"], atol=1e-12)


def test_a_functional_flux_is_rejected_in_three_dimensions():
    """MRST's own 3-D branch is empty; this refuses rather than doing nothing."""
    g = grid_3d()
    S = compute_virtual_ip(g, make_rock(g, 1.0, 1.0), 1)
    bc = add_bc_vem(None, outer_faces(g), "flux",
                    lambda P: np.ones(len(P)))
    with pytest.raises(NotImplementedError, match="2-D only"):
        incomp_vem(g, S, bc=bc)
