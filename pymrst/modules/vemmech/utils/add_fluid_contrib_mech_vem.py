"""Fluid pressure load on the mechanics boundary (MRST `addFluidContribMechVEM`).

This is the *boundary* half of the poroelastic coupling. The interior half -
the volumetric term `div' * p` - is applied inside `vem_lin_elast`. This
module supplies what a pressure boundary condition does to the mechanics: a
traction `alpha * p * n` spread over the nodes of each boundary face, pointing
out of the cell that owns the face.

Without it a poroelastic model feels fluid pressure only in the interior and
nothing on its pressurized boundary, which is not a small correction - it is
the load that drives compaction problems.
"""

from typing import Any, Optional

import numpy as np

from pymrst.gridprocessing.grid_structure import Grid
from pymrst.modules.vemmech.utils.calculate_qc import calculate_qc
from pymrst.utils.mcolon import mcolon


def add_fluid_contrib_mech_vem(
    g: Grid,
    bc: Optional[Any],
    rock: Any,
    isdirdofs: np.ndarray,
) -> np.ndarray:
    """Return the nodal force vector from a fluid pressure boundary condition.

    `bc` carries `face` and `value`, as MRST's `addBC` produces. `rock` must
    provide `alpha`, the Biot coefficient per cell. `isdirdofs` is the boolean
    mask of prescribed displacement degrees of freedom; the returned vector is
    restricted to the free ones, so it can be added straight to the reduced
    right-hand side.

    Passing `bc=None` (or an empty face list) returns the correctly sized zero
    vector, which is what makes the term safe to add unconditionally.
    """
    isdirdofs = np.asarray(isdirdofs, dtype=bool)
    dim = g.griddim
    ndof = g.nodes.num * dim

    faces = _faces_of(bc)
    if faces.size == 0:
        return np.zeros(ndof)[~isdirdofs]

    node_pos = np.asarray(g.faces.nodePos, dtype=np.int64)
    neighbors = np.asarray(g.faces.neighbors, dtype=np.int64)

    if not np.all(np.sum(neighbors[faces] < 0, axis=1) == 1):
        raise ValueError(
            "fluid pressure boundary conditions must name boundary faces"
        )

    per_face = node_pos[faces + 1] - node_pos[faces]
    inodes = mcolon(node_pos[faces], node_pos[faces + 1])
    nodes = np.asarray(g.faces.nodes, dtype=np.int64)[inodes]
    local_face = np.repeat(np.arange(faces.size, dtype=np.int64), per_face)

    # outward normal with respect to the cell that owns the boundary face
    owner = neighbors[faces].max(axis=1)
    orientation = np.where(neighbors[faces, 0] == owner, 1.0, -1.0)
    normals = np.asarray(g.faces.normals, dtype=float)[faces] * orientation[:, None]

    if dim == 2:
        # an edge splits its normal evenly between its two nodes
        weights = (normals / 2.0)[local_face]
    else:
        # in 3D the face normal is distributed by sub-face area, so a node
        # facing a larger part of the face takes a larger share of the load
        _, qf, _ = calculate_qc(g)
        unit_normals = normals / np.asarray(g.faces.areas, dtype=float)[faces, None]
        weights = unit_normals[local_face] * np.asarray(qf, dtype=float)[inodes, None]

    alpha = np.asarray(rock.alpha if hasattr(rock, "alpha") else rock["alpha"],
                       dtype=float)
    face_alpha = alpha[owner][local_face]
    pressure = np.repeat(np.asarray(_values_of(bc, faces.size), dtype=float),
                         per_face)
    force = weights * (face_alpha * pressure)[:, None]

    dofs = mcolon(dim * nodes, dim * nodes + dim)
    fbc = np.bincount(dofs, weights=force.ravel(), minlength=ndof)
    return fbc[~isdirdofs]


def _faces_of(bc) -> np.ndarray:
    if bc is None:
        return np.zeros(0, dtype=np.int64)
    faces = bc.face if hasattr(bc, "face") else bc["face"]
    if faces is None:
        return np.zeros(0, dtype=np.int64)
    return np.atleast_1d(np.asarray(faces, dtype=np.int64)).ravel()


def _values_of(bc, count: int) -> np.ndarray:
    value = bc.value if hasattr(bc, "value") else bc["value"]
    value = np.atleast_1d(np.asarray(value, dtype=float)).ravel()
    if value.size == 1:
        return np.full(count, value[0])
    if value.size != count:
        raise ValueError(
            f"boundary pressure has {value.size} values for {count} faces"
        )
    return value


addFluidContribMechVEM = add_fluid_contrib_mech_vem


if __name__ == "__main__":
    from types import SimpleNamespace

    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.vemmech.vem import create_augmented_grid

    def _grid(dims, size):
        return create_augmented_grid(compute_geometry(cart_grid(dims, size)))

    def _rock(g, alpha=1.0):
        return SimpleNamespace(alpha=np.full(g.cells.num, float(alpha)))

    def test_no_boundary_condition_gives_a_zero_vector() -> None:
        g = _grid([2, 2, 2], [1.0, 1.0, 1.0])
        isdir = np.zeros(3 * g.nodes.num, dtype=bool)
        fbc = add_fluid_contrib_mech_vem(g, None, _rock(g), isdir)
        assert fbc.shape == (3 * g.nodes.num,)
        assert np.all(fbc == 0.0)
        print("ok test_no_boundary_condition_gives_a_zero_vector")

    def test_uniform_pressure_on_a_closed_boundary_cancels() -> None:
        """Equal pressure on every outer face is pure compression: the total
        force on the body must vanish."""
        g = _grid([3, 3, 3], [1.0, 1.0, 1.0])
        boundary = np.flatnonzero(np.any(g.faces.neighbors < 0, axis=1))
        bc = SimpleNamespace(face=boundary, value=1.0)
        isdir = np.zeros(3 * g.nodes.num, dtype=bool)
        fbc = add_fluid_contrib_mech_vem(g, bc, _rock(g), isdir)
        total = fbc.reshape(-1, 3).sum(axis=0)
        assert np.allclose(total, 0.0, atol=1e-12), total
        print("ok test_uniform_pressure_on_a_closed_boundary_cancels")

    def test_one_face_pushes_along_its_outward_normal() -> None:
        g = _grid([2, 2, 2], [1.0, 1.0, 1.0])
        top = np.flatnonzero(
            (np.any(g.faces.neighbors < 0, axis=1))
            & np.isclose(g.faces.centroids[:, 2], 1.0)
        )
        bc = SimpleNamespace(face=top, value=2.0)
        isdir = np.zeros(3 * g.nodes.num, dtype=bool)
        fbc = add_fluid_contrib_mech_vem(g, bc, _rock(g), isdir)
        total = fbc.reshape(-1, 3).sum(axis=0)
        # outward normal on the top face is +z, and the loaded area is 1
        assert np.allclose(total[:2], 0.0, atol=1e-12), total
        assert np.isclose(total[2], 2.0), total
        print("ok test_one_face_pushes_along_its_outward_normal")

    def test_force_scales_with_pressure_and_alpha() -> None:
        g = _grid([2, 2, 2], [1.0, 1.0, 1.0])
        top = np.flatnonzero(
            (np.any(g.faces.neighbors < 0, axis=1))
            & np.isclose(g.faces.centroids[:, 2], 1.0)
        )
        isdir = np.zeros(3 * g.nodes.num, dtype=bool)
        base = add_fluid_contrib_mech_vem(
            g, SimpleNamespace(face=top, value=1.0), _rock(g, 1.0), isdir)
        doubled = add_fluid_contrib_mech_vem(
            g, SimpleNamespace(face=top, value=2.0), _rock(g, 1.0), isdir)
        alpha_half = add_fluid_contrib_mech_vem(
            g, SimpleNamespace(face=top, value=2.0), _rock(g, 0.5), isdir)
        assert np.allclose(doubled, 2.0 * base)
        assert np.allclose(alpha_half, base)
        print("ok test_force_scales_with_pressure_and_alpha")

    def test_2d_edge_splits_its_normal_evenly() -> None:
        g = _grid([2, 2], [1.0, 1.0])
        right = np.flatnonzero(
            (np.any(g.faces.neighbors < 0, axis=1))
            & np.isclose(g.faces.centroids[:, 0], 1.0)
        )
        bc = SimpleNamespace(face=right, value=1.0)
        isdir = np.zeros(2 * g.nodes.num, dtype=bool)
        fbc = add_fluid_contrib_mech_vem(g, bc, _rock(g), isdir)
        total = fbc.reshape(-1, 2).sum(axis=0)
        assert np.isclose(total[0], 1.0), total     # unit length, outward +x
        assert np.allclose(total[1], 0.0, atol=1e-12), total
        print("ok test_2d_edge_splits_its_normal_evenly")

    def test_dirichlet_dofs_are_removed() -> None:
        g = _grid([2, 2, 2], [1.0, 1.0, 1.0])
        boundary = np.flatnonzero(np.any(g.faces.neighbors < 0, axis=1))
        bc = SimpleNamespace(face=boundary, value=1.0)
        isdir = np.zeros(3 * g.nodes.num, dtype=bool)
        isdir[:9] = True
        fbc = add_fluid_contrib_mech_vem(g, bc, _rock(g), isdir)
        assert fbc.shape == (3 * g.nodes.num - 9,)
        print("ok test_dirichlet_dofs_are_removed")

    def test_interior_faces_are_rejected() -> None:
        g = _grid([3, 3, 3], [1.0, 1.0, 1.0])
        interior = np.flatnonzero(np.all(g.faces.neighbors >= 0, axis=1))[:2]
        isdir = np.zeros(3 * g.nodes.num, dtype=bool)
        try:
            add_fluid_contrib_mech_vem(
                g, SimpleNamespace(face=interior, value=1.0), _rock(g), isdir)
        except ValueError:
            print("ok test_interior_faces_are_rejected")
            return
        raise AssertionError("interior faces should be rejected")

    test_no_boundary_condition_gives_a_zero_vector()
    test_uniform_pressure_on_a_closed_boundary_cancels()
    test_one_face_pushes_along_its_outward_normal()
    test_force_scales_with_pressure_and_alpha()
    test_2d_edge_splits_its_normal_evenly()
    test_dirichlet_dofs_are_removed()
    test_interior_faces_are_rejected()
    print("\nAll tests passed!")
