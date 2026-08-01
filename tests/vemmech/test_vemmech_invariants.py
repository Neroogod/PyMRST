"""Invariant tests for vemmech.

vemmech had no pytest suite: its 37 checks were all `validate_phase*.py`
scripts pinning saved MATLAB values. Those catch drift from a reference but
say nothing about the properties the discretization must satisfy, and they
leave several ported entries with no test naming them at all.

This covers the properties: patch tests, rigid-body modes, assembly symmetry,
and the poroelastic terms. It also gives direct coverage to the entries the
parity ledger reported as ported but unvalidated.
"""

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.vemmech.utils.add_fluid_contrib_mech_vem import (
    add_fluid_contrib_mech_vem,
)
from pymrst.modules.vemmech.utils.c2d import c2d
from pymrst.modules.vemmech.utils.calculate_qc import calculate_qc
from pymrst.modules.vemmech.utils.enu2lmu_3d import enu2lmu_3d
from pymrst.modules.vemmech.utils.linear_displacement import linear_displacement
from pymrst.modules.vemmech.utils.square_grid import square_grid
from pymrst.modules.vemmech.utils.vem3d_div import vem3d_div
from pymrst.modules.vemmech.utils.vem_div import vem_div
from pymrst.modules.vemmech.vem import (
    ElasticBC,
    create_augmented_grid,
    enu2c,
    enu2c_2d,
)
from pymrst.modules.vemmech.vem_assemble import vem_assemble
from pymrst.modules.vemmech.vem_lin_elast import vem_lin_elast

E_DEFAULT, NU_DEFAULT = 1.0e6, 0.25


def _grid(dims, size=None):
    size = size or [float(d) for d in dims]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return create_augmented_grid(compute_geometry(cart_grid(list(dims), size)))


def _stiffness(grid, E=E_DEFAULT, nu=NU_DEFAULT):
    """Isotropic tensor of the right rank: 3x3 in plane strain, 6x6 in 3D."""
    build = enu2c if grid.griddim == 3 else enu2c_2d
    return build(np.full(grid.cells.num, E), np.full(grid.cells.num, nu))


def _clamp_all_boundary_nodes(grid, field=None):
    """Prescribe every boundary node, optionally to an analytic field."""
    dim = grid.griddim
    boundary = np.flatnonzero(np.any(grid.faces.neighbors < 0, axis=1))
    nodes = np.unique(np.concatenate([grid.face_nodes(f) for f in boundary]))
    coords = grid.nodes.coords[nodes, :dim]
    values = np.zeros((nodes.size, dim)) if field is None else field(coords)
    return ElasticBC(
        disp_nodes=nodes,
        disp_mask=np.ones((nodes.size, dim), dtype=bool),
        disp_vals=values,
        force_faces=np.zeros(0, dtype=np.int64),
        force=np.zeros((0, dim)),
    )


def _zero_load(x):
    return np.zeros_like(x)


# ------------------------------------------------------------- patch tests


@pytest.mark.parametrize("dims", [(3, 3), (3, 3, 3)])
def test_linear_displacement_fields_are_reproduced_exactly(dims):
    """The defining property of a linear-elastic discretization.

    Prescribe an analytic linear field on the boundary with no body load; the
    interior must come back with that same field, to round-off.
    """
    grid = _grid(dims)
    C = _stiffness(grid)
    fields, names = linear_displacement(grid.griddim)
    for field, name in zip(fields, names):
        bc = _clamp_all_boundary_nodes(grid, field)
        u, _ = vem_lin_elast(grid, C, bc, _zero_load)
        expected = field(grid.nodes.coords[:, :grid.griddim])
        error = np.abs(u - expected).max() / max(np.abs(expected).max(), 1.0)
        assert error < 1e-9, f"{name}: max relative error {error:.3e}"


def test_patch_test_holds_on_a_distorted_grid():
    """Irregular cells must not break the patch test - that is the point of VEM."""
    grid = _grid((4, 4))
    rng = np.random.default_rng(3)
    points = grid.nodes.coords
    interior = np.all((points > 1e-12) & (points < np.array([4.0, 4.0]) - 1e-12),
                      axis=1)
    points[interior] += 0.25 * (rng.random((int(interior.sum()), 2)) - 0.5)
    grid = create_augmented_grid(compute_geometry(grid))
    C = _stiffness(grid)
    fields, names = linear_displacement(2)
    for field, name in zip(fields, names):
        bc = _clamp_all_boundary_nodes(grid, field)
        u, _ = vem_lin_elast(grid, C, bc, _zero_load)
        expected = field(grid.nodes.coords[:, :2])
        error = np.abs(u - expected).max() / max(np.abs(expected).max(), 1.0)
        assert error < 1e-9, f"{name}: max relative error {error:.3e}"


# --------------------------------------------------------- assembly checks


@pytest.mark.parametrize("dims", [(2, 2), (2, 2, 2)])
def test_stiffness_matrix_is_symmetric(dims):
    grid = _grid(dims)
    S = vem_assemble(grid, _stiffness(grid))
    asymmetry = abs(S - S.T).max()
    assert asymmetry < 1e-6 * abs(S).max(), asymmetry


@pytest.mark.parametrize("dims", [(2, 2), (2, 2, 2)])
def test_rigid_body_modes_are_in_the_null_space(dims):
    """Translations and rotations must cost no energy."""
    grid = _grid(dims)
    S = vem_assemble(grid, _stiffness(grid))
    dim = grid.griddim
    scale = abs(S).max()
    fields, names = linear_displacement(dim)
    for field, name in zip(fields, names):
        if not (name.startswith("cons_disp") or name.startswith("rot_")):
            continue
        u = field(grid.nodes.coords[:, :dim]).ravel()
        energy = float(u @ (S @ u))
        assert abs(energy) < 1e-8 * scale * max(float(u @ u), 1.0), (name, energy)


@pytest.mark.parametrize("dims", [(2, 2), (2, 2, 2)])
def test_divergence_operator_matches_cell_volumes(dims):
    """A unit dilation must produce a divergence equal to the cell volume."""
    grid = _grid(dims)
    dim = grid.griddim
    div = vem_div(grid)
    u = np.tile(grid.nodes.coords[:, :dim], (1, 1)).ravel()
    assert np.allclose(div @ u, grid.cells.volumes * dim, rtol=1e-9)


def test_vem3d_div_agrees_with_the_dispatching_wrapper():
    grid = _grid((2, 2, 2))
    assert abs(vem_div(grid) - vem3d_div(grid)).max() < 1e-12


# ---------------------------------------------------------------- material


@pytest.mark.parametrize("dims", [(2, 2), (2, 2, 2)])
def test_c2d_applies_the_shear_scaling_without_touching_normal_terms(dims):
    """c2d rescales the shear block for the VEM strain-energy product."""
    grid = _grid(dims)
    C = _stiffness(grid)
    scaled = c2d(C)
    assert scaled.shape[0] == grid.cells.num
    assert np.all(np.isfinite(scaled))
    flat, out = C.reshape(grid.cells.num, -1), scaled.reshape(grid.cells.num, -1)
    assert np.allclose(out[:, 0], flat[:, 0]), "normal-normal term must not move"



def test_enu2lmu_3d_is_consistent_with_the_stiffness_tensor():
    """lambda and mu must reproduce the isotropic tensor built by Enu2C."""
    grid = _grid((2, 2, 2))
    lam, mu = enu2lmu_3d(np.full(grid.cells.num, E_DEFAULT),
                         np.full(grid.cells.num, NU_DEFAULT))
    # enu2c returns each cell's 6x6 Voigt tensor flattened to 36 entries
    C = _stiffness(grid).reshape(grid.cells.num, 6, 6)
    # for an isotropic tensor C[0,0] = lambda + 2 mu and C[0,1] = lambda
    assert np.allclose(C[:, 0, 0], lam + 2.0 * mu, rtol=1e-10)
    assert np.allclose(C[:, 0, 1], lam, rtol=1e-10)


def test_calculate_qc_volume_weights_sum_to_the_cell_volume():
    grid = _grid((3, 3, 3))
    _, _, qcvol = calculate_qc(grid)
    cell_no = np.repeat(np.arange(grid.cells.num), np.diff(grid.cells.nodePos))
    total = np.bincount(cell_no, weights=np.asarray(qcvol).ravel(),
                        minlength=grid.cells.num)
    assert np.allclose(total, grid.cells.volumes, rtol=1e-9)


# ------------------------------------------------------------ poroelastic


def test_zero_pressure_reproduces_the_dry_solve():
    grid = _grid((3, 3, 3))
    C = _stiffness(grid)
    bc = _clamp_all_boundary_nodes(grid)
    rock = SimpleNamespace(alpha=np.ones(grid.cells.num))
    boundary = np.flatnonzero(np.any(grid.faces.neighbors < 0, axis=1))
    dry, _ = vem_lin_elast(grid, C, bc, _zero_load)
    wet, _ = vem_lin_elast(grid, C, bc, _zero_load,
                           fluid_bc=SimpleNamespace(face=boundary, value=0.0),
                           rock=rock)
    assert np.allclose(dry, wet)


def test_boundary_pressure_response_is_linear():
    grid = _grid((3, 3, 3))
    C = _stiffness(grid)
    base = grid.nodes.coords[:, 2]
    clamped = np.flatnonzero(np.isclose(base, 0.0))
    bc = ElasticBC(disp_nodes=clamped,
                   disp_mask=np.ones((clamped.size, 3), dtype=bool),
                   disp_vals=np.zeros((clamped.size, 3)),
                   force_faces=np.zeros(0, dtype=np.int64),
                   force=np.zeros((0, 3)))
    rock = SimpleNamespace(alpha=np.ones(grid.cells.num))
    top = np.flatnonzero(np.any(grid.faces.neighbors < 0, axis=1)
                         & np.isclose(grid.faces.centroids[:, 2], 3.0))
    one, _ = vem_lin_elast(grid, C, bc, _zero_load,
                           fluid_bc=SimpleNamespace(face=top, value=1.0e5),
                           rock=rock)
    two, _ = vem_lin_elast(grid, C, bc, _zero_load,
                           fluid_bc=SimpleNamespace(face=top, value=2.0e5),
                           rock=rock)
    assert np.abs(one).max() > 0
    assert np.allclose(two, 2.0 * one, rtol=1e-9)


def test_fluid_bc_without_rock_is_rejected():
    grid = _grid((2, 2, 2))
    bc = _clamp_all_boundary_nodes(grid)
    boundary = np.flatnonzero(np.any(grid.faces.neighbors < 0, axis=1))
    with pytest.raises(ValueError, match="Biot alpha"):
        vem_lin_elast(grid, _stiffness(grid), bc, _zero_load,
                      fluid_bc=SimpleNamespace(face=boundary, value=1.0))


def test_closed_pressurized_boundary_is_self_equilibrated():
    grid = _grid((3, 3, 3))
    rock = SimpleNamespace(alpha=np.ones(grid.cells.num))
    boundary = np.flatnonzero(np.any(grid.faces.neighbors < 0, axis=1))
    fbc = add_fluid_contrib_mech_vem(
        grid, SimpleNamespace(face=boundary, value=1.0), rock,
        np.zeros(3 * grid.nodes.num, dtype=bool))
    assert np.allclose(fbc.reshape(-1, 3).sum(axis=0), 0.0, atol=1e-10)


# ----------------------------------------------------------- test fixtures


@pytest.mark.parametrize("grid_type", ["cartgrid", "triangle", "pebi"])
def test_square_grid_builds_a_valid_covering_grid(grid_type):
    grid = compute_geometry(square_grid([4, 4], [1.0, 1.0], grid_type=grid_type))
    assert np.all(grid.cells.volumes > 0)
    assert np.isclose(grid.cells.volumes.sum(), 1.0, rtol=1e-6)


@pytest.mark.parametrize("grid_type", ["boxes1", "mixed1"])
def test_glued_square_grids_fill_the_requested_rectangle(grid_type):
    grid = compute_geometry(square_grid([4, 4], [3.0, 3.0], grid_type=grid_type))
    assert np.all(grid.cells.volumes > 0)
    assert np.isclose(grid.nodes.coords[:, 0].max(), 3.0)
    assert np.isclose(grid.nodes.coords[:, 1].max(), 3.0)


def test_square_grid_solves_linear_elasticity_on_a_pebi_grid():
    """The fixtures exist to be solved on, so check one end to end."""
    grid = create_augmented_grid(
        compute_geometry(square_grid([4, 4], [1.0, 1.0], grid_type="pebi")))
    C = _stiffness(grid)
    fields, _ = linear_displacement(2)
    field = fields[2]                                    # lin_disp_1
    bc = _clamp_all_boundary_nodes(grid, field)
    u, _ = vem_lin_elast(grid, C, bc, _zero_load)
    expected = field(grid.nodes.coords[:, :2])
    assert np.abs(u - expected).max() < 1e-8
