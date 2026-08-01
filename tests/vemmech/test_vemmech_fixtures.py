"""The vemmech test-grid fixtures, and the elasticity problems built on them.

`square_test`, `square_layers_test`, `square_grid`, `complex_3d_grid`,
`explore_square_grid` and `make_compaction_test` are what MRST's vemmech
examples stand on. A fixture that quietly builds the wrong boundary mask does
not fail loudly - it produces a plausible but stiffer answer - so these check
the properties that would catch that.
"""

import numpy as np
import pytest

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.vemmech.examples.utils.make_compaction_test import (
    CompactionOptions,
    make_compaction_test,
)
from pymrst.modules.vemmech.utils.complex_3d_grid import (
    ComplexGridOptions,
    complex_3d_grid,
)
from pymrst.modules.vemmech.utils.explore_square_grid import explore_square_grid
from pymrst.modules.vemmech.utils.square_grid import GRID_TYPES, square_grid
from pymrst.modules.vemmech.utils.square_layers_test import (
    layer_index,
    square_layers_test,
)
from pymrst.modules.vemmech.utils.square_test import (
    TEST_NAMES_2D,
    TEST_NAMES_3D,
    square_test,
)
from pymrst.modules.vemmech.vem import create_augmented_grid, enu2c, enu2c_2d
from pymrst.modules.vemmech.vem_lin_elast import vem_lin_elast


def _stiffness(grid, E=1.0e9, nu=0.25):
    build = enu2c if grid.griddim == 3 else enu2c_2d
    return build(np.full(grid.cells.num, E), np.full(grid.cells.num, nu))


# --------------------------------------------------------------- square_test


def test_square_test_default_selects_only_the_first_named_case():
    """MRST's `switch` does not fall through, so 'all' is a default, not a set."""
    _, sides, cases = square_test(cart_dims=[4, 4], L=[1.0, 1.0],
                                  triangulate=False)
    assert len(sides) == 4
    named = [c.name for c in cases if c.solution is None]
    assert named == ["original_2D_test"], named


def test_square_test_can_build_every_named_case_on_request():
    _, _, cases = square_test(cart_dims=[4, 4], L=[1.0, 1.0],
                              triangulate=False, every_case=True)
    assert set(TEST_NAMES_2D) <= {case.name for case in cases}


def test_square_test_selects_the_block_a_name_belongs_to():
    """One block builds two cases and only its first label selects it."""
    _, _, cases = square_test(cart_dims=[4, 4], L=[1.0, 1.0],
                              triangulate=False, make_testcases=False,
                              test_name="hanging_rod_2D")
    assert [c.name for c in cases] == ["hanging_rod_2D", "hanging_cantilever_2D"]
    with pytest.raises(ValueError, match="not recognized"):
        square_test(cart_dims=[4, 4], triangulate=False,
                    make_testcases=False, test_name="hanging_cantilever_2D")


def test_square_test_3d_builds_six_sides_and_its_cases():
    _, sides, cases = square_test(cart_dims=[3, 3, 3], L=[1.0, 1.0, 1.0],
                                  triangulate=False)
    assert len(sides) == 6
    assert set(TEST_NAMES_3D) <= {case.name for case in cases}


@pytest.mark.parametrize("dims", [[4, 4], [3, 3, 3]])
def test_square_test_patch_cases_are_reproduced_exactly(dims):
    """The patch cases exist to be exact; anything else is a real failure."""
    grid, _, cases = square_test(cart_dims=dims, L=[1.0] * len(dims),
                                 triangulate=False, E=1e6, nu=0.25)
    dim = grid.griddim
    checked = 0
    for case in cases:
        if case.solution is None:
            continue
        u, _ = vem_lin_elast(grid, case.C, case.el_bc, case.load)
        expected = case.solution(grid.nodes.coords[:, :dim])
        error = np.abs(u - expected).max() / max(np.abs(expected).max(), 1.0)
        assert error < 1e-9, f"{case.name}: {error:.3e}"
        checked += 1
    assert checked == (6 if dim == 2 else 12)


def test_square_test_named_cases_all_solve():
    grid, _, cases = square_test(cart_dims=[4, 4], L=[1.0, 1.0],
                                 triangulate=False, E=1e6, nu=0.25,
                                 make_testcases=False)
    for case in cases:
        u, _ = vem_lin_elast(grid, case.C, case.el_bc, case.load)
        assert np.all(np.isfinite(u)), case.name
        assert np.abs(u).max() > 0, case.name


def test_square_test_triangulation_keeps_the_side_sets_populated():
    """Triangulating every face must not lose the boundary tags pside needs."""
    _, sides, _ = square_test(cart_dims=[3, 3], L=[1.0, 1.0], triangulate=True)
    assert all(side.face.size > 0 for side in sides)
    assert all(side.nodes.size > 0 for side in sides)


# -------------------------------------------------------- square_layers_test


def test_layer_index_places_the_first_material_at_the_bottom():
    grid, _, _ = square_layers_test(cart_dims=[4, 4], L=[1.0, 1.0])
    index = layer_index(grid, [0.5, 0.5], [1.0, 1.0])
    below = grid.cells.centroids[:, 1] < 0.5
    assert np.all(index[below] == 0) and np.all(index[~below] == 1)


def test_layered_material_only_keeps_the_strain_free_patches_exact():
    """A layered material is not compatible with a constant-strain field:
    the traction jumps across the interface, so only translations and
    rotations remain exact solutions."""
    grid, _, cases = square_layers_test(cart_dims=[4, 4], L=[1.0, 1.0],
                                        E=[1.0e6, 5.0e6])
    exact = inexact = 0
    for case in cases:
        if case.solution is None:
            continue
        u, _ = vem_lin_elast(grid, case.C, case.el_bc, case.load)
        expected = case.solution(grid.nodes.coords[:, :2])
        error = np.abs(u - expected).max() / max(np.abs(expected).max(), 1.0)
        if case.name.startswith("cons_disp") or case.name.startswith("rot_"):
            assert error < 1e-9, f"{case.name} must stay exact: {error:.3e}"
            exact += 1
        else:
            inexact += 1
    assert exact and inexact


def test_matching_layers_restore_every_patch_test():
    grid, _, cases = square_layers_test(cart_dims=[4, 4], L=[1.0, 1.0],
                                        E=[1.0e6, 1.0e6])
    for case in cases:
        if case.solution is None:
            continue
        u, _ = vem_lin_elast(grid, case.C, case.el_bc, case.load)
        expected = case.solution(grid.nodes.coords[:, :2])
        assert np.abs(u - expected).max() < 1e-9 * max(np.abs(expected).max(), 1.0)


# --------------------------------------------------------------- square_grid


@pytest.mark.parametrize("grid_type", GRID_TYPES)
def test_every_square_grid_type_builds_valid_cells(grid_type):
    size = [3.0, 3.0]
    grid = compute_geometry(square_grid([4, 4], size, grid_type=grid_type))
    assert grid.cells.num > 0
    assert np.all(grid.cells.volumes > 0)


@pytest.mark.parametrize("grid_type", ["triangle", "pebi", "boxes1", "mixed1"])
def test_non_cartesian_grids_are_tagged_on_all_four_sides(grid_type):
    grid = square_grid([4, 4], [1.0, 1.0], grid_type=grid_type)
    assert grid.cells.tags is not None
    assert {1, 2, 3, 4} <= set(np.unique(grid.cells.tags))


def test_explore_square_grid_covers_every_declared_type():
    grids = explore_square_grid()
    assert set(grids) == set(GRID_TYPES)
    for name, grid in grids.items():
        assert np.all(compute_geometry(grid).cells.volumes > 0), name


# ------------------------------------------------------------ complex_3d_grid


def test_complex_box_triangulates_without_changing_the_volume():
    plain, _ = complex_3d_grid(
        ComplexGridOptions(cart_dims=[3, 3, 3], triangulate=False), "box")
    cut, _ = complex_3d_grid(
        ComplexGridOptions(cart_dims=[3, 3, 3], triangulate=True), "box")
    assert cut.faces.num > plain.faces.num
    assert cut.cells.num == plain.cells.num
    assert np.isclose(cut.cells.volumes.sum(), plain.cells.volumes.sum())


def test_complex_grdecl_case_builds_a_corner_point_grid():
    grid, _ = complex_3d_grid(ComplexGridOptions(), "grdecl")
    assert grid.griddim == 3 and np.all(grid.cells.volumes > 0)


@pytest.mark.parametrize("case", ["sbed", "norne"])
def test_dataset_backed_cases_are_refused_with_a_reason(case):
    with pytest.raises(NotImplementedError, match="dataset"):
        complex_3d_grid(ComplexGridOptions(), case)


# ------------------------------------------------------- make_compaction_test


def _augmented(dims, size):
    return create_augmented_grid(compute_geometry(cart_grid(dims, size)))


def test_compaction_pins_something_but_never_everything():
    grid = _augmented([3, 3, 3], [1.0, 1.0, 1.0])
    el_bc, _ = make_compaction_test(grid, CompactionOptions())
    assert el_bc.disp_mask.any()
    assert not el_bc.disp_mask.all(), "the sides must be free to roll"


def test_compaction_gravity_load_scales_with_density_and_gravity():
    grid = _augmented([2, 2, 2], [1.0, 1.0, 1.0])
    _, load = make_compaction_test(grid, CompactionOptions(),
                                   gravity=10.0, density=3000.0)
    value = load(grid.cells.centroids)
    assert np.allclose(value[:, :2], 0.0)
    assert np.allclose(value[:, 2], -30000.0)


def test_compaction_free_top_releases_nodes():
    grid = _augmented([3, 3, 3], [1.0, 1.0, 1.0])
    free, _ = make_compaction_test(grid, CompactionOptions(free_top=True))
    held, _ = make_compaction_test(grid, CompactionOptions(free_top=False))
    assert held.disp_nodes.size > free.disp_nodes.size


def test_compaction_response_is_linear_in_the_overburden():
    grid = _augmented([4, 4, 4], [10.0, 10.0, 10.0])
    settle = []
    for force in (30000.0, 60000.0):
        el_bc, load = make_compaction_test(
            grid, CompactionOptions(gravity_load=False), top_force=force)
        u, _ = vem_lin_elast(grid, _stiffness(grid), el_bc, load)
        settle.append(u[np.isclose(grid.nodes.coords[:, 2], 0.0), 2].mean())
    assert settle[0] < 0
    assert np.isclose(settle[1], 2.0 * settle[0], rtol=1e-9)


def test_compaction_holds_the_supported_side():
    grid = _augmented([4, 4, 4], [10.0, 10.0, 10.0])
    el_bc, load = make_compaction_test(grid, CompactionOptions())
    u, _ = vem_lin_elast(grid, _stiffness(grid), el_bc, load)
    supported = np.isclose(grid.nodes.coords[:, 2], 10.0)
    assert np.abs(u[supported, 2]).max() < 1e-12


@pytest.mark.parametrize("dims,size", [([4, 4], [10.0, 10.0]),
                                       ([3, 3, 3], [10.0, 10.0, 10.0])])
def test_compaction_solves_in_both_dimensions(dims, size):
    grid = _augmented(dims, size)
    el_bc, load = make_compaction_test(grid, CompactionOptions())
    u, _ = vem_lin_elast(grid, _stiffness(grid), el_bc, load)
    assert np.all(np.isfinite(u)) and np.abs(u).max() > 0
