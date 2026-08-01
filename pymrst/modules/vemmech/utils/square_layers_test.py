"""Layered square elasticity test cases (MRST `squareLayersTest`).

`squareTest` with one difference that matters: the material is layered in
``y``, so the same boundary conditions produce a discontinuous strain field
across the layer interfaces. That is what makes it a test - a scheme can pass
the homogeneous patch tests and still mishandle a material jump.

The boundary sets and the named cases are shared with `square_test`; only the
stiffness assignment differs, and the grid is never triangulated.
"""

from typing import Any, List, Sequence, Tuple

import numpy as np

from pymrst.gridprocessing.compute_geometry import compute_geometry
from pymrst.modules.vemmech.utils.square_grid import square_grid
from pymrst.modules.vemmech.utils.square_test import (
    SideBC,
    TestCase,
    _cases_2d,
    _cases_3d,
    _patch_test_cases,
    _side_boundary_conditions,
)
from pymrst.modules.vemmech.vem import create_augmented_grid, enu2c, enu2c_2d


def layer_index(grid, zlayers: Sequence[float], L: Sequence[float]) -> np.ndarray:
    """Return the 0-based layer of every cell, counted from the bottom.

    MRST accumulates how many cumulative layer tops a cell sits below and then
    reads the material list backwards, which puts the *first* entry of ``E`` in
    the *bottom* layer. That indirection is preserved here.
    """
    zlayers = np.asarray(zlayers, dtype=float).ravel()
    if not np.isclose(zlayers.sum(), 1.0):
        raise ValueError("the sum of the layer fractions must be one")
    tops = np.cumsum(zlayers) * float(L[1])
    centroid_y = np.asarray(grid.cells.centroids, dtype=float)[:, 1]
    below = (centroid_y[:, None] < tops[None, :]).sum(axis=1)
    return zlayers.size - below


def square_layers_test(
    *,
    L: Sequence[float] = (1.0, 1.0),
    cart_dims: Sequence[int] = (20, 20),
    grid_type: str = "cartgrid",
    disturb: float = 0.0,
    E: Sequence[float] = (1.0, 2.0),
    nu: Sequence[float] = (0.3, 0.3),
    zlayers: Sequence[float] = (0.5, 0.5),
    make_testcases: bool = True,
    test_name: str = "all",
) -> Tuple[Any, List[SideBC], List[TestCase]]:
    """Return ``(grid, side_bcs, test_cases)`` for the layered square domain."""
    L = [float(v) for v in np.atleast_1d(L).ravel()]
    cart_dims = [int(n) for n in np.atleast_1d(cart_dims).ravel()]
    E = np.atleast_1d(np.asarray(E, dtype=float)).ravel()
    nu = np.atleast_1d(np.asarray(nu, dtype=float)).ravel()
    zlayers = np.atleast_1d(np.asarray(zlayers, dtype=float)).ravel()
    if not (E.size == nu.size == zlayers.size):
        raise ValueError("E, nu and zlayers must name the same number of layers")
    if len(cart_dims) != 2 and grid_type != "cartgrid":
        raise ValueError("only 'cartgrid' is defined outside two dimensions")

    grid = square_grid(cart_dims, L, grid_type=grid_type, disturb=disturb)
    grid = create_augmented_grid(compute_geometry(grid))

    side_bcs = _side_boundary_conditions(grid, grid_type, L)
    index = layer_index(grid, zlayers, L)
    build = enu2c if grid.griddim == 3 else enu2c_2d
    C = build(E[index], nu[index])

    cases: List[TestCase] = []
    if make_testcases:
        cases += _patch_test_cases(grid, side_bcs, C)
    cases += (_cases_3d(grid, side_bcs, C) if grid.griddim == 3
              else _cases_2d(grid, side_bcs, C, L, test_name))
    return grid, side_bcs, cases


squareLayersTest = square_layers_test


if __name__ == "__main__":
    from pymrst.modules.vemmech.vem_lin_elast import vem_lin_elast

    def test_two_layers_split_the_domain_in_half() -> None:
        grid, _, _ = square_layers_test(cart_dims=[4, 4], L=[1.0, 1.0])
        index = layer_index(grid, [0.5, 0.5], [1.0, 1.0])
        below = grid.cells.centroids[:, 1] < 0.5
        assert np.all(index[below] == 0), "bottom half takes the first material"
        assert np.all(index[~below] == 1)
        print("ok test_two_layers_split_the_domain_in_half")

    def test_stiffness_differs_between_the_layers() -> None:
        grid, _, cases = square_layers_test(cart_dims=[4, 4], L=[1.0, 1.0],
                                            E=[1.0, 7.0])
        C = cases[0].C
        index = layer_index(grid, [0.5, 0.5], [1.0, 1.0])
        assert not np.allclose(C[index == 0][0], C[index == 1][0])
        # the stiffer layer must have the larger leading Voigt entry
        assert C[index == 1][0][0] > C[index == 0][0][0]
        print("ok test_stiffness_differs_between_the_layers")

    def test_uneven_layers_are_placed_by_fraction() -> None:
        grid, _, _ = square_layers_test(cart_dims=[4, 8], L=[1.0, 1.0],
                                        E=[1.0, 2.0], nu=[0.3, 0.3],
                                        zlayers=[0.25, 0.75])
        index = layer_index(grid, [0.25, 0.75], [1.0, 1.0])
        below = grid.cells.centroids[:, 1] < 0.25
        assert np.all(index[below] == 0) and np.all(index[~below] == 1)
        print("ok test_uneven_layers_are_placed_by_fraction")

    def test_only_the_strain_free_fields_stay_exact_across_the_jump() -> None:
        """Layering breaks the strain-producing patch tests, and must.

        A field like ``u = (x, 0)`` has constant strain, so a layered material
        gives it a different stress in each layer. Across a horizontal
        interface that is a jump in the traction ``sigma . y``, which is not an
        equilibrium state - the field is no longer a solution of the layered
        problem. Only the strain-free fields, the translations and rotations,
        survive; a scheme that reproduced the others would be wrong.
        """
        grid, _, cases = square_layers_test(cart_dims=[4, 4], L=[1.0, 1.0],
                                            E=[1.0e6, 5.0e6])
        exact, inexact = 0, 0
        for case in cases:
            if case.solution is None:
                continue
            u, _ = vem_lin_elast(grid, case.C, case.el_bc, case.load)
            expected = case.solution(grid.nodes.coords[:, :2])
            error = np.abs(u - expected).max() / max(np.abs(expected).max(), 1.0)
            strain_free = (case.name.startswith("cons_disp")
                           or case.name.startswith("rot_"))
            if strain_free:
                assert error < 1e-9, f"{case.name} must stay exact: {error:.3e}"
                exact += 1
            else:
                inexact += 1
        assert exact > 0 and inexact > 0
        print(f"ok test_only_the_strain_free_fields_stay_exact_across_the_jump "
              f"({exact} exact, {inexact} legitimately not)")

    def test_a_uniform_material_restores_every_patch_test() -> None:
        """The same fields are exact again once the layers match, which shows
        the failures above come from the material jump and nothing else."""
        grid, _, cases = square_layers_test(cart_dims=[4, 4], L=[1.0, 1.0],
                                            E=[1.0e6, 1.0e6])
        checked = 0
        for case in cases:
            if case.solution is None:
                continue
            u, _ = vem_lin_elast(grid, case.C, case.el_bc, case.load)
            expected = case.solution(grid.nodes.coords[:, :2])
            error = np.abs(u - expected).max() / max(np.abs(expected).max(), 1.0)
            assert error < 1e-9, f"{case.name}: {error:.3e}"
            checked += 1
        print(f"ok test_a_uniform_material_restores_every_patch_test "
              f"({checked} fields)")

    def test_named_cases_solve_on_a_layered_domain() -> None:
        grid, _, cases = square_layers_test(cart_dims=[4, 4], L=[1.0, 1.0],
                                            E=[1.0e6, 5.0e6],
                                            make_testcases=False)
        for case in cases:
            u, _ = vem_lin_elast(grid, case.C, case.el_bc, case.load)
            assert np.all(np.isfinite(u)), case.name
        print(f"ok test_named_cases_solve_on_a_layered_domain ({len(cases)} cases)")

    def test_layer_fractions_must_sum_to_one() -> None:
        try:
            square_layers_test(cart_dims=[4, 4], zlayers=[0.3, 0.3])
        except ValueError:
            print("ok test_layer_fractions_must_sum_to_one")
            return
        raise AssertionError("layer fractions must be checked")

    def test_material_lists_must_agree_in_length() -> None:
        try:
            square_layers_test(cart_dims=[4, 4], E=[1.0, 2.0, 3.0])
        except ValueError:
            print("ok test_material_lists_must_agree_in_length")
            return
        raise AssertionError("mismatched material lists must be rejected")

    test_two_layers_split_the_domain_in_half()
    test_stiffness_differs_between_the_layers()
    test_uneven_layers_are_placed_by_fraction()
    test_only_the_strain_free_fields_stay_exact_across_the_jump()
    test_a_uniform_material_restores_every_patch_test()
    test_named_cases_solve_on_a_layered_domain()
    test_layer_fractions_must_sum_to_one()
    test_material_lists_must_agree_in_length()
    print("\nAll tests passed!")
