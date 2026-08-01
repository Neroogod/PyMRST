"""The vemmech examples, checked against physics rather than just for running.

Two of them - the compaction tests - have a closed-form solution, so those are
real verifications. The other two are checked for the behaviour they exist to
demonstrate: that VEM copes with a grid of mixed cell shapes, and that a
stiffness contrast shows up in the strain field.
"""

import numpy as np
import pytest

from pymrst.modules.vemmech.examples.compaction_test_2d import (
    analytic_displacement,
    run_compaction_test_2d,
)
from pymrst.modules.vemmech.examples.compaction_test_3d import run_compaction_test_3d
from pymrst.modules.vemmech.examples.example_2d import run_example_2d
from pymrst.modules.vemmech.examples.example_2d_layers import run_example_2d_layers


# ------------------------------------------------------- compaction, verified


@pytest.mark.parametrize("triangulate", [False, True])
def test_2d_compaction_matches_its_analytic_solution(triangulate):
    result = run_compaction_test_2d(cart_dims=[10, 10], L=[15.0, 15.0],
                                    triangulate=triangulate, twist=False)
    assert result.relative_error < 1e-10, result.relative_error


def test_2d_compaction_converges_on_triangulated_grids():
    """Guards the tag-alignment defect: this plateaued at 1.2 instead of
    converging, while the untriangulated grid stayed perfect."""
    errors = [run_compaction_test_2d(cart_dims=[n, n], triangulate=True,
                                     twist=False).relative_error
              for n in (5, 10, 20)]
    assert max(errors) < 1e-10, errors


def test_2d_compaction_tolerates_a_twisted_grid():
    result = run_compaction_test_2d(cart_dims=[10, 10], twist=True,
                                    triangulate=True)
    assert result.relative_error < 1e-2, result.relative_error


@pytest.mark.parametrize("triangulate", [False, True])
def test_3d_compaction_matches_its_analytic_solution(triangulate):
    result = run_compaction_test_3d(triangulate=triangulate)
    assert result.relative_error < 1e-10, result.relative_error


def test_compaction_without_gravity_still_matches_the_overburden_profile():
    result = run_compaction_test_2d(cart_dims=[8, 8], twist=False,
                                    triangulate=False, gravity_load=False,
                                    top_load=True)
    assert result.relative_error < 1e-10


def test_analytic_displacement_vanishes_at_the_fixed_end():
    z = np.linspace(0.0, 15.0, 11)
    u = analytic_displacement(z, start=15.0, top=0.0, stiffness=6.4e8,
                              face_force=0.0, gravity_load=True)
    assert np.isclose(u[-1], 0.0), "the supported end does not move"
    assert np.all(u[:-1] < 0), "everything else settles"


# ----------------------------------------------------- demonstration examples


def test_example_2d_runs_on_a_grid_of_mixed_cell_shapes():
    result = run_example_2d(dims=(8, 8))
    assert result.case_name == "original_2D_test"
    assert result.grid.cells.num > 100, "mixed4 glues several blocks together"
    assert np.all(np.isfinite(result.u))
    assert np.abs(result.u).max() > 0
    assert result.divergence.size == result.grid.cells.num


def test_example_2d_cells_are_not_all_the_same_shape():
    """The point of the mixed4 grid: VEM assembles over varied polygons."""
    grid = run_example_2d(dims=(8, 8)).grid
    sides = np.diff(grid.cells.facePos)
    assert np.unique(sides).size > 1, "expected cells with differing face counts"


def test_example_2d_layers_shows_the_stiffness_contrast():
    """The stiff middle layer must strain far less than the soft outer ones."""
    result = run_example_2d_layers()
    assert np.all(np.isfinite(result.u))
    means = [abs(result.divergence[result.layer == k].mean())
             for k in range(result.layer.max() + 1)]
    assert means[1] < means[0] and means[1] < means[2], means


def test_example_2d_layers_assigns_every_cell_a_layer():
    result = run_example_2d_layers()
    assert result.layer.size == result.grid.cells.num
    assert set(np.unique(result.layer)) == {0, 1, 2}


# ------------------------------------------------- complex grid and coupling


def test_complex_grid_example_solves_on_a_domain_with_holes():
    from pymrst.modules.vemmech.examples.example_2d_complex_grid import (
        run_example_2d_complex_grid,
    )

    result = run_example_2d_complex_grid()
    assert result.removed > 0, "the example is about cutting cells out"
    assert np.all(np.isfinite(result.u))
    assert np.all(result.grid.cells.volumes > 0)
    sides = np.diff(result.grid.cells.facePos)
    assert np.unique(sides).size > 2, "expected a genuine mix of cell shapes"
    assert result.stress.shape[0] == result.grid.cells.num


def test_complex_grid_example_is_held_on_its_vertical_sides():
    from pymrst.modules.vemmech.examples.example_2d_complex_grid import (
        run_example_2d_complex_grid,
    )

    result = run_example_2d_complex_grid()
    coords = result.grid.nodes.coords
    held = np.isclose(coords[:, 0], 0.0) | np.isclose(coords[:, 0], 1.0)
    assert np.abs(result.u[held]).max() < 1e-12
    assert np.abs(result.u[~held]).max() > 0


def test_poroelasticity_pressurizes_and_rebounds():
    """Injection raises pressure, and the pressurized pore space pushes the
    skeleton back up: settlement must shrink, not grow."""
    from pymrst.modules.vemmech.examples.example_poroelasticity import (
        run_example_poroelasticity,
    )

    result = run_example_poroelasticity(steps=5)
    assert np.all(np.isfinite(result.u))
    assert np.all(np.isfinite(result.pressure))
    assert result.pressure.max() > result.pressure_history[0].min()
    first, last = result.settlement_history[0], result.settlement_history[-1]
    assert last > first, (first, last)
    assert abs(last) < abs(result.u0[:, 1].min())


def test_poroelasticity_approaches_a_steady_state():
    from pymrst.modules.vemmech.examples.example_poroelasticity import (
        run_example_poroelasticity,
    )

    result = run_example_poroelasticity(steps=8)
    changes = np.abs(np.diff(result.settlement_history))
    assert changes[-1] < changes[0], changes


def test_poroelasticity_couples_both_fluid_terms():
    """With zero pressure everywhere the coupled step must reproduce the dry
    elastic solve, which only holds if the two fluid terms cancel correctly."""
    from pymrst.modules.vemmech.examples.example_poroelasticity import (
        run_example_poroelasticity,
    )

    result = run_example_poroelasticity(steps=1, initial_pressure=0.0,
                                        bhp=0.0, boundary_pressure=0.0)
    assert np.all(np.isfinite(result.u))
    assert np.abs(result.u - result.u0).max() < 0.05 * abs(result.u0[:, 1].min())


# --------------------------------------------------------------- calibration


def test_calibration_reduces_its_objective_substantially():
    """The verifiable claim: the optimizer improves the fit by a wide margin.

    Recovering the exact controls is not verifiable - the problem is
    under-determined by this data - so that is deliberately not asserted.
    """
    from pymrst.modules.vemmech.examples.simple_matching_example import (
        run_simple_matching_example,
    )

    result = run_simple_matching_example(grid_res=(5, 5, 5), max_it=25)
    assert np.isfinite(result.objective)
    assert result.objective < 0.5 * result.initial_objective, (
        result.initial_objective, result.objective)


def test_calibration_recovers_the_principal_orientation():
    """Orientation is the quantity this data does constrain."""
    from pymrst.modules.vemmech.examples.simple_matching_example import (
        run_simple_matching_example,
    )

    result = run_simple_matching_example(grid_res=(5, 5, 5), max_it=25)
    assert result.mean_error["theta"] < result.initial_error["theta"]


def test_calibration_produces_finite_stresses_and_moves_the_controls():
    from pymrst.modules.vemmech.examples.simple_matching_example import (
        run_simple_matching_example,
    )

    result = run_simple_matching_example(grid_res=(5, 5, 5), max_it=15)
    for name, values in result.matched.items():
        assert np.all(np.isfinite(values)), name
        assert values.shape[0] == result.grid.cells.num
    assert not np.allclose(result.u_opt, result.u_target), (
        "the optimizer must actually move away from the initial guess")
