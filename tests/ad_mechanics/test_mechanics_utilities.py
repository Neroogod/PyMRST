import numpy as np
import pytest

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_mechanics.utils.add_derived_quantities import add_derived_quantities
from pymrst.modules.ad_mechanics.utils.compute_init_disp import compute_init_disp
from pymrst.modules.ad_mechanics.utils.equations_black_oil_mech import equations_black_oil_mech
from pymrst.modules.ad_mechanics.utils.equations_oil_water_mech import equations_oil_water_mech
from pymrst.modules.ad_mechanics.utils.equations_poro_mechanics import equations_poro_mechanics
from pymrst.modules.ad_mechanics.utils.equations_water_mech import equations_water_mech
from pymrst.modules.ad_mechanics.utils.setup_operators_vem import setup_operators_vem
from pymrst.modules.vemmech.vem import ElasticBC, create_augmented_grid, enu2c_2d
from pymrst.modules.ad_mechanics.mechanics import MechanicModel
from pymrst.modules.vemmech.utils.complex_3d_grid import ComplexGridOptions, complex_3d_grid
from pymrst.modules.vemmech.vem import enu2c


def build_operators():
    grid = create_augmented_grid(compute_geometry(cart_grid([1, 1], [1.0, 1.0])))
    nodes = np.arange(grid.nodes.num)
    bc = ElasticBC(
        disp_nodes=nodes,
        disp_mask=np.column_stack([np.ones(nodes.size, dtype=bool), np.zeros(nodes.size, dtype=bool)]),
        disp_vals=np.zeros((nodes.size, 2)),
    )
    C = enu2c_2d(np.array([1.0e9]), np.array([0.25]))
    return grid, setup_operators_vem(grid, C, 1.0, bc)


def test_setup_operators_vem_exposes_complete_python_operator_shapes():
    grid, operators = build_operators()
    assert operators.A.shape == (grid.nodes.num, grid.nodes.num)
    assert operators.gradP.shape == (grid.nodes.num, grid.cells.num)
    assert operators.div.shape == (grid.cells.num, grid.nodes.num)
    assert operators.strain.shape[0] == 3 * grid.cells.num
    assert operators.stress.shape[0] == 3 * grid.cells.num


def test_setup_operators_vem_accepts_mrst_custom_stabilization_matrix():
    grid = create_augmented_grid(compute_geometry(cart_grid([1, 1], [1.0, 1.0])))
    nodes = np.arange(grid.nodes.num)
    bc = ElasticBC(
        disp_nodes=nodes,
        disp_mask=np.column_stack([np.ones(nodes.size, dtype=bool), np.zeros(nodes.size, dtype=bool)]),
        disp_vals=np.zeros((nodes.size, 2)),
    )
    C = enu2c_2d(np.array([1.0e9]), np.array([0.25]))
    ndof = grid.griddim * grid.nodes.num
    custom = np.eye(ndof) * 7.5e8
    direct = setup_operators_vem(grid, C, 1.0, bc, S=custom)
    alias = setup_operators_vem(grid, C, 1.0, bc, stabilization=custom)
    np.testing.assert_allclose(direct.A.toarray(), alias.A.toarray())
    with pytest.raises(ValueError, match="only one"):
        setup_operators_vem(grid, C, 1.0, bc, S=custom, stabilization=custom)


def test_moving_displacement_boundary_updates_rhs_without_reassembly():
    grid = create_augmented_grid(compute_geometry(cart_grid([2, 1], [2.0, 1.0])))
    coords = grid.nodes.coords
    bottom = np.flatnonzero(np.isclose(coords[:, 1], 0.0))
    top = np.flatnonzero(np.isclose(coords[:, 1], 1.0))
    nodes = np.concatenate([bottom, top])
    masks = np.vstack([
        np.ones((bottom.size, 2), dtype=bool),
        np.tile([False, True], (top.size, 1)),
    ])
    bc = ElasticBC(nodes, masks, np.zeros((nodes.size, 2)))
    C = enu2c_2d(np.full(grid.cells.num, 1e9), np.full(grid.cells.num, 0.25))
    model = MechanicModel(grid, C, np.ones(grid.cells.num), bc)
    stiffness = model.operators.A.copy()
    old_rhs = model.operators.rhs.copy()
    old_full = model.operators.u_dir.copy()
    old_div = model.volumetric_strain_full(old_full)
    model.set_dirichlet(top, 1, -1e-3)
    np.testing.assert_allclose(model.operators.A.toarray(), stiffness.toarray())
    assert not np.allclose(model.operators.rhs, old_rhs)
    assert np.all(model.operators.u_dir[2 * top + 1] == -1e-3)
    # Stored full states remain evaluated against their own boundary values.
    np.testing.assert_allclose(model.volumetric_strain_full(old_full), old_div)
    assert np.all(model.volumetric_strain_full(model.operators.u_dir) < 0.0)


def test_mechanics_solves_loaded_general_polyhedral_grid():
    grid, _ = complex_3d_grid(
        ComplexGridOptions(cart_dims=(2, 2, 2), triangulate=True), "box"
    )
    base = np.flatnonzero(np.isclose(grid.nodes.coords[:, 2], 0.0))
    bc = ElasticBC(
        base, np.ones((base.size, 3), dtype=bool), np.zeros((base.size, 3))
    )
    C = enu2c(np.full(grid.cells.num, 1e9), np.full(grid.cells.num, 0.25))
    model = MechanicModel(
        grid, C, np.ones(grid.cells.num), bc,
        load=lambda x: np.tile([0.0, 0.0, -1e4], (x.shape[0], 1)),
    )
    xd = model.solve(np.zeros(grid.cells.num))
    cached_factorization = model._solve_stiffness
    xd_again = model.solve(np.zeros(grid.cells.num))
    assert model._solve_stiffness is cached_factorization
    np.testing.assert_allclose(xd_again, xd)
    u = model.full_displacement(xd)
    assert np.all(np.isfinite(xd))
    assert np.max(np.abs(u.reshape(-1, 3)[base])) < 1e-12
    assert np.min(u.reshape(-1, 3)[:, 2]) < 0.0
    assert np.max(np.abs(model.equations(xd, np.zeros(grid.cells.num)))) < 1e-10


def test_initial_stress_is_assembled_as_equivalent_weak_load():
    grid = create_augmented_grid(compute_geometry(cart_grid([2, 1], [2.0, 1.0])))
    base = np.flatnonzero(np.isclose(grid.nodes.coords[:, 1], 0.0))
    bc = ElasticBC(base, np.ones((base.size, 2), dtype=bool), np.zeros((base.size, 2)))
    C = enu2c_2d(np.full(grid.cells.num, 1e9), np.full(grid.cells.num, 0.25))
    sigma0 = np.tile([0.0, -2e6, 0.0], (grid.cells.num, 1))
    model = MechanicModel(
        grid, C, np.ones(grid.cells.num), bc, initial_stress=sigma0
    )
    assert np.linalg.norm(model.operators.rhs) > 0.0
    xd = model.solve(np.zeros(grid.cells.num))
    np.testing.assert_allclose(
        model.equations(xd, np.zeros(grid.cells.num)), 0.0, atol=1e-14
    )
    with pytest.raises(ValueError, match="initial_stress"):
        MechanicModel(grid, C, np.ones(grid.cells.num), bc, initial_stress=np.zeros((1, 2)))


class FakeMechanics:
    def __init__(self):
        self.G = type("Grid", (), {"griddim": 2, "cells": type("Cells", (), {"num": 1})()})()
        self.operators = type(
            "Operators", (),
            {"is_dir": np.array([True, False, True, False]), "u_dir": np.array([0.0, 0.0, 0.0, 0.0])},
        )()

    def solve(self, pressure):
        return np.array([pressure[0], 2 * pressure[0]])

    def full_displacement(self, xd):
        return np.array([0.0, xd[0], 0.0, xd[1]])

    def diagnostics(self, state):
        u = np.asarray(state["u"])
        return {
            "displacement": u.reshape(-1, 2),
            "volumetric_strain": np.array([u.sum()]),
            "strain": np.array([[1.0, 2.0, 3.0]]),
            "effective_stress": np.array([[4.0, 5.0, 6.0]]),
            "total_stress": np.array([[3.0, 4.0, 6.0]]),
            "displacement_magnitude": np.linalg.norm(u.reshape(-1, 2), axis=1),
        }


def test_compute_init_disp_equilibrium_and_supplied_displacement_paths():
    mech = FakeMechanics()
    coupled = type("Coupled", (), {"mech": mech})()
    equilibrium = compute_init_disp(coupled, {"pressure": np.array([3.0])})
    assert equilibrium["xd"] == pytest.approx([3.0, 6.0])
    assert equilibrium["vdiv"] == pytest.approx([9.0])
    supplied = compute_init_disp(
        coupled, {}, uu=np.array([[0.0, 2.0], [0.0, 4.0]]), mech_equil=False
    )
    assert supplied["xd"] == pytest.approx([2.0, 4.0])
    with pytest.raises(ValueError, match="uu is required"):
        compute_init_disp(coupled, {}, mech_equil=False)


def test_add_derived_quantities_accepts_mechanics_or_coupled_model():
    mech = FakeMechanics()
    for owner in (mech, type("Coupled", (), {"mechModel": mech})()):
        out = add_derived_quantities(owner, {"xd": np.array([1.0, 2.0]), "pressure": np.array([1.0])})
        assert out["uu"].shape == (2, 2)
        assert out["stress"].shape == (1, 3)


def test_equation_landing_functions_delegate_to_operational_models():
    class Model:
        def equations(self, xd, pressure):
            return (xd, pressure)

        def get_equations(self, state0, state, dt):
            return state0, state, dt

    model = Model()
    assert equations_poro_mechanics(model, "xd", "p") == ("xd", "p")
    expected = ({"old": 1}, {"new": 2}, 3.0)
    assert equations_water_mech(model, *expected) == expected
    assert equations_black_oil_mech(model, *expected) == expected


def test_equation_landing_functions_expose_mrst_metadata_and_output_state():
    class Mech:
        def full_displacement(self, xd):
            return np.asarray(xd) + 10.0

    class Model:
        facility = object()
        mech = Mech()
        phase_names = ("water", "oil", "gas")
        n_phases = 3

        def _q_names(self):
            return ["qWs", "qOs", "qGs"]

        def get_equations(self, state0, state, dt):
            return [object() for _ in range(8)]

    state = {"xd": np.array([1.0, 2.0])}
    eqs, names, types, out = equations_black_oil_mech(
        Model(), {}, state, 1.0, {"W": []}, return_metadata=True,
        iteration=2, res_only=True,
    )
    assert len(eqs) == len(names) == len(types) == 8
    assert names == ["water", "oil", "gas", "qWs", "qOs", "qGs",
                     "control", "mechanics"]
    assert types == ["cell"] * 3 + ["well"] * 4 + ["mechanics"]
    assert np.array_equal(out["u"], np.array([11.0, 12.0]))
    assert "u" not in state


def test_oil_water_equation_contract_is_distinct_and_two_phase():
    class Model:
        facility = object()

        def _q_names(self):
            return ["qWs", "qOs"]

        def get_equations(self, state0, state, dt):
            return list(range(6))

    eqs, names, types, _ = equations_oil_water_mech(
        Model(), {}, {}, 1.0, return_metadata=True
    )
    assert eqs == list(range(6))
    assert names == ["water", "oil", "qWs", "qOs", "control", "mechanics"]
    assert types == ["cell", "cell", "well", "well", "well", "mechanics"]
