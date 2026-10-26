"""Reference tests for consolidated MRST source/BC helpers.

Exercises ``getSourceFluxesAD`` and ``getBoundaryConditionFluxesAD`` through
PyMRST's combined ``compute_sources_and_boundary_conditions_ad`` entry point.
"""

from types import SimpleNamespace

import numpy as np

from pymrst.modules.ad_core.utils.boundary_conditions import (
    compute_sources_and_boundary_conditions_ad,
)
from pymrst.params.wells_and_bc import BoundaryCondition, Source
from pymrst.utils.adi import init_variables_adi


def _model():
    grid = SimpleNamespace(
        cells=SimpleNamespace(
            num=2,
            centroids=np.array([[0.5, 0.0, 0.0], [1.5, 0.0, 0.0]]),
        ),
        faces=SimpleNamespace(
            neighbors=np.array([[-1, 0], [0, 1], [1, -1]]),
            centroids=np.array([[0.0, 0.0, 0.0],
                                [1.0, 0.0, 0.0],
                                [2.0, 0.0, 0.0]]),
        ),
    )
    from pymrst.modules.ad_core.backends.auto_diff_backend import AutoDiffBackend
    rho_s = np.array([1000.0, 800.0])
    t_all = np.array([2.0, 1.0, 3.0])
    return SimpleNamespace(
        G=grid,
        operators=SimpleNamespace(t_all=t_all, T_all=t_all),
        fluid=SimpleNamespace(rho_s=rho_s),
        gravity_magnitude=0.0,
        # MRST's model API, which the ported functions read.
        get_surface_densities=lambda: rho_s,
        get_active_phases=lambda: np.array([True, True]),
        get_gravity_vector=lambda: np.zeros(3),
        AutoDiffBackend=AutoDiffBackend(),
    )


def _props():
    pressure = [np.array([100.0, 90.0]), np.array([100.0, 90.0])]
    saturation = [np.array([0.2, 0.4]), np.array([0.8, 0.6])]
    mobility = [np.array([1.0, 2.0]), np.array([3.0, 2.0])]
    density = [np.array([1100.0, 1200.0]), np.array([760.0, 720.0])]
    return pressure, saturation, mobility, density


def test_mixed_sources_and_pressure_surface_flux_boundaries():
    model = _model()
    pressure, sat, mob, rho = _props()
    bc = BoundaryCondition(
        face=np.array([0, 2]),
        type=["pressure", "flux"],
        value=np.array([110.0, -8.0]),
        sat=np.array([[0.75, 0.25], [0.1, 0.9]]),
    )
    src = Source(
        cell=np.array([0, 1]),
        rate=np.array([5.0, -4.0]),
        sat=np.array([[1.0, 0.0], [0.0, 1.0]]),
    )
    src_c, bc_c = compute_sources_and_boundary_conditions_ad(
        model, pressure, sat, mob, rho, None, {"bc": bc, "src": src}
    )

    # Source injection uses its declared split. Production uses cell mobility.
    assert np.allclose(src_c.phaseMass[0], [5000.0, -2000.0])
    assert np.allclose(src_c.phaseMass[1], [0.0, -1600.0])
    assert np.allclose(src_c.phaseVolume[0], [5 / 1.1, -2 / 1.2])
    assert np.allclose(src_c.phaseVolume[1], [0.0, -2 / 0.9])

    # 10 Pa pressure inflow: total inside mobility 4, outside split .75/.25.
    assert np.allclose(bc_c.phaseVolume[0], [60.0, -4 / 1.2])
    assert np.allclose(bc_c.phaseVolume[1], [20.0, -4 / 0.9])
    assert np.allclose(bc_c.phaseMass[0], [66_000.0, -4000.0])
    assert np.allclose(bc_c.phaseMass[1], [15_200.0, -3200.0])
    assert bc_c.mapping.shape == (2, 2)
    assert np.array_equal(bc_c.sourceCells, [0, 1])


def test_repeated_cells_mapping_and_dissolution_override():
    model = _model()
    pressure, sat, mob, rho = _props()
    # Both boundary faces map to cell zero in this deliberately corner-like grid.
    model.G.faces.neighbors[2] = np.array([0, -1])
    bc = BoundaryCondition(
        face=np.array([0, 2]),
        type=["flux", "flux"],
        value=np.array([2.0, -3.0]),
        sat=np.array([[0.0, 1.0], [0.0, 1.0]]),
        dissolution=np.array([[0.0, 0.25], [0.0, 0.0]]),
    )
    dissolved = [[None, np.array([0.5, 0.6])], [None, None]]
    _, out = compute_sources_and_boundary_conditions_ad(
        model, pressure, sat, mob, rho, dissolved, {"bc": bc}
    )
    assert out.mapping.shape == (2, 2)
    assert np.array_equal(out.mapping.toarray(), [[1.0, 1.0], [1.0, 1.0]])
    # Gas/oil-like component zero receives the injected override and the
    # producing cell ratio from phase one.
    q1 = np.asarray(out.phaseMass[1]) / 800.0
    expected0 = np.array([0.25 * q1[0], -0.75 + 0.5 * q1[1]])
    assert np.allclose(np.asarray(out.phaseMass[0]) / 1000.0, expected0)


def test_phase_pressure_and_outside_properties_control_inflow():
    model = _model()
    pressure, sat, mob, rho = _props()
    bc = BoundaryCondition(
        face=np.array([0]),
        type=["pressure"],
        value=np.array([[110.0, 120.0]]),
        sat=np.array([[0.5, 0.5]]),
        mob=np.array([[5.0, 7.0]]),
        rho=np.array([[900.0, 640.0]]),
    )
    _, out = compute_sources_and_boundary_conditions_ad(
        model, pressure, sat, mob, rho, None, {"bc": bc}
    )
    # T=2 and dp=[10, 20], so the declared outside mobilities give qR.
    assert np.allclose(np.column_stack(out.phaseVolume), [[100.0, 280.0]])
    # Outside b=[.9, .8], not the cell's inside b=[1.1, .95].
    assert np.allclose(np.column_stack(out.phaseMass), [[90_000.0, 179_200.0]])


def test_fixed_pressure_boundary_retains_cell_pressure_derivative():
    model = _model()
    pressure, sat, mob, rho = _props()
    (p,) = init_variables_adi(pressure[0])
    bc = BoundaryCondition(
        face=np.array([0]), type=["pressure"], value=np.array([110.0]),
        sat=np.array([[0.75, 0.25]]),
    )
    _, out = compute_sources_and_boundary_conditions_ad(
        model, [p, p], sat, mob, rho, None, {"bc": bc}
    )
    # q_w = mobility_out*T*(p_bc-p_cell); p_bc has zero Jacobian.
    derivative = out.phaseVolume[0].jac[0].toarray()
    assert derivative[0, 0] == -6.0
    assert derivative[0, 1] == 0.0


def test_reservoir_flux_boundary_without_gravity():
    model = _model()
    pressure, sat, mob, rho = _props()
    bc = BoundaryCondition(
        face=np.array([0]),
        type=["reservoir"],
        value=np.array([12.0]),
        sat=np.array([[0.25, 0.75]]),
    )
    _, out = compute_sources_and_boundary_conditions_ad(
        model, pressure, sat, mob, rho, None, {"bc": bc}
    )
    # With zero gravity, outside mobility is total mobility times saturation.
    assert np.allclose(np.column_stack(out.phaseVolume), [[3.0, 9.0]])
