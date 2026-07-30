"""Residual-level tests for MRST ``addFluxesFromSourcesAndBC``."""

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
    GenericBlackOilModel,
)
from pymrst.modules.ad_core.statefunctions.state_function import State
from pymrst.modules.ad_core.utils.boundary_conditions import add_bc_and_sources
from pymrst.modules.ad_props import init_simple_adi_fluid
from pymrst.params import make_rock
from pymrst.params.wells_and_bc import add_source
from pymrst.utils.units import BARSA, CENTI, DARCY, MILLI, POISE


def _case():
    grid = compute_geometry(cart_grid([2, 1, 1], [20.0, 10.0, 10.0]))
    rock = make_rock(grid, 100 * MILLI * DARCY, 0.25)
    fluid = init_simple_adi_fluid(
        mu=np.array([1.0, 5.0]) * CENTI * POISE,
        rho=[1000.0, 800.0],
        n=[2, 2],
        c=[0.0, 0.0],
        p_ref=200 * BARSA,
    )
    model = GenericBlackOilModel(grid, rock, fluid, [])
    state = State(
        pressure=np.full(2, 200 * BARSA),
        sW=np.array([0.25, 0.75]),
        qWs=np.zeros(0),
        qOs=np.zeros(0),
        bhp=np.zeros(0),
    )
    return model, state


def test_no_forces_leave_component_equations_unchanged():
    model, state = _case()
    equations = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
    output = add_bc_and_sources(model, equations, state)
    assert output is equations
    assert np.array_equal(output[0], [1.0, 2.0])
    assert np.array_equal(output[1], [3.0, 4.0])


def test_surface_source_is_scattered_into_component_residuals():
    model, state = _case()
    source = add_source(
        None,
        np.array([0]),
        np.array([2.0]),
        sat=np.array([[1.0, 0.0]]),
    )
    equations = [np.zeros(2), np.zeros(2)]
    output = add_bc_and_sources(
        model, equations, state, src=source
    )

    # PyMRST's component equations use surface volume, so a two-unit pure
    # water source contributes exactly two units in cell zero and no oil.
    assert np.allclose(output[0], [-2.0, 0.0])
    assert np.allclose(output[1], [0.0, 0.0])
