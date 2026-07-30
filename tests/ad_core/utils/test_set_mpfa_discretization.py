from types import SimpleNamespace

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_core.statefunctions.flow_discretization import (
    FlowDiscretization,
)
from pymrst.modules.ad_core.utils import set_mpfa_discretization
from pymrst.modules.ad_core.utils.operators import setup_operators_tpfa
from pymrst.params import make_rock
from pymrst.utils.units import DARCY, MILLI


def _model():
    grid = compute_geometry(cart_grid([4, 3], [4.0, 3.0]))
    # Anisotropic full tensor makes MPFA genuinely different from TPFA.
    perm = np.tile(
        np.array([100.0, 35.0, 55.0]) * MILLI * DARCY,
        (grid.cells.num, 1),
    )
    rock = make_rock(grid, perm, 0.2)
    model = SimpleNamespace(
        G=grid,
        rock=rock,
        operators=setup_operators_tpfa(grid, rock),
        fluid=SimpleNamespace(has_pc=False),
        component_state_keys=lambda: [],
    )
    model.FlowDiscretization = FlowDiscretization(model)
    return model


def test_mpfa_gradient_is_conservative_and_annihilates_constants():
    model = set_mpfa_discretization(_model())
    gradient = model.operators.mpfagrad
    assert gradient.shape == (
        model.operators.N.shape[0],
        model.G.cells.num,
    )
    assert np.allclose(np.asarray(gradient @ np.ones(model.G.cells.num)), 0.0)
    pressure = model.FlowDiscretization.get_state_function("PressureGradient")
    values = np.arange(model.G.cells.num, dtype=float)
    assert np.allclose(pressure.Grad(values), gradient @ values)


def test_mpfa_installs_gravity_weight_and_supports_wrapper():
    parent = _model()
    wrapper = SimpleNamespace(parentModel=parent)
    assert set_mpfa_discretization(wrapper) is wrapper
    gravity = wrapper.parentModel.FlowDiscretization.get_state_function(
        "GravityPotentialDifference"
    )
    nfaces = wrapper.parentModel.operators.N.shape[0]
    assert gravity.weight.shape == (nfaces, nfaces)
