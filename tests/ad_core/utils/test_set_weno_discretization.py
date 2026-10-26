"""Numerical coverage for the WENO machinery used by MRST's wenoExampleAD."""

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
    GenericBlackOilModelAD,
)
from pymrst.modules.ad_core.models.wrapper_model import WrapperModel
from pymrst.modules.ad_core.statefunctions.flow_discretization import FaceMobility
from pymrst.modules.ad_core.statefunctions.flux.face_component_mobility import (
    FaceComponentMobility,
)
from pymrst.modules.ad_core.statefunctions.flux.upwind import (
    WENOUpwindDiscretizationAD,
)
from pymrst.modules.ad_core.utils import set_weno_discretization_ad
from pymrst.modules.ad_props import init_simple_adi_fluid
from pymrst.params import make_rock


def _model():
    grid = compute_geometry(cart_grid([8, 1, 1], [8.0, 1.0, 1.0]))
    rock = make_rock(grid, 1e-13, 0.25)
    fluid = init_simple_adi_fluid(
        mu=[1.0, 2.0], rho=[1000.0, 800.0], n=[2, 2]
    )
    return GenericBlackOilModelAD(grid, rock, fluid, [])


def test_installs_weno_on_both_mobility_state_functions():
    model = set_weno_discretization_ad(_model(), dimension=1)
    fm = model.FlowDiscretization.get_state_function("FaceMobility")
    fcm = model.FlowDiscretization.get_state_function("FaceComponentMobility")
    assert isinstance(fm, FaceMobility)
    assert isinstance(fcm, FaceComponentMobility)
    assert isinstance(fm.upstream, WENOUpwindDiscretizationAD)
    assert fcm.upwind_discretization is fm.upstream


def test_wrapper_parent_is_updated():
    wrapped = WrapperModel(_model())
    returned = set_weno_discretization_ad(wrapped, dimension=1)
    assert returned is wrapped
    fm = wrapped.parent_model.FlowDiscretization.get_state_function("FaceMobility")
    assert isinstance(fm.upstream, WENOUpwindDiscretizationAD)
