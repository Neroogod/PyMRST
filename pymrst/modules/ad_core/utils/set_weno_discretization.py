"""MRST ``ad-core/utils/setWENODiscretization.m``.

A copy of the model whose `FaceMobility` and `FaceComponentMobility` upwind
with `WENOUpwindDiscretization` (of the grid's dimension, with any options
passed on to it), setting up the state-function groupings first if the model
has none. A `WrapperModel` gets this on its parent.

`set_weno_discretization_ad` is PyMRST's earlier version for the legacy
models, kept for its callers: it changes the model in place and takes an
explicit `dimension`.
"""

from __future__ import annotations

from copy import copy
from typing import Any

from pymrst.modules.ad_core.statefunctions.flow_discretization import FaceMobility
from pymrst.modules.ad_core.statefunctions.flux.face_component_mobility import (
    FaceComponentMobility,
)
from pymrst.modules.ad_core.statefunctions.flux.upwind import (
    WENOUpwindDiscretizationAD,
)


def _field(obj, *names):
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def _set(obj, name, value):
    if isinstance(obj, dict):
        obj[name] = value
    else:
        setattr(obj, name, value)


def _install(model, dimension=None, **options):
    grid = _field(model, "G")
    if grid is None:
        raise TypeError("WENO requires model.G")
    if dimension is None:
        dimension = _field(grid, "griddim", "grid_dim")
    if dimension is None:
        dimension = grid.cells.centroids.shape[1]
    weno = WENOUpwindDiscretizationAD(model, int(dimension), **options)

    flow = _field(model, "FlowDiscretization")
    if flow is None:
        setup = _field(model, "setup_state_function_groupings",
                       "setupStateFunctionGroupings")
        if setup is None:
            raise TypeError("model does not expose FlowDiscretization")
        updated = setup()
        if updated is not None:
            model = updated
        flow = _field(model, "FlowDiscretization")

    flow.set_state_function("FaceMobility", FaceMobility(model, weno))
    flow.set_state_function(
        "FaceComponentMobility", FaceComponentMobility(model, weno)
    )
    _set(model, "FlowDiscretization", flow)
    return model


def set_weno_discretization(model: Any, **options: Any) -> Any:
    """MRST `setWENODiscretization`."""
    from pymrst.modules.ad_core.ad_oo.wrapper_model import WrapperModel
    is_wrapper = isinstance(model, WrapperModel)
    m = model.parentModel if is_wrapper else model
    m = _set_weno(m, **options)
    if is_wrapper:
        model = copy(model)
        model.parentModel = m
        return model
    return m


def _set_weno(model: Any, **options: Any) -> Any:
    from pymrst.modules.ad_core.ad_oo.statefunctions.flux.face_component_mobility import (
        FaceComponentMobility as FaceComponentMobilityOO,
    )
    from pymrst.modules.ad_core.ad_oo.statefunctions.flux.face_mobility import (
        FaceMobility as FaceMobilityOO,
    )
    from pymrst.modules.ad_core.ad_oo.statefunctions.flux.weno_upwind_discretization import (
        WENOUpwindDiscretization as WENOUpwindDiscretizationOO,
    )
    from pymrst.modules.ad_core.ad_oo.wrapper_model import value_copy_model
    weno = WENOUpwindDiscretizationOO(model, model.G.griddim, **options)
    model = value_copy_model(model)
    if model.FlowDiscretization is None:
        model = model.setup_state_function_groupings()
    fd = copy(model.FlowDiscretization)
    fd.functions = dict(fd.functions)
    fd = fd.set_state_function("FaceMobility", FaceMobilityOO(model, weno))
    fd = fd.set_state_function("FaceComponentMobility", FaceComponentMobilityOO(model, weno))
    model.FlowDiscretization = fd
    return model


def set_weno_discretization_ad(model, dimension=None, **options):
    """Replace phase and component face mobility upwinding by WENO.

    Wrapper models are updated through their parent, matching MRST.
    ``dimension`` can be supplied for grids embedded in a higher-dimensional
    coordinate array (for example a 1-D Cartesian column stored with xyz
    centroids).
    """
    parent = _field(model, "parent_model", "parentModel")
    if parent is None:
        return _install(model, dimension, **options)
    parent = _install(parent, dimension, **options)
    _set(model, "parent_model" if hasattr(model, "parent_model")
         else "parentModel", parent)
    return model


setWENODiscretization = set_weno_discretization

__all__ = ["set_weno_discretization", "set_weno_discretization_ad",
           "setWENODiscretization"]


if __name__ == "__main__":
    import contextlib
    import io

    import numpy as np

    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
        GenericBlackOilModel,
    )
    from pymrst.modules.ad_props import init_simple_adi_fluid
    from pymrst.params.rock import make_rock

    def _model():
        G = compute_geometry(cart_grid([4, 3], [4.0, 3.0]))
        fluid = init_simple_adi_fluid(phases="WO", n=[2, 2], rho=[1000, 800], mu=[1, 5])
        return GenericBlackOilModel(G, make_rock(G, 1, 1), fluid, gas=False)

    def test_face_mobilities_upwind_with_weno_on_a_copy():
        model = _model()
        with contextlib.redirect_stdout(io.StringIO()):
            weno_model = set_weno_discretization(model, cap=False)
        fd = weno_model.FlowDiscretization
        for name in ("FaceMobility", "FaceComponentMobility"):
            upstream = fd.get_state_function(name).UpwindDiscretization
            assert type(upstream).__name__ == "WENOUpwindDiscretization"
            assert upstream.dim == 2 and upstream.cap is False
        assert model.FlowDiscretization is None or type(
            model.FlowDiscretization.get_state_function("FaceMobility").UpwindDiscretization
        ).__name__ != "WENOUpwindDiscretization"
        print("ok test_face_mobilities_upwind_with_weno_on_a_copy")

    def test_a_wrapper_gets_it_on_its_parent():
        from pymrst.modules.ad_core.ad_oo.wrapper_model import WrapperModel
        wrapper = WrapperModel(_model())
        with contextlib.redirect_stdout(io.StringIO()):
            out = set_weno_discretization(wrapper)
        assert isinstance(out, WrapperModel) and out is not wrapper
        fm = out.parentModel.FlowDiscretization.get_state_function("FaceMobility")
        assert type(fm.UpwindDiscretization).__name__ == "WENOUpwindDiscretization"
        print("ok test_a_wrapper_gets_it_on_its_parent")

    test_face_mobilities_upwind_with_weno_on_a_copy()
    test_a_wrapper_gets_it_on_its_parent()
    print("\nAll tests passed!")
