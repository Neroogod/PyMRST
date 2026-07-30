"""Install WENO face reconstruction on a reservoir model."""

from __future__ import annotations

from pymrst.modules.ad_core.statefunctions.flow_discretization import FaceMobility
from pymrst.modules.ad_core.statefunctions.flux.face_component_mobility import (
    FaceComponentMobility,
)
from pymrst.modules.ad_core.statefunctions.flux.upwind import (
    WENOUpwindDiscretization,
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
    weno = WENOUpwindDiscretization(model, int(dimension), **options)

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


def set_weno_discretization(model, dimension=None, **options):
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

__all__ = ["set_weno_discretization", "setWENODiscretization"]
