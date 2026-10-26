"""MRST ``ad-core/utils/setTimeDiscretization.m``.

`set_time_discretization(model, type, ...)` gives a generic model the flow
state builder for the time discretization `type`: 'fully-implicit'/'fim',
'adaptive-implicit'/'aim' or 'explicit'/'impes', constructed with the given
options (e.g. `saturationCFL`, `verbose`). A sequential model gets it on its
transport model's parent model. Only the AD-OO "Generic" models support
this; others are refused, as in MRST.

`set_time_discretization_ad`, below, is PyMRST's earlier version for the
legacy models, kept for its callers.
"""

from __future__ import annotations

from typing import Any

from pymrst.modules.ad_core.statefunctions.flux.flow_state import (
    AdaptiveImplicitFlowStateBuilder,
    ExplicitFlowStateBuilder,
    ImplicitFlowStateBuilder,
)
from pymrst.modules.ad_core.utils.cfl import estimate_flow_cfl


def _field(value, *names):
    for name in names:
        if isinstance(value, dict) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return None


def _set_field(value, name, item):
    if isinstance(value, dict):
        value[name] = item
    else:
        setattr(value, name, item)


def _normalize_options(options):
    aliases = {
        "saturationCFL": "saturation_cfl",
        "compositionCFL": "composition_cfl",
        "explicitFlux": "explicit_flux",
        "explicitFlow": "explicit_flow",
        "explicitPVT": "explicit_pvt",
        "initialStep": "initial_step",
        "changeTimestep": "change_timestep",
        "firstStepImplicit": "first_step_implicit",
        "implicitWells": "implicit_wells",
        "cflEstimator": "cfl_estimator",
    }
    return {aliases.get(key, key): value for key, value in options.items()}


def _set_builder(model, builder):
    flow = _field(model, "FlowDiscretization", "flow_discretization")
    if flow is None:
        validate = _field(model, "validate_model", "validateModel")
        if validate is None:
            raise TypeError(
                "model must expose FlowDiscretization or validateModel()"
            )
        validated = validate()
        if validated is not None:
            model = validated
        flow = _field(model, "FlowDiscretization", "flow_discretization")
    if flow is None:
        raise TypeError("model validation did not create FlowDiscretization")

    setter = _field(flow, "set_flow_state_builder", "setFlowStateBuilder")
    if setter is not None:
        updated = setter(builder)
        if updated is not None:
            flow = updated
    else:
        _set_field(flow, "FlowStateBuilder", builder)
    _set_field(model, "FlowDiscretization", flow)
    return model


def set_time_discretization(model: Any, type: str, **options: Any) -> Any:
    """MRST `model = setTimeDiscretization(model, type, ...)`."""
    from pymrst.modules.ad_core.ad_oo.statefunctions.flux.adaptive_implicit_flow_state_builder import (
        AdaptiveImplicitFlowStateBuilder as AIM,
    )
    from pymrst.modules.ad_core.ad_oo.statefunctions.flux.explicit_flow_state_builder import (
        ExplicitFlowStateBuilder as Explicit,
    )
    from pymrst.modules.ad_core.ad_oo.statefunctions.flux.implicit_flow_state_builder import (
        ImplicitFlowStateBuilder as Implicit,
    )
    kind = type.lower()
    if kind in ("fully-implicit", "fim"):
        fb = Implicit(**options)
    elif kind in ("adaptive-implicit", "aim"):
        fb = AIM(**options)
    elif kind in ("explicit", "impes"):
        fb = Explicit(**options)
    else:
        raise ValueError(f"{type} not supported. Valid choices: FIM, AIM or IMPES")
    from copy import copy
    model = copy(model)                       # MATLAB: a value class
    if _is_sequential(model):
        transport = model.transportModel.validate_model()
        transport.parentModel = _set_fsb(transport.parentModel, fb)
        model.transportModel = transport
    else:
        model = _set_fsb(model, fb)
    return model


def _is_sequential(model: Any) -> bool:
    return any(c.__name__ == "SequentialPressureTransportModel" for c in type(model).__mro__)


def _set_fsb(model: Any, fb: Any) -> Any:
    from copy import copy
    from pymrst.modules.ad_core.ad_oo.generic_reservoir_model import GenericReservoirModel
    assert isinstance(model, GenericReservoirModel), (
        'Only the "Generic" class of models support different temporal discretization')
    if getattr(model, "FlowDiscretization", None) is None:
        model = model.validate_model()
    flux = copy(model.FlowDiscretization)
    flux = flux.set_flow_state_builder(fb)
    model = copy(model)
    model.FlowDiscretization = flux
    return model


def set_time_discretization_ad(model, discretization_type, **options):
    """Set temporal discretization to FIM, AIM, or IMPES (legacy models).

    MATLAB-style option names such as ``initialStep`` are accepted alongside
    their Python snake-case equivalents.
    """
    key = str(discretization_type).strip().lower()
    options = _normalize_options(options)
    if key in ("adaptive-implicit", "aim", "explicit", "impes"):
        options.setdefault("cfl_estimator", estimate_flow_cfl)
    if key in ("fully-implicit", "fim"):
        builder = ImplicitFlowStateBuilder(**options)
    elif key in ("adaptive-implicit", "aim"):
        builder = AdaptiveImplicitFlowStateBuilder(**options)
    elif key in ("explicit", "impes"):
        builder = ExplicitFlowStateBuilder(**options)
    else:
        raise ValueError(
            f"{discretization_type!r} is not supported; "
            "valid choices are FIM, AIM, or IMPES"
        )

    transport = _field(model, "transportModel", "transport_model")
    if transport is not None:
        validate = _field(transport, "validate_model", "validateModel")
        if validate is not None:
            validated = validate()
            if validated is not None:
                transport = validated
        parent = _field(transport, "parentModel", "parent_model")
        parent = _set_builder(parent, builder)
        _set_field(transport, "parentModel", parent)
        if hasattr(model, "transport_model"):
            _set_field(model, "transport_model", transport)
        else:
            _set_field(model, "transportModel", transport)
        return model

    return _set_builder(model, builder)


setTimeDiscretization = set_time_discretization


__all__ = ["set_time_discretization", "set_time_discretization_ad", "setTimeDiscretization"]


if __name__ == "__main__":
    import numpy as np

    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_blackoil.models import GenericBlackOilModel, TwoPhaseOilWaterModel
    from pymrst.modules.ad_props import init_simple_adi_fluid
    from pymrst.params.rock import make_rock

    G = compute_geometry(cart_grid([4], [4.0]))
    FLUID = init_simple_adi_fluid(phases="WO", mu=[1e-3, 1e-3], rho=[1000, 800])

    def _generic():
        return GenericBlackOilModel(G, make_rock(G, 1e-12, 0.5), FLUID, gas=False)

    def test_each_type_gets_its_builder():
        names = {"fim": "ImplicitFlowStateBuilder", "fully-implicit": "ImplicitFlowStateBuilder",
                 "AIM": "AdaptiveImplicitFlowStateBuilder",
                 "adaptive-implicit": "AdaptiveImplicitFlowStateBuilder",
                 "explicit": "ExplicitFlowStateBuilder", "IMPES": "ExplicitFlowStateBuilder"}
        for kind, cls in names.items():
            model = set_time_discretization(_generic(), kind)
            assert type(model.FlowDiscretization.get_flow_state_builder()).__name__ == cls
        print("ok test_each_type_gets_its_builder")

    def test_options_reach_the_builder_and_the_input_is_unchanged():
        base = _generic().validate_model()
        model = set_time_discretization(base, "explicit", saturationCFL=5, compositionCFL=np.inf)
        fb = model.FlowDiscretization.get_flow_state_builder()
        assert (fb.saturationCFL, fb.compositionCFL) == (5, np.inf)
        assert type(base.FlowDiscretization.get_flow_state_builder()).__name__ == \
            "ImplicitFlowStateBuilder", "MATLAB returns a changed copy"
        print("ok test_options_reach_the_builder_and_the_input_is_unchanged")

    def test_only_generic_models_are_accepted():
        try:
            set_time_discretization(TwoPhaseOilWaterModel(G, make_rock(G, 1e-12, 0.5), FLUID),
                                    "explicit")
        except AssertionError as refused:
            assert "Generic" in str(refused)
        else:
            raise AssertionError("a non-generic model must be refused")
        try:
            set_time_discretization(_generic(), "rk4")
        except ValueError:
            pass
        else:
            raise AssertionError("an unknown type must be refused")
        print("ok test_only_generic_models_are_accepted")

    test_each_type_gets_its_builder()
    test_options_reach_the_builder_and_the_input_is_unchanged()
    test_only_generic_models_are_accepted()
    print("\nAll tests passed!")
