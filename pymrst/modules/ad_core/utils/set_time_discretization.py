"""Select FIM, AIM, or IMPES temporal flux discretization."""

from __future__ import annotations

from pymrst.modules.ad_core.statefunctions.flux.flow_state import (
    AdaptiveImplicitFlowStateBuilder,
    ExplicitFlowStateBuilder,
    ImplicitFlowStateBuilder,
)


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


def set_time_discretization(model, discretization_type, **options):
    """Set temporal discretization to FIM, AIM, or IMPES.

    MATLAB-style option names such as ``initialStep`` are accepted alongside
    their Python snake-case equivalents.
    """
    key = str(discretization_type).strip().lower()
    options = _normalize_options(options)
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


__all__ = ["set_time_discretization", "setTimeDiscretization"]
