from types import SimpleNamespace

import numpy as np
import pytest

from pymrst.modules.ad_core.statefunctions.flux.flow_state import (
    AdaptiveImplicitFlowStateBuilder,
    ExplicitFlowStateBuilder,
    ImplicitFlowStateBuilder,
)
from pymrst.modules.ad_core.utils import set_time_discretization


class _Flow:
    def __init__(self):
        self.FlowStateBuilder = None

    def setFlowStateBuilder(self, builder):
        self.FlowStateBuilder = builder
        return self


def test_fim_aim_and_impes_builders_and_options():
    model = SimpleNamespace(FlowDiscretization=_Flow())
    assert set_time_discretization(model, "fim") is model
    assert isinstance(model.FlowDiscretization.FlowStateBuilder,
                      ImplicitFlowStateBuilder)

    set_time_discretization(
        model, "aim", firstStepImplicit=False, saturationCFL=0.7
    )
    aim = model.FlowDiscretization.FlowStateBuilder
    assert isinstance(aim, AdaptiveImplicitFlowStateBuilder)
    assert aim.first_step_implicit is False
    assert aim.saturation_cfl == 0.7

    set_time_discretization(
        model, "impes", initialStep=12.0, changeTimestep=False
    )
    impes = model.FlowDiscretization.FlowStateBuilder
    assert isinstance(impes, ExplicitFlowStateBuilder)
    assert impes.initial_step == 12.0
    assert impes.change_timestep is False


def test_sequential_wrapper_updates_transport_parent_model():
    parent = SimpleNamespace(FlowDiscretization=_Flow())

    class _Transport:
        parentModel = parent

        def validateModel(self):
            self.validated = True
            return self

    wrapper = SimpleNamespace(transportModel=_Transport())
    out = set_time_discretization(wrapper, "explicit")
    assert out is wrapper
    assert wrapper.transportModel.validated
    assert isinstance(
        wrapper.transportModel.parentModel.FlowDiscretization.FlowStateBuilder,
        ExplicitFlowStateBuilder,
    )


def test_invalid_type_is_rejected():
    with pytest.raises(ValueError, match="valid choices"):
        set_time_discretization(
            SimpleNamespace(FlowDiscretization=_Flow()), "crank-nicolson"
        )
