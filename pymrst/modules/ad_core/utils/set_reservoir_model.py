"""MRST ``ad-core/utils/setReservoirModel.m``.

Replace the reservoir model under a model: in both halves of a sequential
model, at the bottom of a chain of wrappers, or the model itself when it
wraps nothing. MATLAB returns a changed copy; so does this.
"""

from __future__ import annotations

from copy import copy
from typing import Any


def set_reservoir_model(model: Any, rmodel: Any) -> Any:
    """MRST `model = setReservoirModel(model, rmodel)`."""
    from pymrst.modules.ad_core.ad_oo.reservoir_model import ReservoirModel
    from pymrst.modules.ad_core.utils.get_reservoir_model import _is_a
    if _is_a(model, "SequentialPressureTransportModel"):
        model = copy(model)
        model.pressureModel = set_reservoir_model(model.pressureModel, rmodel)
        model.transportModel = set_reservoir_model(model.transportModel, rmodel)
        return model
    if not hasattr(model, "parentModel"):
        return rmodel
    model = copy(model)
    if not isinstance(model.parentModel, ReservoirModel):
        model.parentModel = set_reservoir_model(model.parentModel, rmodel)
        return model
    model.parentModel = rmodel
    return model


setReservoirModel = set_reservoir_model

__all__ = ["setReservoirModel", "set_reservoir_model"]


if __name__ == "__main__":
    from types import SimpleNamespace

    def test_a_model_without_a_parent_is_replaced():
        new = SimpleNamespace()
        assert set_reservoir_model(SimpleNamespace(), new) is new
        print("ok test_a_model_without_a_parent_is_replaced")

    def test_the_bottom_of_a_chain_is_replaced_on_copies():
        """Wrappers are descended until the parent is a `ReservoirModel`
        (here, never), and each level is a copy."""
        inner = SimpleNamespace(parentModel=SimpleNamespace(parentModel=None))
        outer = SimpleNamespace(parentModel=inner)
        new = SimpleNamespace()
        out = set_reservoir_model(outer, new)
        assert out is not outer and out.parentModel.parentModel.parentModel is new
        assert inner.parentModel.parentModel is None
        print("ok test_the_bottom_of_a_chain_is_replaced_on_copies")

    test_a_model_without_a_parent_is_replaced()
    test_the_bottom_of_a_chain_is_replaced_on_copies()
    print("\nAll tests passed!")
