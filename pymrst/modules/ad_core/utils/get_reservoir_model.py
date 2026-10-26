"""MRST ``ad-core/utils/getReservoirModel.m``.

The reservoir model under a model: follow `parentModel` until there is none.
A sequential model without a parent starts from its pressure model.
"""

from __future__ import annotations

from typing import Any


def get_reservoir_model(model: Any) -> Any:
    """MRST `rmodel = getReservoirModel(model)`."""
    rmodel = model
    if _is_a(rmodel, "SequentialPressureTransportModel") and model.parentModel is None:
        rmodel = model.pressureModel
    while hasattr(rmodel, "parentModel"):
        rmodel = rmodel.parentModel
    return rmodel


def _is_a(model: Any, class_name: str) -> bool:
    return any(c.__name__ == class_name for c in type(model).__mro__)


getReservoirModel = get_reservoir_model

__all__ = ["getReservoirModel", "get_reservoir_model"]


if __name__ == "__main__":
    from types import SimpleNamespace

    def test_it_follows_parent_models():
        base = SimpleNamespace(name="reservoir")
        wrapped = SimpleNamespace(parentModel=SimpleNamespace(parentModel=base))
        assert get_reservoir_model(wrapped) is base
        assert get_reservoir_model(base) is base
        print("ok test_it_follows_parent_models")

    test_it_follows_parent_models()
    print("\nAll tests passed!")
