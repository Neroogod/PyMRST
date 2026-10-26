"""MRST ``sequential/utils/getSequentialModelFromFI.m``.

The sequential pressure-transport split of a fully implicit model. A
"Generic" reservoir model, its state function groupings removed, gets a
`PressureModel` and a `TransportModel` around it; MRST's other models get
their own pressure/transport classes (oil-water, black-oil, polymer,
compositional). They are combined in a `SequentialPressureTransportModel`
whose step is one pressure-transport pass unless `outerLoop` is asked for
(`useAcceleration` selects MRST's `AcceleratedSequentialModel`). A model that
is already sequential is returned as it is.

Only the generic path is ported; the model-specific classes and the
accelerated model are not, and say so. The fully implicit model is copied
before its groupings are removed (MATLAB's value semantics).
"""

from __future__ import annotations

from typing import Any

from pymrst.modules.ad_core.ad_oo.generic_reservoir_model import GenericReservoirModel
from pymrst.modules.ad_core.ad_oo.wrapper_model import value_copy_model


def get_sequential_model_from_fi(fimodel: Any, *, useAcceleration: bool = False,
                                 outerLoop: bool = False, **options: Any) -> Any:
    """MRST `model = getSequentialModelFromFI(fimodel, ...)`."""
    from pymrst.modules.sequential.models.pressure_model import PressureModel
    from pymrst.modules.sequential.models.sequential_pressure_transport_model import (
        SequentialPressureTransportModel,
    )
    from pymrst.modules.sequential.models.transport_model import TransportModel
    if isinstance(fimodel, SequentialPressureTransportModel):
        return fimodel
    if isinstance(fimodel, GenericReservoirModel):
        fimodel = value_copy_model(fimodel).remove_state_function_groupings()
        pressureModel = PressureModel(value_copy_model(fimodel))
        transportModel = TransportModel(value_copy_model(fimodel))
    else:
        raise NotImplementedError(
            f"Sequential model for '{type(fimodel).__name__}': only the generic models' "
            f"PressureModel/TransportModel are ported.")
    if useAcceleration:
        raise NotImplementedError("AcceleratedSequentialModel is not ported.")
    return SequentialPressureTransportModel(pressureModel, transportModel,
                                            stepFunctionIsLinear=not outerLoop, **options)


getSequentialModelFromFI = get_sequential_model_from_fi

__all__ = ["getSequentialModelFromFI", "get_sequential_model_from_fi"]


if __name__ == "__main__":
    import warnings

    from pymrst.modules.sequential.testing import toy_case

    warnings.simplefilter("ignore")

    def test_a_generic_model_is_split_and_left_unchanged():
        case = toy_case()
        seq = get_sequential_model_from_fi(case.model)
        assert type(seq.pressureModel).__name__ == "PressureModel"
        assert type(seq.transportModel).__name__ == "TransportModel"
        assert get_sequential_model_from_fi(seq) is seq
        validated = case.model.validate_model()
        assert type(validated.FlowDiscretization.get_state_function(
            "PhaseFlux")).__name__ == "PhaseFlux", "the original keeps implicit fluxes"
        print("ok test_a_generic_model_is_split_and_left_unchanged")

    def test_only_the_generic_path_is_ported():
        from types import SimpleNamespace
        try:
            get_sequential_model_from_fi(SimpleNamespace())
        except NotImplementedError:
            pass
        else:
            raise AssertionError("expected NotImplementedError")
        print("ok test_only_the_generic_path_is_ported")

    test_a_generic_model_is_split_and_left_unchanged()
    test_only_the_generic_path_is_ported()
    print("\nAll tests passed!")
