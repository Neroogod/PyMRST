"""M4.2/M4.3 surrogate contract, OOD, fallback and evidence accounting."""

import numpy as np
import pytest

from pymrst.research_studio.components import (
    GuardedSurrogateRuntime, ReferenceSamplingPolicy, SurrogateContract,
    SurrogatePrediction, TensorField, ValidityDomain,
)


def _contract():
    return SurrogateContract(
        "three-component-flash-v1",
        inputs=(TensorField("pressure", "Pa", (None, 1)),
                TensorField("z1", "fraction", (None, 1)),
                TensorField("z2", "fraction", (None, 1)),
                TensorField("z3", "fraction", (None, 1))),
        outputs=(TensorField("vapor_fraction", "fraction", (None, 1)),),
        validity=ValidityDomain(
            {"pressure": (1e5, 5e7), "z1": (0.0, 1.0), "z2": (0.0, 1.0),
             "z3": (0.0, 1.0)}, composition_fields=("z1", "z2", "z3")),
        training_data_fingerprint="sha256:training", uncertainty_kind="ensemble",
        invariant_names=("vapor_fraction_bounds",),
        physics_fallback="numpy:zeros_like",
        reference_sampling=ReferenceSamplingPolicy(0.5, 1, seed=7),
        maximum_batch_size=8)


def _inputs(pressure=1e7):
    return {"pressure": np.array([[pressure], [pressure]]),
            "z1": np.array([[0.2], [0.3]]), "z2": np.array([[0.3], [0.2]]),
            "z3": np.array([[0.5], [0.5]])}


def _physics(values):
    return {"vapor_fraction": 0.5*values["z3"]}


def test_in_domain_surrogate_is_sampled_against_physics_and_error_recorded():
    def predictor(values, device):
        return SurrogatePrediction({"vapor_fraction": 0.5*values["z3"]+1e-4},
                                   uncertainty=np.zeros((2, 1)))
    runtime = GuardedSurrogateRuntime(_contract(), predictor, fallback=_physics,
                                      uncertainty_limit=0.1)
    output = runtime(_inputs())
    assert output["vapor_fraction"].shape == (2, 1)
    event = runtime.events[0]
    assert event.used_surrogate and event.reference_samples == 1
    assert event.maximum_reference_error == pytest.approx(1e-4)
    assert runtime.report()["estimated_accumulated_error"] == pytest.approx(1e-4)


def test_ood_uncertainty_and_invariant_failures_each_force_physics():
    calls = {"predictor": 0, "fallback": 0}
    def fallback(values):
        calls["fallback"] += 1
        return _physics(values)
    def predictor(values, device):
        calls["predictor"] += 1
        return SurrogatePrediction({"vapor_fraction": np.ones((2, 1))},
                                   uncertainty=np.ones((2, 1)))
    runtime = GuardedSurrogateRuntime(_contract(), predictor, fallback=fallback,
                                      uncertainty_limit=0.1)
    runtime(_inputs(pressure=9e7))
    assert runtime.events[-1].fallback_reason == "out_of_domain"
    assert calls["predictor"] == 0
    runtime(_inputs())
    assert runtime.events[-1].fallback_reason == "uncertainty_limit"
    assert calls == {"predictor": 1, "fallback": 2}

    def invalid(values, device):
        return SurrogatePrediction({"vapor_fraction": np.ones((2, 1))},
                                   invariant_failures=("vapor_fraction_bounds",))
    guarded = GuardedSurrogateRuntime(_contract(), invalid, fallback=fallback)
    guarded(_inputs())
    assert guarded.events[0].fallback_reason == "invariant_failure"


def test_schema_composition_batch_and_device_errors_are_explicit():
    contract = _contract()
    bad = _inputs()
    bad["z3"][:] = 0.4
    _, reasons = contract.validate_inputs(bad)
    assert "composition does not sum to one" in reasons
    with pytest.raises(ValueError, match="inputs differ"):
        contract.validate_inputs({"pressure": np.ones((1, 1))})
    with pytest.raises(ValueError, match="does not support cuda"):
        GuardedSurrogateRuntime(contract, lambda x: x, device="cuda")
