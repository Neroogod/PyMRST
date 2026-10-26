"""M4.1 capability matching fails before numerical execution."""

import pytest

from pymrst.research_studio.components import (
    CapabilityRegistry, ComponentCapability, ComponentRequest,
    IncompatibleComponents, registered_reference_components,
)


def test_reference_blackoil_stack_is_declared_and_compatible():
    registry = registered_reference_components()
    selected = registry.validate(ComponentRequest(
        {"model": "ad-blackoil-reference", "linear_solver": "scipy-direct",
         "backend": "numpy-scipy-reference"},
        phases=("water", "oil", "gas"), physics=("black-oil",),
        derivative="adi", restart=True, required_features=()))
    assert selected["model"].resolve().__name__ == "GenericBlackOilModelAD"
    assert selected["backend"].devices == ("cpu",)


def test_incompatibilities_name_every_reason_before_execution():
    registry = registered_reference_components()
    with pytest.raises(IncompatibleComponents) as caught:
        registry.validate(ComponentRequest(
            {"model": "ad-blackoil-reference"}, phases=("co2",),
            physics=("geothermal",), derivative="autograd", restart=True,
            device="cuda", precision="float32", required_features=("fractures",)))
    text = str(caught.value)
    for expected in ("phase co2", "physics geothermal", "derivative autograd",
                     "device cuda", "float32", "fractures"):
        assert expected in text


def test_requirements_conflicts_kinds_and_unknown_ids_are_rejected():
    registry = CapabilityRegistry((
        ComponentCapability("m", "model", "builtins:dict", requires={"backend": ("b",)}),
        ComponentCapability("b", "backend", "builtins:list", conflicts=("m",)),
    ))
    with pytest.raises(IncompatibleComponents) as caught:
        registry.validate(ComponentRequest(
            {"model": "m", "backend": "b", "linear_solver": "missing"}))
    assert "conflicts" in str(caught.value) and "unknown component" in str(caught.value)
    with pytest.raises(IncompatibleComponents, match="has kind"):
        registry.validate(ComponentRequest({"model": "b"}))


def test_registry_round_trip_and_duplicate_refusal():
    original = registered_reference_components()
    loaded = CapabilityRegistry.from_dict(original.to_dict())
    assert loaded.to_dict() == original.to_dict()
    with pytest.raises(KeyError, match="already registered"):
        loaded.register(loaded.get("scipy-direct"))
