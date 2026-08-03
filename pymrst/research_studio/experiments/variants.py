"""Deterministic explicit variants for ExperimentDefinition v1."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import product
import math
from typing import Any, Mapping

import numpy as np

from pymrst.research_studio.runtime.identifiers import ExperimentId, MemberId


@dataclass(frozen=True, slots=True)
class VariantDefinition:
    name: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    components: Mapping[str, str] = field(default_factory=dict)
    comparison_group: str = "default"

    def __post_init__(self):
        from .definition import canonical_value
        if not self.name or not self.comparison_group:
            raise ValueError("variant name and comparison group are required")
        parameters = canonical_value(self.parameters)
        components = canonical_value(self.components)
        if not parameters and not components:
            raise ValueError("a variant must change a parameter or component")
        if any(not path or path.startswith(".") or path.endswith(".")
               for path in parameters):
            raise ValueError("parameter keys must be explicit dotted paths")
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "components", components)

    def to_dict(self):
        return {"name": self.name, "parameters": dict(self.parameters),
                "components": dict(self.components),
                "comparison_group": self.comparison_group}

    @classmethod
    def from_dict(cls, value):
        return cls(**value)


@dataclass(frozen=True, slots=True)
class VariantCoordinate:
    experiment_id: ExperimentId
    member_id: MemberId
    ordinal: int
    variant_name: str
    comparison_group: str
    parameters: Mapping[str, Any]
    components: Mapping[str, str]

    def __post_init__(self):
        if self.ordinal < 0:
            raise ValueError("variant ordinal must be nonnegative")
        object.__setattr__(self, "experiment_id", ExperimentId(self.experiment_id))
        object.__setattr__(self, "member_id", MemberId(self.member_id))

    def to_dict(self):
        return {"experiment_id": str(self.experiment_id),
                "member_id": str(self.member_id), "ordinal": self.ordinal,
                "variant_name": self.variant_name,
                "comparison_group": self.comparison_group,
                "parameters": dict(self.parameters), "components": dict(self.components)}

    @classmethod
    def from_dict(cls, value):
        return cls(**value)


def explicit_variant_coordinates(definition) -> tuple[VariantCoordinate, ...]:
    """Expand only named explicit variants; Cartesian/sampling belongs to M3.2."""
    from .definition import content_digest
    experiment_id = definition.experiment_id
    variants = definition.variants
    if not variants:
        payloads = [("base", "default", {}, {})]
    else:
        payloads = [(item.name, item.comparison_group,
                     item.parameters, item.components) for item in variants]
    coordinates = []
    for ordinal, (name, group, parameters, components) in enumerate(payloads):
        digest = content_digest({
            "experiment_id": str(experiment_id), "ordinal": ordinal,
            "variant_name": name, "comparison_group": group,
            "parameters": parameters, "components": components,
        })
        coordinates.append(VariantCoordinate(
            experiment_id=experiment_id, member_id=MemberId(f"member_{digest}"),
            ordinal=ordinal, variant_name=name, comparison_group=group,
            parameters=parameters, components=components,
        ))
    return tuple(coordinates)


VARIANT_MATRIX_SCHEMA_VERSION = 1
SAMPLING_ALGORITHM_V1 = "pymrst_pcg64_uniform_box_muller_v1"


@dataclass(frozen=True, slots=True)
class ParameterAxis:
    path: str
    values: tuple[Any, ...]

    def __post_init__(self):
        from .definition import canonical_value
        if not self.path or not self.values:
            raise ValueError("parameter axes require a path and values")
        object.__setattr__(self, "values", tuple(canonical_value(v) for v in self.values))

    def to_dict(self):
        return {"path": self.path, "values": list(self.values)}

    @classmethod
    def from_dict(cls, value):
        return cls(path=value["path"], values=tuple(value["values"]))


@dataclass(frozen=True, slots=True)
class ComponentAxis:
    slot: str
    owners: tuple[str, ...]

    def __post_init__(self):
        if not self.slot or not self.owners or any(not owner for owner in self.owners):
            raise ValueError("component axes require a slot and owners")
        if len(set(self.owners)) != len(self.owners):
            raise ValueError("component owners must be unique")

    def to_dict(self):
        return {"slot": self.slot, "owners": list(self.owners)}

    @classmethod
    def from_dict(cls, value):
        return cls(slot=value["slot"], owners=tuple(value["owners"]))


@dataclass(frozen=True, slots=True)
class ExplicitVariantMatrix:
    variants: tuple[VariantDefinition, ...]
    schema_version: int = VARIANT_MATRIX_SCHEMA_VERSION
    matrix_type: str = "explicit"

    def __post_init__(self):
        if self.schema_version != VARIANT_MATRIX_SCHEMA_VERSION or not self.variants:
            raise ValueError("explicit matrix requires variants and schema v1")
        names = [variant.name for variant in self.variants]
        if len(set(names)) != len(names):
            raise ValueError("explicit matrix variant names must be unique")

    def expand(self):
        return self.variants

    def to_dict(self):
        return {"schema_version": self.schema_version, "matrix_type": self.matrix_type,
                "variants": [variant.to_dict() for variant in self.variants]}


@dataclass(frozen=True, slots=True)
class CartesianVariantMatrix:
    parameter_axes: tuple[ParameterAxis, ...] = ()
    component_axes: tuple[ComponentAxis, ...] = ()
    comparison_group: str = "cartesian"
    schema_version: int = VARIANT_MATRIX_SCHEMA_VERSION
    matrix_type: str = "cartesian"

    def __post_init__(self):
        if self.schema_version != VARIANT_MATRIX_SCHEMA_VERSION:
            raise ValueError("unsupported Cartesian matrix schema")
        if not self.parameter_axes and not self.component_axes:
            raise ValueError("Cartesian matrix requires at least one axis")
        names = [axis.path for axis in self.parameter_axes] + [
            axis.slot for axis in self.component_axes]
        if len(set(names)) != len(names):
            raise ValueError("Cartesian matrix axis identities must be unique")
        if not self.comparison_group:
            raise ValueError("comparison group is required")

    def expand(self):
        axes = [axis.values for axis in self.parameter_axes]
        axes += [axis.owners for axis in self.component_axes]
        variants = []
        for ordinal, coordinate in enumerate(product(*axes)):
            split = len(self.parameter_axes)
            parameters = {axis.path: coordinate[index]
                          for index, axis in enumerate(self.parameter_axes)}
            components = {axis.slot: coordinate[split + index]
                          for index, axis in enumerate(self.component_axes)}
            variants.append(VariantDefinition(
                name=f"cartesian_{ordinal:04d}", parameters=parameters,
                components=components, comparison_group=self.comparison_group))
        return tuple(variants)

    def to_dict(self):
        return {"schema_version": self.schema_version, "matrix_type": self.matrix_type,
                "parameter_axes": [axis.to_dict() for axis in self.parameter_axes],
                "component_axes": [axis.to_dict() for axis in self.component_axes],
                "comparison_group": self.comparison_group}


@dataclass(frozen=True, slots=True)
class DistributionSpec:
    kind: str
    parameters: Mapping[str, Any]

    def __post_init__(self):
        from .definition import canonical_value
        parameters = canonical_value(self.parameters)
        required = {"uniform": {"low", "high"},
                    "log_uniform": {"low", "high"},
                    "normal": {"mean", "standard_deviation"},
                    "choice": {"values"}}
        if self.kind not in required or set(parameters) != required[self.kind]:
            raise ValueError(f"invalid parameters for distribution {self.kind!r}")
        if self.kind in {"uniform", "log_uniform"}:
            low, high = float(parameters["low"]), float(parameters["high"])
            if not low < high or (self.kind == "log_uniform" and low <= 0):
                raise ValueError("distribution bounds are invalid")
        elif self.kind == "normal" and float(parameters["standard_deviation"]) <= 0:
            raise ValueError("normal standard deviation must be positive")
        elif self.kind == "choice" and not parameters["values"]:
            raise ValueError("choice distribution requires values")
        object.__setattr__(self, "parameters", parameters)

    def sample(self, generator: np.random.Generator):
        u = float(generator.random())
        if self.kind == "uniform":
            low, high = self.parameters["low"], self.parameters["high"]
            return low + (high - low) * u
        if self.kind == "log_uniform":
            low, high = math.log(self.parameters["low"]), math.log(self.parameters["high"])
            return math.exp(low + (high - low) * u)
        if self.kind == "normal":
            second = max(float(generator.random()), np.finfo(float).tiny)
            standard = math.sqrt(-2.0 * math.log(second)) * math.cos(2.0 * math.pi * u)
            return self.parameters["mean"] + self.parameters["standard_deviation"] * standard
        values = self.parameters["values"]
        return values[min(int(u * len(values)), len(values) - 1)]

    def to_dict(self):
        return {"kind": self.kind, "parameters": dict(self.parameters)}

    @classmethod
    def from_dict(cls, value):
        return cls(**value)


@dataclass(frozen=True, slots=True)
class SampledParameter:
    path: str
    distribution: DistributionSpec

    def __post_init__(self):
        if not self.path:
            raise ValueError("sampled parameter path is required")

    def to_dict(self):
        return {"path": self.path, "distribution": self.distribution.to_dict()}

    @classmethod
    def from_dict(cls, value):
        return cls(path=value["path"],
                   distribution=DistributionSpec.from_dict(value["distribution"]))


@dataclass(frozen=True, slots=True)
class SampledVariantMatrix:
    parameters: tuple[SampledParameter, ...]
    sample_count: int
    seed: int
    comparison_group: str = "sampled"
    sampling_algorithm: str = SAMPLING_ALGORITHM_V1
    schema_version: int = VARIANT_MATRIX_SCHEMA_VERSION
    matrix_type: str = "sampled"

    def __post_init__(self):
        if self.schema_version != VARIANT_MATRIX_SCHEMA_VERSION:
            raise ValueError("unsupported sampled matrix schema")
        if not self.parameters or self.sample_count <= 0 or self.seed < 0:
            raise ValueError("sampled matrix requires parameters, samples, and nonnegative seed")
        if self.sampling_algorithm != SAMPLING_ALGORITHM_V1:
            raise ValueError("unsupported sampling algorithm")
        paths = [parameter.path for parameter in self.parameters]
        if len(set(paths)) != len(paths):
            raise ValueError("sampled parameter paths must be unique")

    def expand(self):
        generator = np.random.Generator(np.random.PCG64(self.seed))
        return tuple(VariantDefinition(
            name=f"sample_{index:04d}",
            parameters={parameter.path: parameter.distribution.sample(generator)
                        for parameter in self.parameters},
            comparison_group=self.comparison_group,
        ) for index in range(self.sample_count))

    def to_dict(self):
        return {"schema_version": self.schema_version, "matrix_type": self.matrix_type,
                "parameters": [parameter.to_dict() for parameter in self.parameters],
                "sample_count": self.sample_count, "seed": self.seed,
                "comparison_group": self.comparison_group,
                "sampling_algorithm": self.sampling_algorithm}


def variant_matrix_from_dict(value):
    kind = value.get("matrix_type")
    if kind == "explicit":
        return ExplicitVariantMatrix(
            variants=tuple(VariantDefinition.from_dict(v) for v in value["variants"]),
            schema_version=value["schema_version"])
    if kind == "cartesian":
        return CartesianVariantMatrix(
            parameter_axes=tuple(ParameterAxis.from_dict(v)
                                 for v in value["parameter_axes"]),
            component_axes=tuple(ComponentAxis.from_dict(v)
                                 for v in value["component_axes"]),
            comparison_group=value["comparison_group"],
            schema_version=value["schema_version"])
    if kind == "sampled":
        return SampledVariantMatrix(
            parameters=tuple(SampledParameter.from_dict(v) for v in value["parameters"]),
            sample_count=value["sample_count"], seed=value["seed"],
            comparison_group=value["comparison_group"],
            sampling_algorithm=value["sampling_algorithm"],
            schema_version=value["schema_version"])
    raise ValueError(f"unsupported variant matrix type: {kind!r}")


@dataclass(frozen=True, slots=True)
class MaterializedVariantMatrix:
    matrix_id: str
    base_experiment_id: ExperimentId
    base_execution_fingerprint: str
    matrix: Any
    definition: Any
    coordinates: tuple[VariantCoordinate, ...]

    def to_dict(self):
        return {"matrix_id": self.matrix_id,
                "base_experiment_id": str(self.base_experiment_id),
                "base_execution_fingerprint": self.base_execution_fingerprint,
                "matrix": self.matrix.to_dict(),
                "definition": self.definition.to_dict(),
                "coordinates": [coordinate.to_dict() for coordinate in self.coordinates]}

    @classmethod
    def from_dict(cls, value):
        from .definition import ExperimentDefinition, content_digest
        if set(value) != {"matrix_id", "base_experiment_id",
                          "base_execution_fingerprint", "matrix", "definition",
                          "coordinates"}:
            raise ValueError("invalid materialized variant matrix fields")
        matrix = variant_matrix_from_dict(value["matrix"])
        definition = ExperimentDefinition.from_dict(value["definition"])
        coordinates = tuple(VariantCoordinate.from_dict(item)
                            for item in value["coordinates"])
        expected_id = f"matrix_{content_digest({'base_execution_fingerprint': value['base_execution_fingerprint'], 'matrix': matrix.to_dict()})}"
        if value["matrix_id"] != expected_id:
            raise ValueError("materialized variant matrix identity mismatch")
        if coordinates != explicit_variant_coordinates(definition):
            raise ValueError("materialized variant coordinates are inconsistent")
        return cls(value["matrix_id"], ExperimentId(value["base_experiment_id"]),
                   value["base_execution_fingerprint"], matrix, definition, coordinates)


def materialize_variant_matrix(definition, matrix) -> MaterializedVariantMatrix:
    from .definition import content_digest
    if definition.variants:
        if not (isinstance(matrix, ExplicitVariantMatrix)
                and matrix.variants == definition.variants):
            raise ValueError("materializing a matrix requires a definition without variants")
        expanded = definition
    else:
        expanded = replace(definition, variants=tuple(matrix.expand()))
    coordinates = explicit_variant_coordinates(expanded)
    matrix_id = f"matrix_{content_digest({'base_execution_fingerprint': definition.execution_fingerprint, 'matrix': matrix.to_dict()})}"
    return MaterializedVariantMatrix(
        matrix_id, definition.experiment_id, definition.execution_fingerprint,
        matrix, expanded, coordinates)


__all__ = ["VARIANT_MATRIX_SCHEMA_VERSION", "SAMPLING_ALGORITHM_V1",
           "CartesianVariantMatrix", "ComponentAxis", "DistributionSpec",
           "ExplicitVariantMatrix", "MaterializedVariantMatrix", "ParameterAxis",
           "SampledParameter", "SampledVariantMatrix", "VariantCoordinate",
           "VariantDefinition", "explicit_variant_coordinates",
           "materialize_variant_matrix", "variant_matrix_from_dict"]
