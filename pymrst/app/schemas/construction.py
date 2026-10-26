"""Typed SI-unit contracts for guided reservoir-model construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping

from ._common import json_value, nonempty


class FieldSourceKind(StrEnum):
    UNIFORM = "uniform"
    GRID_ARRAY = "grid_array"


@dataclass(frozen=True, slots=True)
class FieldSource:
    kind: FieldSourceKind
    value: float | None = None
    array: str = ""
    scale: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, "kind", FieldSourceKind(self.kind))
        if self.kind is FieldSourceKind.UNIFORM and self.value is None:
            raise ValueError("uniform field source requires a value")
        if self.kind is FieldSourceKind.GRID_ARRAY:
            object.__setattr__(self, "array", nonempty(self.array, "grid array").upper())
        if not float(self.scale) or not __import__("math").isfinite(float(self.scale)):
            raise ValueError("field source scale must be finite and nonzero")

    def to_dict(self):
        return {"kind": self.kind.value, "value": self.value, "array": self.array, "scale": self.scale}

    @classmethod
    def from_dict(cls, value): return cls(**value)


@dataclass(frozen=True, slots=True)
class RockDefinition:
    porosity: FieldSource
    permeability_x: FieldSource
    permeability_y: FieldSource | None = None
    permeability_z: FieldSource | None = None
    net_to_gross: FieldSource | None = None

    def to_dict(self):
        return {name: None if getattr(self, name) is None else getattr(self, name).to_dict()
                for name in self.__slots__}

    @classmethod
    def from_dict(cls, value):
        return cls(**{name: None if item is None else FieldSource.from_dict(item)
                      for name, item in value.items()})


@dataclass(frozen=True, slots=True)
class FluidDefinition:
    phases: tuple[str, ...] = ("W", "O")
    viscosity_pa_s: tuple[float, ...] = (1e-3, 5e-3)
    density_kg_m3: tuple[float, ...] = (1000.0, 800.0)
    relperm_exponent: tuple[float, ...] = (2.0, 2.0)
    compressibility_pa_inverse: tuple[float, ...] = (0.0, 0.0)
    reference_pressure_pa: float = 10_000_000.0

    def __post_init__(self):
        phases = tuple(str(item).upper() for item in self.phases)
        if not phases or len(set(phases)) != len(phases) or any(item not in {"W", "O", "G"} for item in phases):
            raise ValueError("fluid phases must be unique W/O/G identifiers")
        object.__setattr__(self, "phases", phases)
        for name in ("viscosity_pa_s", "density_kg_m3", "relperm_exponent", "compressibility_pa_inverse"):
            values = tuple(float(item) for item in getattr(self, name))
            if len(values) != len(phases):
                raise ValueError(f"{name} must contain one value per phase")
            object.__setattr__(self, name, values)

    def to_dict(self):
        return {name: list(getattr(self, name)) if isinstance(getattr(self, name), tuple)
                else getattr(self, name) for name in self.__slots__}

    @classmethod
    def from_dict(cls, value): return cls(**value)


@dataclass(frozen=True, slots=True)
class InitialStateDefinition:
    pressure: FieldSource
    saturations: Mapping[str, FieldSource]
    dissolved_gas_ratio: FieldSource | None = None

    def __post_init__(self):
        normalized = {str(name).upper(): item for name, item in self.saturations.items()}
        if len(normalized) != len(self.saturations):
            raise ValueError("initial saturation phase names must be unique")
        object.__setattr__(self, "saturations", normalized)

    def to_dict(self):
        return {"pressure": self.pressure.to_dict(),
                "saturations": {name: item.to_dict() for name, item in self.saturations.items()},
                "dissolved_gas_ratio": None if self.dissolved_gas_ratio is None else self.dissolved_gas_ratio.to_dict()}

    @classmethod
    def from_dict(cls, value):
        return cls(FieldSource.from_dict(value["pressure"]),
                   {name: FieldSource.from_dict(item) for name, item in value["saturations"].items()},
                   None if value.get("dissolved_gas_ratio") is None else FieldSource.from_dict(value["dissolved_gas_ratio"]))


@dataclass(frozen=True, slots=True)
class WellDefinition:
    name: str
    cells: tuple[int, ...]
    control_type: str
    target_si: float
    radius_m: float = 0.1
    direction: str = "z"
    composition: tuple[float, ...] = (1.0, 0.0)
    sign: int = 0
    status: bool = True
    limits: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "name", nonempty(self.name, "well name"))
        object.__setattr__(self, "cells", tuple(int(item) for item in self.cells))
        object.__setattr__(self, "control_type", str(self.control_type).lower())
        object.__setattr__(self, "direction", str(self.direction).lower())
        object.__setattr__(self, "composition", tuple(float(item) for item in self.composition))
        object.__setattr__(self, "limits", json_value(dict(self.limits)))
        object.__setattr__(self, "target_si", float(self.target_si))
        object.__setattr__(self, "radius_m", float(self.radius_m))
        object.__setattr__(self, "sign", int(self.sign))

    def to_dict(self):
        return {"name": self.name, "cells": list(self.cells), "control_type": self.control_type,
                "target_si": self.target_si, "radius_m": self.radius_m, "direction": self.direction,
                "composition": list(self.composition), "sign": self.sign, "status": self.status,
                "limits": dict(self.limits)}

    @classmethod
    def from_dict(cls, value): return cls(**value)


@dataclass(frozen=True, slots=True)
class ControlPeriod:
    name: str
    well_overrides: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "name", nonempty(self.name, "control-period name"))
        object.__setattr__(self, "well_overrides", json_value(self.well_overrides))

    def to_dict(self): return {"name": self.name, "well_overrides": json_value(self.well_overrides)}
    @classmethod
    def from_dict(cls, value): return cls(**value)


@dataclass(frozen=True, slots=True)
class ScheduleDefinition:
    timestep_seconds: tuple[float, ...]
    step_control: tuple[int, ...]
    controls: tuple[ControlPeriod, ...]

    def __post_init__(self):
        object.__setattr__(self, "timestep_seconds", tuple(float(item) for item in self.timestep_seconds))
        object.__setattr__(self, "step_control", tuple(int(item) for item in self.step_control))
        object.__setattr__(self, "controls", tuple(self.controls))

    def to_dict(self):
        return {"timestep_seconds": list(self.timestep_seconds), "step_control": list(self.step_control),
                "controls": [item.to_dict() for item in self.controls]}

    @classmethod
    def from_dict(cls, value):
        return cls(value["timestep_seconds"], value["step_control"],
                   tuple(ControlPeriod.from_dict(item) for item in value["controls"]))


@dataclass(frozen=True, slots=True)
class ModelConstructionDocument:
    rock: RockDefinition
    fluid: FluidDefinition
    initial_state: InitialStateDefinition
    wells: tuple[WellDefinition, ...]
    schedule: ScheduleDefinition
    schema_version: int = 1

    def to_dict(self):
        return {"schema_version": self.schema_version, "rock": self.rock.to_dict(),
                "fluid": self.fluid.to_dict(), "initial_state": self.initial_state.to_dict(),
                "wells": [item.to_dict() for item in self.wells], "schedule": self.schedule.to_dict()}

    @classmethod
    def from_dict(cls, value):
        if int(value.get("schema_version", 0)) != 1: raise ValueError("unsupported construction schema")
        return cls(RockDefinition.from_dict(value["rock"]), FluidDefinition.from_dict(value["fluid"]),
                   InitialStateDefinition.from_dict(value["initial_state"]),
                   tuple(WellDefinition.from_dict(item) for item in value["wells"]),
                   ScheduleDefinition.from_dict(value["schedule"]))


__all__ = ["ControlPeriod", "FieldSource", "FieldSourceKind", "FluidDefinition",
           "InitialStateDefinition", "ModelConstructionDocument", "RockDefinition",
           "ScheduleDefinition", "WellDefinition"]


@dataclass(frozen=True, slots=True)
class ModelDraft:
    draft_id: str; project_id: str; grid_snapshot_id: str; name: str
    version: int; status: str; document_hash: str
    document: ModelConstructionDocument; issues: tuple[Mapping[str, Any], ...]
    created_at: str; updated_at: str
    def to_dict(self):
        return {"draft_id": self.draft_id, "project_id": self.project_id,
                "grid_snapshot_id": self.grid_snapshot_id, "name": self.name,
                "version": self.version, "status": self.status,
                "document_hash": self.document_hash, "document": self.document.to_dict(),
                "issues": [dict(item) for item in self.issues],
                "created_at": self.created_at, "updated_at": self.updated_at}


@dataclass(frozen=True, slots=True)
class ModelRevision:
    revision_id: str; draft_id: str; project_id: str; revision_number: int
    content_hash: str; relative_path: str; summary: Mapping[str, Any]; created_at: str
    def to_dict(self): return {name: getattr(self, name) for name in self.__slots__}


__all__ += ["ModelDraft", "ModelRevision"]
