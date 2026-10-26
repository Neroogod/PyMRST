"""Contract for the curated, run-in-app PyMRST example catalog."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping

from ._common import required


class ExampleRuntimeClass(StrEnum):
    FAST = "fast"
    MEDIUM = "medium"
    SLOW = "slow"


@dataclass(frozen=True, slots=True)
class ExampleDefinition:
    example_id: str
    title: str
    description: str
    category: str
    module: str
    function: str
    runtime_class: ExampleRuntimeClass

    def __post_init__(self) -> None:
        for name in ("example_id", "title", "description", "category", "module", "function"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} cannot be empty")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "runtime_class", ExampleRuntimeClass(self.runtime_class))

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id, "title": self.title,
            "description": self.description, "category": self.category,
            "runtime_class": self.runtime_class.value,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExampleDefinition":
        data = required(
            value,
            {"example_id", "title", "description", "category", "module", "function", "runtime_class"},
            "ExampleDefinition",
        )
        return cls(**data)


__all__ = ["ExampleDefinition", "ExampleRuntimeClass"]
