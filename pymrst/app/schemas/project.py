"""Project and immutable input-asset schemas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ._common import nonempty, required


PROJECT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ProjectCreate:
    name: str
    description: str = ""
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", nonempty(self.name, "project name"))
        object.__setattr__(self, "description", str(self.description).strip())
        tags = tuple(sorted(set(nonempty(item, "project tag") for item in self.tags)))
        object.__setattr__(self, "tags", tags)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "tags": list(self.tags)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProjectCreate":
        data = required(value, {"name", "description", "tags"}, "ProjectCreate")
        data["tags"] = tuple(data["tags"])
        return cls(**data)


@dataclass(frozen=True, slots=True)
class ProjectRecord:
    project_id: str
    name: str
    description: str
    tags: tuple[str, ...]
    root_path: str
    content_hash: str
    created_at: str
    updated_at: str
    schema_version: int = PROJECT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PROJECT_SCHEMA_VERSION:
            raise ValueError("unsupported project schema version")
        for name in ("project_id", "name", "root_path", "created_at", "updated_at"):
            object.__setattr__(self, name, nonempty(getattr(self, name), name))
        digest = str(self.content_hash).lower()
        if len(digest) != 64 or any(item not in "0123456789abcdef" for item in digest):
            raise ValueError("project content_hash must be SHA-256")
        object.__setattr__(self, "content_hash", digest)
        object.__setattr__(self, "description", str(self.description))
        object.__setattr__(self, "tags", tuple(self.tags))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version, "project_id": self.project_id,
            "name": self.name, "description": self.description,
            "tags": list(self.tags), "root_path": self.root_path,
            "content_hash": self.content_hash, "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProjectRecord":
        data = required(value, {
            "schema_version", "project_id", "name", "description", "tags",
            "root_path", "content_hash", "created_at", "updated_at",
        }, "ProjectRecord")
        data["tags"] = tuple(data["tags"])
        return cls(**data)


@dataclass(frozen=True, slots=True)
class ProjectAsset:
    asset_id: str
    project_id: str
    kind: str
    original_name: str
    relative_path: str
    content_hash: str
    size_bytes: int
    created_at: str

    def __post_init__(self) -> None:
        for name in ("asset_id", "project_id", "kind", "original_name", "relative_path", "created_at"):
            object.__setattr__(self, name, nonempty(getattr(self, name), name))
        digest = str(self.content_hash).lower()
        if len(digest) != 64 or any(item not in "0123456789abcdef" for item in digest):
            raise ValueError("asset content_hash must be SHA-256")
        object.__setattr__(self, "content_hash", digest)
        size = int(self.size_bytes)
        if size < 0:
            raise ValueError("asset size cannot be negative")
        object.__setattr__(self, "size_bytes", size)

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__slots__}


__all__ = [
    "PROJECT_SCHEMA_VERSION", "ProjectAsset", "ProjectCreate", "ProjectRecord",
]
