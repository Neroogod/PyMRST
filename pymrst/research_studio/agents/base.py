"""Provider and tool boundaries; no concrete AI dependency is imported here."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, Mapping, Protocol, Sequence, TypeVar

from ._serialization import frozen_json_mapping, require_fields
from .permissions import AgentPermission


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class ToolCall:
    call_id: str
    tool: str
    arguments: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("call_id", "tool"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} cannot be empty")
            object.__setattr__(self, name, value)
        object.__setattr__(
            self, "arguments", frozen_json_mapping(self.arguments, "$.arguments")
        )

    def to_dict(self) -> dict[str, Any]:
        from ._serialization import json_value
        return json_value({
            "call_id": self.call_id, "tool": self.tool,
            "arguments": self.arguments,
        })

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ToolCall":
        return cls(**require_fields(
            value, required={"call_id", "tool", "arguments"}, name="ToolCall",
        ))


@dataclass(frozen=True, slots=True)
class ToolResult:
    call_id: str
    ok: bool
    value: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None

    def __post_init__(self) -> None:
        call_id = str(self.call_id).strip()
        if not call_id:
            raise ValueError("call_id cannot be empty")
        object.__setattr__(self, "call_id", call_id)
        object.__setattr__(
            self, "value", frozen_json_mapping(self.value, "$.value")
        )
        error = None if self.error is None else str(self.error).strip()
        if self.ok and error:
            raise ValueError("successful tool results cannot contain an error")
        if not self.ok and not error:
            raise ValueError("failed tool results require an error")
        object.__setattr__(self, "error", error)

    def to_dict(self) -> dict[str, Any]:
        from ._serialization import json_value
        return json_value({
            "call_id": self.call_id, "ok": self.ok,
            "value": self.value, "error": self.error,
        })

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ToolResult":
        return cls(**require_fields(
            value, required={"call_id", "ok", "value", "error"}, name="ToolResult",
        ))


@dataclass(frozen=True, slots=True)
class ToolDescriptor:
    """Static authority declaration used before a tool can be invoked."""

    name: str
    description: str
    permission: AgentPermission
    mutates_case: bool = False
    starts_workload: bool = False
    requires_approval: bool = False

    def __post_init__(self) -> None:
        for name in ("name", "description"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} cannot be empty")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "permission", AgentPermission(self.permission))
        if self.mutates_case and self.permission < AgentPermission.CREATE_VARIANT:
            raise ValueError("case-mutating tools require CREATE_VARIANT permission")
        if self.starts_workload and self.permission < AgentPermission.RUN_SIMULATION:
            raise ValueError("workload tools require RUN_SIMULATION permission")


class AgentTool(Protocol):
    descriptor: ToolDescriptor

    def invoke(self, call: ToolCall) -> ToolResult: ...


class ReasoningProvider(Protocol, Generic[T]):
    """Minimal provider interface used by orchestration, not simulation."""

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        response_type: type[T],
    ) -> T: ...


__all__ = [
    "AgentTool", "ReasoningProvider", "ToolCall", "ToolDescriptor", "ToolResult",
]
