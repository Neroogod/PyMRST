"""Constrained deterministic tools exposed to engineering agents."""

from __future__ import annotations

from typing import Iterable

from ..base import AgentTool


class ToolRegistry:
    """Explicit catalog; importing a module never auto-registers a tool."""

    def __init__(self, tools: Iterable[AgentTool] = ()) -> None:
        self._tools: dict[str, AgentTool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: AgentTool) -> None:
        name = tool.descriptor.name
        if name in self._tools:
            raise KeyError(f"agent tool already registered: {name}")
        self._tools[name] = tool

    def get(self, name: str) -> AgentTool:
        try:
            return self._tools[name]
        except KeyError as error:
            raise KeyError(f"unknown agent tool: {name}") from error

    def tools(self) -> tuple[AgentTool, ...]:
        return tuple(self._tools[name] for name in sorted(self._tools))

    def descriptors(self):
        return tuple(tool.descriptor for tool in self.tools())


__all__ = ["ToolRegistry"]
