"""Minimal structural protocols at the Research Studio/runtime boundary."""

from __future__ import annotations

from typing import Any, Protocol


class ScheduleSolver(Protocol):
    def simulate_schedule(self, model: Any, initial_state: Any, schedule: Any): ...


__all__ = ["ScheduleSolver"]
