"""M5.5 bounded teaching execution; never advertised as field scale."""

from __future__ import annotations

from dataclasses import dataclass
import time


class TeachingLimitExceeded(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class TeachingLimits:
    maximum_cells: int = 5000
    maximum_report_steps: int = 100
    maximum_members: int = 8
    maximum_wall_seconds: float = 120.0
    maximum_result_bytes: int = 256*1024*1024

    def __post_init__(self):
        if min(self.maximum_cells, self.maximum_report_steps,
               self.maximum_members, self.maximum_result_bytes) <= 0:
            raise ValueError("teaching limits must be positive")
        if self.maximum_wall_seconds <= 0:
            raise ValueError("teaching wall limit must be positive")


class TeachingRuntime:
    def __init__(self, limits=TeachingLimits()):
        self.limits = limits

    def validate(self, *, cells, report_steps, members=1, estimated_result_bytes=0):
        checks = {
            "cells": (int(cells), self.limits.maximum_cells),
            "report_steps": (int(report_steps), self.limits.maximum_report_steps),
            "members": (int(members), self.limits.maximum_members),
            "estimated_result_bytes": (int(estimated_result_bytes),
                                       self.limits.maximum_result_bytes),
        }
        exceeded = [f"{name}={value} exceeds {limit}"
                    for name, (value, limit) in checks.items() if value > limit]
        if exceeded:
            raise TeachingLimitExceeded("; ".join(exceeded))
        return checks

    def run(self, function, *, cells, report_steps, members=1,
            estimated_result_bytes=0):
        self.validate(cells=cells, report_steps=report_steps, members=members,
                      estimated_result_bytes=estimated_result_bytes)
        started = time.monotonic()
        result = function()
        elapsed = time.monotonic()-started
        if elapsed > self.limits.maximum_wall_seconds:
            raise TeachingLimitExceeded(
                f"wall_seconds={elapsed:.3f} exceeds {self.limits.maximum_wall_seconds}")
        return result

    def disclosure(self):
        return {"mode": "bounded teaching runtime",
                "field_scale_claim": False,
                "limits": {name: getattr(self.limits, name)
                           for name in self.limits.__slots__}}


__all__ = ["TeachingLimitExceeded", "TeachingLimits", "TeachingRuntime"]
