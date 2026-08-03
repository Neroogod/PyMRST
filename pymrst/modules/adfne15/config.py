"""ADFNE 1.5 numerical defaults without MATLAB global state."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ADFNEConfig:
    tolerance: float = 1e-14
    polygon_segments: int = 24
    interval: float = 0.1
    report: bool = False


DEFAULT_CONFIG = ADFNEConfig()
