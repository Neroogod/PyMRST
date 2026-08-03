"""Immutable replacement for ADFNE15 ``Globals.m`` script state."""

from dataclasses import dataclass, field
import numpy as np


def _polygons():
    return {
        "Left": np.array([[0,0,0],[0,0,1],[0,1,1],[0,1,0]], float),
        "Right": np.array([[1,0,0],[1,0,1],[1,1,1],[1,1,0]], float),
        "Top": np.array([[0,0,1],[1,0,1],[1,1,1],[0,1,1]], float),
        "Bottom": np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]], float),
        "Front": np.array([[0,0,0],[1,0,0],[1,0,1],[0,0,1]], float),
        "Back": np.array([[0,1,0],[1,1,0],[1,1,1],[0,1,1]], float),
        "X45": np.array([[0,1,0],[0,0,1],[1,0,1],[1,1,0]], float),
        "X135": np.array([[0,0,0],[1,0,0],[1,1,1],[0,1,1]], float),
        "Y45": np.array([[0,0,0],[1,0,1],[1,1,1],[0,1,0]], float),
        "Y135": np.array([[1,0,0],[0,0,1],[0,1,1],[1,1,0]], float),
        "Z45": np.array([[1,0,0],[1,0,1],[0,1,1],[0,1,0]], float),
        "Z135": np.array([[0,0,0],[1,1,0],[1,1,1],[0,0,1]], float),
    }


@dataclass(frozen=True, slots=True)
class ADFNEGlobals:
    tolerance: float = 1e-14
    segment: int = 24
    interval: float = .1
    report: bool = True
    details: bool = False
    runtime: float = 10
    silent: bool = False
    polygons: dict = field(default_factory=_polygons)
    lines: dict = field(default_factory=lambda: {
        "Left":np.array([0,0,0,1.]), "Right":np.array([1,0,1,1.]),
        "Top":np.array([0,1,1,1.]), "Bottom":np.array([0,0,1,0.]),
        "D45":np.array([0,0,1,1.]), "D135":np.array([1,0,0,1.]),
        "CH":np.array([0,.5,1,.5]), "CV":np.array([.5,0,.5,1.])})


GLOBALS = ADFNEGlobals()
