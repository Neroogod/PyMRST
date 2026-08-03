"""INIT static properties and UNRST solution arrays (M2.6).

INIT holds what does not change - porosity, permeability, pore volume,
transmissibility, cell depths. UNRST holds what does: pressure and saturations
at each report step. Together with EGRID they are what a case another
simulator ran actually consists of.

**UNRST repeats its keywords per report step**, which is the thing to get
right. `PRESSURE` appears once per step, separated by `SEQNUM`, so a reader
that builds a keyword mapping keeps the first step and silently discards the
rest of the simulation. The step boundary is `SEQNUM`, and everything here is
built on the record *sequence* rather than on a mapping.

Units are not converted. An ECLIPSE file is written in whatever unit system
its RUNSPEC declared, and INIT does not carry that declaration - guessing
would silently rescale a field. The values are returned as stored, and the
unit system is the caller's to supply from the deck.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pymrst.research_studio.eclipse.records import read_records


@dataclass(frozen=True, slots=True)
class InitFile:
    """Static properties, as stored."""

    arrays: dict[str, np.ndarray]
    cells: int

    def get(self, name: str) -> np.ndarray | None:
        return self.arrays.get(name.upper())

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self.arrays))


@dataclass(frozen=True, slots=True)
class RestartStep:
    """One report step of a UNRST file."""

    sequence: int
    arrays: dict[str, np.ndarray] = field(default_factory=dict)

    def get(self, name: str) -> np.ndarray | None:
        return self.arrays.get(name.upper())


def read_init(path: str | Path) -> InitFile:
    """Static arrays from an INIT file.

    Cell count comes from `INTEHEAD` when present. It is reported rather than
    used to reshape anything, because an INIT may carry arrays over active
    cells and others over the full grid, and silently reconciling the two
    would misplace values.
    """
    arrays: dict[str, np.ndarray] = {}
    cells = 0
    for name, value_type, values in read_records(path):
        if value_type == "MESS":
            continue
        key = name.upper()
        if key == "INTEHEAD":
            head = np.asarray(values, dtype=np.int64).ravel()
            # INTEHEAD[1] is NX*NY*NZ in the layouts this reads; absent or
            # implausible, the count stays zero and the caller is not misled.
            if head.size > 1 and head[1] > 0:
                cells = int(head[1])
        arrays.setdefault(key, np.asarray(values))
    if not arrays:
        raise ValueError(f"{path} contains no records; is it an INIT file?")
    return InitFile(arrays, cells)


def read_restart(path: str | Path) -> list[RestartStep]:
    """Every report step in a UNRST file, in file order.

    Split on `SEQNUM`. A file with no `SEQNUM` at all is treated as a single
    step, which is what a non-unified restart looks like - but a file with
    *some* steps and stray leading records is not silently merged, because
    that would attribute one step's arrays to another.
    """
    steps: list[RestartStep] = []
    current: dict[str, np.ndarray] | None = None
    sequence = 0

    for name, value_type, values in read_records(path):
        if value_type == "MESS":
            continue
        key = name.upper()
        if key == "SEQNUM":
            if current is not None:
                steps.append(RestartStep(sequence, current))
            sequence = int(np.asarray(values, dtype=np.int64).ravel()[0])
            current = {}
            continue
        if current is None:
            current = {}
        current[key] = np.asarray(values)

    if current:
        steps.append(RestartStep(sequence, current))
    if not steps:
        raise ValueError(f"{path} contains no restart records")
    return steps


def restart_series(path: str | Path, name: str) -> np.ndarray:
    """One keyword across every report step, as a `(steps, cells)` array.

    Refuses a ragged result rather than padding it: arrays of different
    lengths across steps mean the file mixes grids or the keyword is not a
    cell array, and either way stacking would invent values.
    """
    steps = read_restart(path)
    key = name.upper()
    missing = [s.sequence for s in steps if key not in s.arrays]
    if missing:
        raise KeyError(f"{name} is absent from report step(s) {missing}")
    widths = {s.arrays[key].size for s in steps}
    if len(widths) != 1:
        raise ValueError(
            f"{name} has differing lengths across report steps ({sorted(widths)}); "
            f"refusing to stack them"
        )
    return np.vstack([np.asarray(s.arrays[key]).ravel() for s in steps])


__all__ = ["InitFile", "RestartStep", "read_init", "read_restart",
           "restart_series"]
