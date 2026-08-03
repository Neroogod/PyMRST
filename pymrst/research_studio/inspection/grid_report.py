"""Ask a grid what is wrong with it (M2.11).

A user whose field model produces a strange answer currently has no way to
interrogate the grid it ran on. PyMRST builds Cartesian, corner-point, faulted
and PEBI grids and computes the geometry a discretization needs, but nothing
reports whether that geometry is *sane*.

Two rules shape this:

* **Read-only.** These functions interrogate a grid; they never normalize,
  repair or annotate one. A validator that quietly fixes its input has changed
  the model, which the covenant forbids however convenient it would be.
* **Severity is part of the answer.** A negative cell volume means the
  discretization cannot be trusted. A two-cell disconnected region is legal
  and probably not intended. Reporting both at one level makes the first
  invisible among the second.

Indices are the grid's own cell and face numbering, so anything reported here
can be located in the plotting and VTK-export layers without translation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Sequence

import numpy as np


class Severity(StrEnum):
    ERROR = "error"        # the discretization cannot be trusted
    WARNING = "warning"    # legal, and probably not intended
    INFO = "info"          # worth knowing, not a problem


@dataclass(frozen=True, slots=True)
class Finding:
    severity: Severity
    check: str
    message: str
    cells: tuple[int, ...] = ()
    faces: tuple[int, ...] = ()

    def __str__(self) -> str:
        where = ""
        if self.cells:
            where = f"  cells {list(self.cells[:6])}{'...' if len(self.cells) > 6 else ''}"
        elif self.faces:
            where = f"  faces {list(self.faces[:6])}{'...' if len(self.faces) > 6 else ''}"
        return f"{self.severity.value.upper():<8} {self.message}{where}"


@dataclass(frozen=True, slots=True)
class GridReport:
    findings: tuple[Finding, ...] = ()

    @property
    def errors(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity is Severity.ERROR)

    @property
    def warnings(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity is Severity.WARNING)

    @property
    def ok(self) -> bool:
        """No errors. Warnings do not make a grid unusable."""
        return not self.errors

    def __bool__(self) -> bool:
        return self.ok

    def __str__(self) -> str:
        if not self.findings:
            return "no findings"
        return "\n".join(str(f) for f in self.findings)


def validate(grid: Any, rock: Any = None, wells: Sequence[Any] = ()) -> GridReport:
    """Structured findings about `grid`, worst first.

    Returns a report rather than a boolean because "is this grid valid" is not
    a yes/no question: a grid can be perfectly legal and still contain the
    thing that explains a strange answer.
    """
    findings: list[Finding] = []
    findings += _volume_findings(grid)
    findings += _geometry_findings(grid)
    findings += _twist_findings(grid)
    findings += _connectivity_findings(grid)
    findings += _fault_findings(grid)
    if rock is not None:
        findings += _rock_findings(grid, rock)
        findings += _transmissibility_findings(grid, rock)
    if wells:
        findings += _well_findings(grid, wells)
    order = {Severity.ERROR: 0, Severity.WARNING: 1, Severity.INFO: 2}
    findings.sort(key=lambda f: order[f.severity])
    return GridReport(tuple(findings))


def _volume_findings(grid) -> list[Finding]:
    volumes = getattr(grid.cells, "volumes", None)
    if volumes is None:
        return [Finding(Severity.WARNING, "volumes",
                        "grid has no cell volumes; run compute_geometry")]
    volumes = np.asarray(volumes, dtype=float)
    out = []
    negative = np.flatnonzero(volumes <= 0.0)
    if negative.size:
        out.append(Finding(
            Severity.ERROR, "volumes",
            f"{negative.size} cell(s) with non-positive volume; the "
            f"discretization divides by these",
            cells=tuple(int(c) for c in negative)))
    positive = volumes[volumes > 0.0]
    if positive.size:
        # Relative to the grid's own scale: an absolute threshold is
        # meaningless across models spanning millimetres to kilometres.
        tiny = np.flatnonzero((volumes > 0.0) & (volumes < np.median(positive)*1e-6))
        if tiny.size:
            out.append(Finding(
                Severity.WARNING, "volumes",
                f"{tiny.size} cell(s) smaller than 1e-6 of the median volume; "
                f"these dominate the timestep restriction",
                cells=tuple(int(c) for c in tiny)))
    return out


def _geometry_findings(grid) -> list[Finding]:
    out = []
    areas = getattr(grid.faces, "areas", None)
    if areas is not None:
        areas = np.asarray(areas, dtype=float)
        degenerate = np.flatnonzero(areas <= 0.0)
        if degenerate.size:
            out.append(Finding(
                Severity.ERROR, "faces",
                f"{degenerate.size} face(s) with non-positive area; "
                f"transmissibility through these is undefined",
                faces=tuple(int(f) for f in degenerate)))
    out += _orientation_findings(grid)
    return out


def _twist_findings(grid) -> list[Finding]:
    """Deferred import: `geometry` imports `Finding` and `Severity` from here,
    on the same pattern as `transmissibility`."""
    from pymrst.research_studio.inspection.geometry import twist_findings
    return twist_findings(grid)


def _orientation_findings(grid) -> list[Finding]:
    from pymrst.research_studio.inspection.geometry import orientation_findings
    return orientation_findings(grid)


def _connectivity_findings(grid) -> list[Finding]:
    neighbors = np.asarray(grid.faces.neighbors, dtype=np.int64)
    interior = np.all(neighbors >= 0, axis=1)
    out = []

    self_connected = np.flatnonzero(interior & (neighbors[:, 0] == neighbors[:, 1]))
    if self_connected.size:
        out.append(Finding(
            Severity.ERROR, "connectivity",
            f"{self_connected.size} face(s) connect a cell to itself",
            faces=tuple(int(f) for f in self_connected)))

    regions = _connected_regions(grid.cells.num, neighbors[interior])
    if len(regions) > 1:
        sizes = sorted((len(r) for r in regions), reverse=True)
        out.append(Finding(
            Severity.WARNING, "connectivity",
            f"{len(regions)} disconnected active regions, sizes {sizes[:6]}"
            f"{'...' if len(sizes) > 6 else ''}; flow cannot cross between them",
            cells=tuple(sorted(min(r) for r in regions))))
    isolated = [next(iter(r)) for r in regions if len(r) == 1]
    if isolated:
        out.append(Finding(
            Severity.WARNING, "connectivity",
            f"{len(isolated)} cell(s) with no interior connection at all",
            cells=tuple(int(c) for c in isolated)))
    return out


def _connected_regions(count: int, pairs: np.ndarray) -> list[set[int]]:
    """Union-find over interior connections; iterative, so a long thin grid
    cannot blow the recursion limit."""
    parent = list(range(count))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[ra] = rb

    groups: dict[int, set[int]] = {}
    for cell in range(count):
        groups.setdefault(find(cell), set()).add(cell)
    return list(groups.values())


def _fault_findings(grid) -> list[Finding]:
    """Delegated. Imported here rather than at module scope because
    `transmissibility` imports `Finding` and `Severity` from this file."""
    from pymrst.research_studio.inspection.transmissibility import fault_findings
    return fault_findings(grid)


def _transmissibility_findings(grid, rock) -> list[Finding]:
    from pymrst.research_studio.inspection.transmissibility import (
        transmissibility_findings,
    )
    try:
        return transmissibility_findings(grid, rock)
    except Exception:
        # Transmissibility needs geometry and a permeability tensor. A grid
        # that cannot supply them is a case this check does not apply to, not
        # a reason for `validate` to fail - the caller came here to find out
        # what is wrong, and raising would answer nothing.
        return []


def _rock_findings(grid, rock) -> list[Finding]:
    out = []
    poro = np.asarray(getattr(rock, "poro", []), dtype=float).reshape(-1)
    if poro.size:
        bad = np.flatnonzero((poro < 0.0) | (poro > 1.0))
        if bad.size:
            out.append(Finding(
                Severity.ERROR, "rock",
                f"{bad.size} cell(s) with porosity outside [0, 1]",
                cells=tuple(int(c) for c in bad)))
        volumes = np.asarray(getattr(grid.cells, "volumes", []), dtype=float)
        if volumes.size == poro.size:
            pore = volumes*poro
            live = pore[pore > 0.0]
            if live.size:
                tiny = np.flatnonzero((pore >= 0.0) & (pore < np.median(live)*1e-8))
                if tiny.size:
                    out.append(Finding(
                        Severity.WARNING, "rock",
                        f"{tiny.size} cell(s) with pore volume below 1e-8 of "
                        f"the median; these behave as numerical obstructions",
                        cells=tuple(int(c) for c in tiny)))
    perm = np.asarray(getattr(rock, "perm", []), dtype=float)
    if perm.size and np.any(perm < 0.0):
        rows = np.unique(np.argwhere(perm < 0.0)[:, 0])
        out.append(Finding(
            Severity.ERROR, "rock",
            f"{rows.size} cell(s) with negative permeability",
            cells=tuple(int(c) for c in rows)))
    return out


def _well_findings(grid, wells) -> list[Finding]:
    out = []
    volumes = np.asarray(getattr(grid.cells, "volumes", []), dtype=float)
    for index, well in enumerate(wells):
        cells = np.asarray(getattr(well, "cells", ()), dtype=np.int64).ravel()
        name = getattr(well, "name", None) or f"well {index}"
        if cells.size == 0:
            out.append(Finding(Severity.WARNING, "wells",
                               f"{name} perforates no cells"))
            continue
        outside = cells[(cells < 0) | (cells >= grid.cells.num)]
        if outside.size:
            out.append(Finding(
                Severity.ERROR, "wells",
                f"{name} perforates {outside.size} cell(s) outside the grid",
                cells=tuple(int(c) for c in outside)))
        if volumes.size:
            inside = cells[(cells >= 0) & (cells < volumes.size)]
            dead = inside[volumes[inside] <= 0.0]
            if dead.size:
                out.append(Finding(
                    Severity.ERROR, "wells",
                    f"{name} perforates {dead.size} cell(s) with no volume",
                    cells=tuple(int(c) for c in dead)))
    return out


__all__ = ["Finding", "GridReport", "Severity", "validate"]
