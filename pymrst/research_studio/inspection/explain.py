"""Why are these two cells connected, and what does this cell look like (M2.11).

`validate` answers "what is wrong with this grid". These answer the question a
user actually arrives with: *this* number looks wrong, what is cell 4219 and
how does it talk to 4220.

Read-only, like everything in this package. Nothing here mutates a grid, a
rock or a state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class Connection:
    """How two cells are joined, or the fact that they are not."""

    cells: tuple[int, int]
    connected: bool
    faces: tuple[int, ...] = ()
    areas: tuple[float, ...] = ()
    transmissibilities: tuple[float, ...] = ()
    reason: str = ""

    def __str__(self) -> str:
        a, b = self.cells
        if not self.connected:
            return f"cells {a} and {b}: not connected - {self.reason}"
        lines = [f"cells {a} and {b}: connected through "
                 f"{len(self.faces)} face(s)"]
        for index, face in enumerate(self.faces):
            area = self.areas[index] if index < len(self.areas) else float("nan")
            entry = f"  face {face}: area {area:.4g}"
            if index < len(self.transmissibilities):
                entry += f", transmissibility {self.transmissibilities[index]:.4g}"
            lines.append(entry)
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class CellReport:
    index: int
    centroid: tuple[float, ...]
    volume: float
    neighbours: tuple[int, ...]
    faces: tuple[int, ...]
    boundary_faces: tuple[int, ...]
    properties: dict

    def __str__(self) -> str:
        centre = ", ".join(f"{c:.4g}" for c in self.centroid)
        lines = [
            f"cell {self.index}",
            f"  centroid    ({centre})",
            f"  volume      {self.volume:.6g}",
            f"  neighbours  {list(self.neighbours)}",
            f"  faces       {len(self.faces)} "
            f"({len(self.boundary_faces)} on the boundary)",
        ]
        for name, value in sorted(self.properties.items()):
            lines.append(f"  {name:<11} {value}")
        return "\n".join(lines)


def explain_connection(grid: Any, first: int, second: int,
                       transmissibility=None) -> Connection:
    """Why `first` and `second` are or are not connected.

    Two cells can share more than one face - a fault split, a corner-point
    pillar with several overlaps - so the faces are reported as a list rather
    than assumed unique. A caller with computed transmissibilities can pass
    them and get the values that actually reach the discretization.
    """
    count = int(grid.cells.num)
    for cell in (first, second):
        if not 0 <= cell < count:
            return Connection((int(first), int(second)), False,
                              reason=f"cell {cell} is outside the grid "
                                     f"(0 to {count - 1})")
    if first == second:
        return Connection((int(first), int(second)), False,
                          reason="a cell is not connected to itself")

    neighbors = np.asarray(grid.faces.neighbors, dtype=np.int64)
    pair = {int(first), int(second)}
    shared = [int(f) for f in range(neighbors.shape[0])
              if set(int(c) for c in neighbors[f] if c >= 0) == pair]

    if not shared:
        return Connection((int(first), int(second)), False,
                          reason=_why_not(grid, neighbors, int(first), int(second)))

    areas = np.asarray(getattr(grid.faces, "areas", []), dtype=float)
    values = ()
    if transmissibility is not None:
        trans = np.asarray(transmissibility, dtype=float).reshape(-1)
        if trans.size == neighbors.shape[0]:
            values = tuple(float(trans[f]) for f in shared)
    return Connection(
        (int(first), int(second)), True, tuple(shared),
        tuple(float(areas[f]) for f in shared) if areas.size else (),
        values,
    )


def _why_not(grid, neighbors, first: int, second: int) -> str:
    """A reason, not just a negative.

    "not connected" is unhelpful when the question is why. Sharing no face at
    all is a different situation from sitting in separate flow regions, and
    the second is usually the surprise.
    """
    from pymrst.research_studio.inspection.grid_report import _connected_regions

    interior = neighbors[np.all(neighbors >= 0, axis=1)]
    regions = _connected_regions(int(grid.cells.num), interior)
    home = {cell: index for index, region in enumerate(regions) for cell in region}
    if home.get(first) != home.get(second):
        return (f"they lie in different flow regions "
                f"({home.get(first)} and {home.get(second)}) - no path of "
                f"interior faces joins them")
    return "they share no face; the grid connects them only through other cells"


def inspect_cell(grid: Any, index: int, rock: Any = None,
                 state: Any = None) -> CellReport:
    """Geometry, connectivity and properties of one cell."""
    count = int(grid.cells.num)
    if not 0 <= index < count:
        raise IndexError(f"cell {index} is outside the grid (0 to {count - 1})")

    faces = np.asarray(
        grid.cells.faces[grid.cells.facePos[index]:grid.cells.facePos[index + 1]],
        dtype=np.int64,
    ).ravel()
    neighbors = np.asarray(grid.faces.neighbors, dtype=np.int64)
    adjacent, boundary = [], []
    for face in faces:
        pair = neighbors[face]
        other = [int(c) for c in pair if c != index and c >= 0]
        if len(pair) and np.any(pair < 0):
            boundary.append(int(face))
        adjacent.extend(other)

    properties: dict[str, Any] = {}
    if rock is not None:
        poro = np.asarray(getattr(rock, "poro", []), dtype=float).reshape(-1)
        if poro.size > index:
            properties["porosity"] = float(poro[index])
        perm = np.asarray(getattr(rock, "perm", []), dtype=float)
        if perm.size and perm.shape[0] > index:
            properties["permeability"] = np.atleast_1d(perm[index]).tolist()
    if state is not None:
        from pymrst.utils.adi import value as strip
        for name in ("pressure", "sW", "sO", "sG"):
            try:
                field = state[name] if hasattr(state, "keys") else getattr(state, name)
            except (KeyError, AttributeError):
                continue
            array = np.asarray(strip(field), dtype=float).reshape(-1)
            if array.size > index:
                properties[name] = float(array[index])

    centroids = np.asarray(getattr(grid.cells, "centroids", []), dtype=float)
    volumes = np.asarray(getattr(grid.cells, "volumes", []), dtype=float)
    return CellReport(
        index=int(index),
        centroid=tuple(float(x) for x in centroids[index]) if centroids.size else (),
        volume=float(volumes[index]) if volumes.size else float("nan"),
        neighbours=tuple(sorted(set(adjacent))),
        faces=tuple(int(f) for f in faces),
        boundary_faces=tuple(boundary),
        properties=properties,
    )


__all__ = ["CellReport", "Connection", "explain_connection", "inspect_cell"]
