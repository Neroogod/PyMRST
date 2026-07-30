"""Named ECLIPSE grid faults (MRST ``processFaults``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from pymrst.gridprocessing.grid_structure import (
    BOTTOM,
    EAST,
    NORTH,
    SOUTH,
    TOP,
    WEST,
)


@dataclass(frozen=True)
class GridFault:
    """Faces and transmissibility multiplier for one named grid fault."""

    name: str
    faces: np.ndarray
    multiplier: float = 1.0

    @property
    def numf(self) -> int:
        return int(self.faces.size)

    @property
    def mult(self) -> float:
        """MRST field spelling."""
        return self.multiplier


_DIRECTION = {
    "X-": WEST, "I-": WEST,
    "X+": EAST, "I+": EAST,
    "Y-": SOUTH, "J-": SOUTH,
    "Y+": NORTH, "J+": NORTH,
    "Z-": TOP, "K-": TOP,
    "Z+": BOTTOM, "K+": BOTTOM,
}


def _field(mapping: Mapping, name: str, default=None):
    matches = [key for key in mapping if str(key).lower() == name.lower()]
    if len(matches) > 1:
        raise ValueError(f"multiple case-insensitive {name} fields")
    return mapping[matches[0]] if matches else default


def _name(value) -> str:
    return str(value).strip().strip("'\"")


def _records(raw) -> list[list]:
    if raw is None:
        return []
    array = np.asarray(raw, dtype=object)
    if array.size == 0:
        return []
    if array.ndim == 1:
        array = array.reshape(1, -1)
    return [list(row) for row in array]


def _cartesian_cells(
    dims: Sequence[int], i1, i2, j1, j2, k1, k2
) -> np.ndarray:
    """Zero-based global cells for one-based inclusive ECLIPSE ranges."""
    nx, ny, nz = (int(value) for value in dims)
    bounds = [int(value) for value in (i1, i2, j1, j2, k1, k2)]
    ia, ib, ja, jb, ka, kb = bounds
    if not (1 <= ia <= ib <= nx and 1 <= ja <= jb <= ny
            and 1 <= ka <= kb <= nz):
        raise ValueError(
            f"fault box {bounds} is outside cartDims {(nx, ny, nz)}"
        )
    i, j, k = np.meshgrid(
        np.arange(ia - 1, ib),
        np.arange(ja - 1, jb),
        np.arange(ka - 1, kb),
        indexing="ij",
    )
    return (i + nx * j + nx * ny * k).ravel(order="F")


def _face_stack(grid, active_cells: np.ndarray, direction: int) -> np.ndarray:
    faces = []
    for cell in active_cells:
        if cell < 0:
            continue
        lo, hi = grid.cells.facePos[cell:cell + 2]
        tags = np.asarray(grid.cells.tags[lo:hi], dtype=np.int64)
        selected = np.asarray(grid.cells.faces[lo:hi], dtype=np.int64)[
            tags == direction
        ]
        faces.extend(selected.tolist())
    return np.asarray(faces, dtype=np.int64)


def process_faults(
    grid, geometry: Mapping
) -> tuple[list[GridFault], dict[str, int]]:
    """Construct named fault face sets from ``FAULTS`` and ``MULTFLT``.

    FAULTS rows have the ECLIPSE form
    ``name, i1, i2, j1, j2, k1, k2, direction``. Input ranges remain
    one-based and inclusive; returned face indices are zero-based.
    """
    fault_rows = _records(_field(geometry, "FAULTS"))
    if not fault_rows:
        return [], {}
    if grid.cartDims is None or len(grid.cartDims) != 3:
        raise ValueError("process_faults requires a 3D Cartesian index map")
    if grid.cells.tags is None:
        raise ValueError("process_faults requires cell-face direction tags")

    multipliers = {
        _name(row[0]): float(row[1])
        for row in _records(_field(geometry, "MULTFLT"))
    }
    by_name: dict[str, list[list]] = {}
    for row in fault_rows:
        if len(row) < 8:
            raise ValueError("FAULTS rows require eight fields")
        by_name.setdefault(_name(row[0]), []).append(row)

    total = int(np.prod(grid.cartDims))
    active_map = np.full(total, -1, dtype=np.int64)
    index_map = np.asarray(grid.cells.indexMap, dtype=np.int64)
    active_map[index_map] = np.arange(index_map.size, dtype=np.int64)

    faults: list[GridFault] = []
    for name in sorted(by_name):
        selected = []
        for row in by_name[name]:
            direction = _name(row[7]).upper()
            if direction not in _DIRECTION:
                raise ValueError(f"unsupported fault direction {row[7]!r}")
            cart = _cartesian_cells(grid.cartDims, *row[1:7])
            selected.extend(
                _face_stack(
                    grid, active_map[cart], _DIRECTION[direction]
                ).tolist()
            )
        faces = np.unique(np.asarray(selected, dtype=np.int64))
        multiplier = multipliers.get(name, 1.0)
        if faces.size and np.isfinite(multiplier):
            faults.append(GridFault(name, faces, multiplier))
    return faults, {fault.name: index for index, fault in enumerate(faults)}


def remove_fault_boundary_faces(
    faults: Iterable[GridFault], grid
) -> tuple[list[GridFault], np.ndarray]:
    """Remove exterior boundary faces and faults that become empty.

    This is MRST ``removeFaultBdryFaces``. The boolean keep mask refers to the
    original fault sequence.
    """
    faults = list(faults)
    keep = np.zeros(len(faults), dtype=bool)
    output = []
    neighbors = np.asarray(grid.faces.neighbors, dtype=np.int64)
    for index, fault in enumerate(faults):
        faces = np.asarray(fault.faces, dtype=np.int64)
        if np.any(faces < 0) or np.any(faces >= grid.faces.num):
            raise IndexError(f"fault {fault.name!r} references invalid faces")
        interior = np.all(neighbors[faces] >= 0, axis=1)
        remaining = faces[interior]
        if remaining.size:
            keep[index] = True
            output.append(
                GridFault(fault.name, remaining, fault.multiplier)
            )
    return output, keep


processFaults = process_faults
removeFaultBdryFaces = remove_fault_boundary_faces

__all__ = [
    "GridFault",
    "process_faults",
    "processFaults",
    "remove_fault_boundary_faces",
    "removeFaultBdryFaces",
]
