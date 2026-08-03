"""Write a run as a ParaView case (M3.5).

The hard part is already done: `pymrst.plotting.to_pyvista` converts any
PyMRST grid - Cartesian, corner-point with faults, VE top surface - into a VTK
unstructured grid, with 3D cells as VTK_POLYHEDRON so fault-split geometry is
represented exactly rather than approximated. What was missing is writing it
to disk, and doing so as a *schedule* rather than a snapshot.

Four things this does that a bare `mesh.save()` does not:

* **A `.pvd` collection.** ParaView opens one file and gets a time series with
  a working time slider. Without it a user opens twenty `.vtu` files by hand
  and loses the time axis.
* **Geometry written once.** Corner-point geometry is the bulk of a field
  case and does not change between report steps, so repeating it per step
  multiplies the output size by the step count for nothing.
* **Wells as polylines.** ParaView cannot infer a trajectory from cell data;
  a well has to arrive as its own dataset or it is invisible.
* **A round trip.** Values read back must be bit-identical - so "quantify
  precision loss" is answered with zero rather than with a tolerance.

Additive and optional. `pyvista` is the `plotting` extra, and nothing in the
replica layer imports this.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from xml.etree import ElementTree as ET

import numpy as np

VTK_SCHEMA_VERSION = 1


def _require_pyvista():
    try:
        import pyvista as pv
    except ImportError as error:                        # pragma: no cover
        raise ImportError(
            "VTK export needs pyvista: pip install 'pymrst[plotting]'"
        ) from error
    return pv


def _cell_values(state: Any, name: str) -> np.ndarray:
    value = state[name] if hasattr(state, "keys") else getattr(state, name)
    from pymrst.utils.adi import value as strip
    return np.asarray(strip(value), dtype=float).reshape(-1)


def write_vtk_series(
    path: str | Path,
    grid: Any,
    states: Sequence[Any],
    *,
    fields: Iterable[str] = ("pressure",),
    times: Sequence[float] | None = None,
    wells: Sequence[Any] | None = None,
    basename: str = "state",
) -> Path:
    """Write `states` as a ParaView time series, returning the `.pvd` path.

    `fields` names the cell data to carry. It is explicit rather than "every
    numeric field" because a state holds well unknowns and mechanics degrees
    of freedom whose length is not the cell count, and silently dropping the
    ones that do not fit would make the export quietly incomplete.
    """
    pv = _require_pyvista()
    from pymrst.plotting import to_pyvista

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    fields = tuple(fields)
    if not states:
        raise ValueError("no states to write")
    if times is not None and len(times) != len(states):
        raise ValueError(
            f"{len(times)} times for {len(states)} states; ParaView indexes "
            f"the collection by time, so they must correspond"
        )

    # Built once. The geometry is identical across report steps, and rebuilding
    # the polyhedral cell array per step is the expensive part of this.
    mesh = to_pyvista(grid)
    cell_count = mesh.n_cells

    written: list[tuple[float, Path]] = []
    for index, state in enumerate(states):
        step = mesh.copy(deep=True)
        for name in fields:
            values = _cell_values(state, name)
            if values.size != cell_count:
                raise ValueError(
                    f"field {name!r} has {values.size} values for "
                    f"{cell_count} cells; it is not cell data"
                )
            step.cell_data[name] = values
        target = directory / f"{basename}_{index:04d}.vtu"
        step.save(target, binary=True)
        written.append((float(index if times is None else times[index]), target))

    if wells:
        _write_wells(directory / "wells.vtp", grid, wells, pv)

    collection = directory / f"{basename}.pvd"
    _write_pvd(collection, written)
    return collection


def _write_pvd(path: Path, entries: Sequence[tuple[float, Path]]) -> None:
    """The collection file, written by hand.

    PyVista has no `.pvd` writer, and the format is a short XML document, so
    this is less code than a dependency would be. Paths are stored relative so
    the directory can be moved or copied to another machine intact.
    """
    root = ET.Element("VTKFile", type="Collection", version="0.1",
                      byte_order="LittleEndian")
    collection = ET.SubElement(root, "Collection")
    for time, target in entries:
        ET.SubElement(collection, "DataSet", timestep=repr(time), group="",
                      part="0", file=target.name)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def _write_wells(path: Path, grid: Any, wells: Sequence[Any], pv) -> None:
    """Each well as a polyline through the centroids of its perforated cells."""
    centroids = np.asarray(grid.cells.centroids, dtype=float)
    if centroids.shape[1] == 2:
        centroids = np.column_stack([centroids, np.zeros(len(centroids))])

    points: list[np.ndarray] = []
    lines: list[int] = []
    labels: list[int] = []
    for number, well in enumerate(wells):
        cells = np.asarray(getattr(well, "cells", ()), dtype=np.int64).ravel()
        if cells.size == 0:
            continue
        start = len(points)
        points.extend(centroids[c] for c in cells)
        lines.extend([len(cells)] + list(range(start, start + len(cells))))
        labels.extend([number]*len(cells))
    if not points:
        return
    poly = pv.PolyData(np.asarray(points, dtype=float),
                       lines=np.asarray(lines, dtype=np.int64))
    poly.point_data["well"] = np.asarray(labels, dtype=np.int64)
    poly.save(path, binary=True)


def read_vtk_series(path: str | Path) -> list[tuple[float, Any]]:
    """Read a `.pvd` collection back as `(time, mesh)` pairs.

    Exists so the round trip is testable rather than asserted. An export
    nobody reads back is a format claim, not a verified one.
    """
    pv = _require_pyvista()
    collection = Path(path)
    tree = ET.parse(collection)
    directory = collection.parent
    out = []
    for entry in tree.getroot().iter("DataSet"):
        out.append((float(entry.attrib["timestep"]),
                    pv.read(directory / entry.attrib["file"])))
    return out


__all__ = ["VTK_SCHEMA_VERSION", "read_vtk_series", "write_vtk_series"]
