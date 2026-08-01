"""Write SAM simulator files (MRST ``writeSimData.m``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pymrst.utils.units import BARSA, CENTI, DARCY, DAY, MILLI, POISE


def _base(casename) -> Path:
    path = Path(casename)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_matrix(path: Path, values) -> None:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    lines = [str(array.shape[0])]
    lines.extend(" ".join(f"{value:.16g}" for value in row) for row in array)
    path.write_text("\n".join(lines) + "\n")


def write_sim_rock(casename, rock) -> None:
    """Write SAM permeability (mD on disk) and porosity files."""
    base = _base(casename)
    permeability = np.asarray(rock.perm, dtype=float) / (MILLI * DARCY)
    porosity = np.asarray(rock.poro, dtype=float).reshape(-1)
    if permeability.ndim == 1:
        permeability = permeability[:, None]
    if permeability.shape[0] != porosity.size:
        raise ValueError("permeability and porosity record counts do not match")
    if permeability.shape[1] not in (1, 2, 3):
        raise ValueError("SAM permeability must have one, two, or three columns")
    _write_matrix(Path(f"{base}-perm.dat"), permeability)
    _write_matrix(Path(f"{base}-poro.dat"), porosity)


def write_sim_grid(casename, grid) -> None:
    """Write SAM geometry, topology, and optional Cartesian map files."""
    base = _base(casename)
    cells, faces, nodes = grid.cells, grid.faces, grid.nodes
    required = (
        cells.centroids, cells.volumes, faces.centroids, faces.normals, faces.areas
    )
    if any(value is None for value in required):
        raise ValueError("SAM grid output requires computed cell and face geometry")
    dim = int(nodes.coords.shape[1])

    geometry = [
        f"geometry 0:{dim}:point {dim-1}:{dim}:normal {dim-1}:{dim}:centroid "
        f"{dim-1}:1:area {dim}:{dim}:centroid {dim}:1:volume",
        "",
        str(nodes.num),
    ]
    geometry.extend(" ".join(f"{v:.16g}" for v in row) for row in np.asarray(nodes.coords))
    unit_normals = np.divide(
        np.asarray(faces.normals, dtype=float),
        np.asarray(faces.areas, dtype=float)[:, None],
        out=np.zeros_like(np.asarray(faces.normals, dtype=float)),
        where=np.asarray(faces.areas, dtype=float)[:, None] != 0.0,
    )
    for values in (unit_normals, faces.centroids):
        geometry.extend(["", str(faces.num)])
        geometry.extend(" ".join(f"{v:.16g}" for v in row) for row in np.asarray(values))
    geometry.extend(["", str(faces.num)])
    geometry.extend(f"{v:.16g}" for v in np.asarray(faces.areas).reshape(-1))
    geometry.extend(["", str(cells.num)])
    geometry.extend(" ".join(f"{v:.16g}" for v in row) for row in np.asarray(cells.centroids))
    geometry.extend(["", str(cells.num)])
    geometry.extend(f"{v:.16g}" for v in np.asarray(cells.volumes).reshape(-1))
    Path(f"{base}-geom.dat").write_text("\n".join(geometry) + "\n")

    half_face_count = int(np.asarray(cells.faces).size)
    topology = [
        f"topology {dim} {dim-1}o {dim-1} 0 {dim}-{dim-1}o "
        f"{dim-1}o-{dim-1} {dim-1}-0",
        "",
        f"{cells.num} {half_face_count} {faces.num} {nodes.num}",
        "",
    ]
    for cell in range(cells.num):
        start, stop = int(cells.facePos[cell]), int(cells.facePos[cell + 1])
        topology.append(" ".join(map(str, [stop - start, *range(start, stop)])))
    topology.append("")
    cell_no = np.repeat(np.arange(cells.num, dtype=np.int64), np.diff(cells.facePos))
    for cell, face in zip(cell_no, np.asarray(cells.faces, dtype=np.int64)):
        orientation = 1 if int(faces.neighbors[face, 0]) == int(cell) else 0
        topology.append(f"{int(face)} {orientation}")
    topology.append("")
    for face in range(faces.num):
        start, stop = int(faces.nodePos[face]), int(faces.nodePos[face + 1])
        face_nodes = np.asarray(faces.nodes[start:stop], dtype=np.int64).tolist()
        topology.append(" ".join(map(str, [len(face_nodes), *face_nodes])))
    Path(f"{base}-topo.dat").write_text("\n".join(topology) + "\n")

    cart_dims = getattr(grid, "cartDims", None)
    ijk = getattr(cells, "ijkMap", None)
    if cart_dims is not None and ijk is not None:
        dims = tuple(int(v) for v in cart_dims)
        if len(dims) != 3:
            raise ValueError("SAM map output requires three Cartesian dimensions")
        lines = [" ".join(map(str, dims)), str(cells.num)]
        lines.extend(" ".join(map(str, row)) for row in np.asarray(ijk, dtype=int))
        Path(f"{base}-map.dat").write_text("\n".join(lines) + "\n")


def write_sim_sources_and_wells(casename, src=None, wells=None) -> None:
    """Write MRST's compact SAM ``-wells.dat`` forcing format."""
    base = _base(casename)
    wells = [] if wells is None else list(wells)
    source_cells = np.asarray(getattr(src, "cell", []), dtype=int).reshape(-1) if src is not None else np.empty(0, int)
    source_rates = np.asarray(getattr(src, "rate", []), dtype=float).reshape(-1) if src is not None else np.empty(0)
    source_sat = np.asarray(getattr(src, "sat", np.ones(source_cells.size)), dtype=float)
    if source_cells.size != source_rates.size:
        raise ValueError("source cells and rates must have equal length")
    lines = [str(source_cells.size + len(wells))]
    for index, (cell, rate) in enumerate(zip(source_cells, source_rates), start=1):
        rate_day = abs(float(rate)) * DAY
        kind = "SimpleInjection" if rate > 0 else "SimpleProduction"
        sat = float(source_sat[index - 1, 0] if source_sat.ndim == 2 else source_sat[index - 1])
        lines.append(f"{kind} {'I' if rate > 0 else 'P'}{index} 1 {rate_day:.16g} {sat:.16g}; {int(cell)} {rate_day:.16g}")
    wi_unit = CENTI * POISE / (DAY * BARSA)
    for well in wells:
        cells = np.asarray(well.cells, dtype=int).reshape(-1)
        compi = np.asarray(well.compi, dtype=float).reshape(-1)
        if str(well.type).lower() == "bhp":
            head = f"Pressure {well.name} {cells.size} {float(well.val):.16g} {compi[0]:.16g}"
        elif str(well.type).lower() == "rate":
            kind = "Injection" if well.val > 0 else "Production"
            head = f"{kind} {well.name} {cells.size} {abs(float(well.val)) * DAY:.16g} {compi[0]:.16g}"
        else:
            raise ValueError(f"unsupported SAM well type {well.type!r}")
        wi = np.asarray(well.WI, dtype=float).reshape(-1)
        tail = "".join(f"; {int(cell)} {float(index) / wi_unit:.16g}" for cell, index in zip(cells, wi))
        lines.append(head + tail)
    Path(f"{base}-wells.dat").write_text("\n".join(lines) + "\n")


def write_sim_data(grid=None, casename="case", *, rock=None, src=None, wells=None) -> None:
    """Write any supplied SAM grid, rock, source, and well data."""
    if grid is not None:
        write_sim_grid(casename, grid)
    if rock is not None:
        write_sim_rock(casename, rock)
    if src is not None or wells is not None:
        write_sim_sources_and_wells(casename, src=src, wells=wells)


writeSimData = write_sim_data

__all__ = [
    "writeSimData", "write_sim_data", "write_sim_grid", "write_sim_rock",
    "write_sim_sources_and_wells",
]
