"""Read SAM simulator grid files (MRST ``readSimData.m``)."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np

from pymrst.gridprocessing.grid_structure import Cells, Faces, Grid, Nodes


def _numbers(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"SAM grid file does not exist: {path}")
    lines = path.read_text().splitlines()
    return re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?", "\n".join(lines[1:]))


class _Tokens:
    def __init__(self, values):
        self.values = values
        self.index = 0

    def integer(self) -> int:
        value = int(float(self.values[self.index])); self.index += 1
        return value

    def floats(self, count: int) -> np.ndarray:
        out = np.asarray(self.values[self.index:self.index + count], dtype=float)
        if out.size != count:
            raise ValueError("unexpected end of SAM file")
        self.index += count
        return out


def _read_topology(base: Path):
    tokens = _Tokens(_numbers(Path(f"{base}-topo.dat")))
    nc, nhf, nf, nn = (tokens.integer() for _ in range(4))
    counts = np.empty(nc, dtype=np.int64)
    local_half_faces = []
    for cell in range(nc):
        counts[cell] = tokens.integer()
        local_half_faces.extend(tokens.integer() for _ in range(int(counts[cell])))
    if sorted(local_half_faces) != list(range(nhf)):
        raise ValueError("SAM topology local half-face indices are not a permutation")
    cell_faces = np.empty(nhf, dtype=np.int64)
    orientation = np.empty(nhf, dtype=np.int8)
    for half_face in range(nhf):
        cell_faces[half_face] = tokens.integer()
        orientation[half_face] = tokens.integer()
    node_counts = np.empty(nf, dtype=np.int64)
    face_nodes = []
    for face in range(nf):
        node_counts[face] = tokens.integer()
        face_nodes.extend(tokens.integer() for _ in range(int(node_counts[face])))
    face_pos = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    node_pos = np.concatenate([[0], np.cumsum(node_counts)]).astype(np.int64)
    cell_no = np.repeat(np.arange(nc, dtype=np.int64), counts)
    neighbors = np.full((nf, 2), -1, dtype=np.int64)
    for cell, face, orient in zip(cell_no, cell_faces, orientation):
        column = 0 if orient > 0 else 1
        if neighbors[face, column] != -1:
            raise ValueError(f"duplicate SAM neighbor orientation for face {face}")
        neighbors[face, column] = cell
    return nc, nf, nn, face_pos, cell_faces, node_pos, np.asarray(face_nodes, dtype=np.int64), neighbors


def _read_geometry(base: Path):
    path = Path(f"{base}-geom.dat")
    header = path.read_text().splitlines()[0]
    match = re.match(r"geometry\s+0:(\d+):point", header.strip())
    if not match:
        raise ValueError(f"cannot determine SAM geometry dimension from {path}")
    dim = int(match.group(1))
    tokens = _Tokens(_numbers(path))
    nn = tokens.integer(); coords = tokens.floats(nn * dim).reshape(nn, dim)
    nf1 = tokens.integer(); unit_normals = tokens.floats(nf1 * dim).reshape(nf1, dim)
    nf2 = tokens.integer(); face_centroids = tokens.floats(nf2 * dim).reshape(nf2, dim)
    nf3 = tokens.integer(); areas = tokens.floats(nf3)
    nc1 = tokens.integer(); cell_centroids = tokens.floats(nc1 * dim).reshape(nc1, dim)
    nc2 = tokens.integer(); volumes = tokens.floats(nc2)
    if not (nf1 == nf2 == nf3) or nc1 != nc2:
        raise ValueError("inconsistent SAM geometry record counts")
    return dim, coords, unit_normals * areas[:, None], face_centroids, areas, cell_centroids, volumes


def _read_map(base: Path, nc: int):
    path = Path(f"{base}-map.dat")
    if not path.is_file():
        return None, np.arange(nc, dtype=np.int64), None
    values = np.fromstring(path.read_text(), sep=" ")
    if values.size < 4:
        raise ValueError(f"invalid SAM map file: {path}")
    dims = tuple(values[:3].astype(int))
    count = int(values[3])
    ijk = values[4:].astype(int).reshape(count, 3)
    if count != nc:
        raise ValueError("SAM map cell count does not match topology")
    index_map = ijk[:, 0] + dims[0] * (ijk[:, 1] + dims[1] * ijk[:, 2])
    return dims, index_map.astype(np.int64), ijk


def read_sim_data(casename) -> Grid:
    """Read ``-topo.dat``, ``-geom.dat`` and optional ``-map.dat``."""
    base = Path(casename)
    nc, nf, nn, face_pos, cell_faces, node_pos, face_nodes, neighbors = _read_topology(base)
    dim, coords, normals, face_centroids, areas, cell_centroids, volumes = _read_geometry(base)
    if (coords.shape[0], face_centroids.shape[0], cell_centroids.shape[0]) != (nn, nf, nc):
        raise ValueError("SAM topology and geometry counts do not agree")
    cart_dims, index_map, ijk = _read_map(base, nc)
    cells = Cells(nc, face_pos, cell_faces, index_map, volumes=volumes, centroids=cell_centroids)
    if ijk is not None:
        cells.ijkMap = ijk
    faces = Faces(nf, node_pos, face_nodes, neighbors, areas=areas, normals=normals, centroids=face_centroids)
    grid = Grid(cells, faces, Nodes(nn, coords), dim, type=["readSimData"], cartDims=cart_dims)
    grid.check()
    return grid


readSimData = read_sim_data

__all__ = ["readSimData", "read_sim_data"]
