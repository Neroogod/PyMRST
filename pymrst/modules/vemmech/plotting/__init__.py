"""Deformed-grid plotting helpers from MRST ``vemmech``.

The ``*_geometry`` functions return the backend-agnostic geometry that would be
sent to a plotting backend. The higher level ``plot_*`` functions return a
lightweight handle object, mirroring MATLAB graphics handles while remaining
usable in headless tests.
"""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

import numpy as np


def deformed_grid(g, u):
    """Return a copy of ``g`` with node coordinates shifted by ``u``."""
    out = deepcopy(g)
    uu = np.asarray(u, dtype=float)
    out.nodes.coords = out.nodes.coords + uu[:, :out.nodes.coords.shape[1]]
    return out


def plot_grid_deformed_geometry(g, u):
    """Geometry counterpart of MRST ``plotGridDeformed``."""
    return deformed_grid(g, u)


def plot_cell_data_deformed_geometry(g, data, u):
    """Geometry counterpart of MRST ``plotCellDataDeformed``."""
    return {"grid": deformed_grid(g, u), "data": np.asarray(data)}


def plot_face_data_deformed_geometry(g, data, u):
    """Geometry counterpart of MRST ``plotFaceDataDeformed``."""
    return {"grid": deformed_grid(g, u), "data": np.asarray(data)}


def _boundary_faces(g, cells: Optional[Iterable[int]] = None) -> np.ndarray:
    if cells is None:
        cell_mask = np.ones(g.cells.num, dtype=bool)
    else:
        cell_mask = np.zeros(g.cells.num, dtype=bool)
        cell_mask[np.asarray(list(cells), dtype=np.int64)] = True
    neigh = g.faces.neighbors
    in0 = (neigh[:, 0] >= 0) & cell_mask[np.maximum(neigh[:, 0], 0)]
    in1 = (neigh[:, 1] >= 0) & cell_mask[np.maximum(neigh[:, 1], 0)]
    return np.flatnonzero(in0 ^ in1)


def _local_numbering(nodes: np.ndarray, nn: np.ndarray, nnode: int):
    present = np.zeros(nnode, dtype=bool)
    present[nodes] = True
    present_nodes = np.flatnonzero(present)
    node_num = np.zeros(nnode, dtype=np.int64)
    node_num[present_nodes] = np.arange(1, present_nodes.size + 1)
    maxn = int(nn.max()) if nn.size else 0
    faces = np.full((nn.size, maxn), np.nan)
    at = 0
    for i, n in enumerate(nn):
        loc = nodes[at:at + n]
        faces[i, :n] = node_num[loc]
        at += n
    return faces, present_nodes


def node_patch_topology(g, entities: Optional[np.ndarray] = None):
    """Patch topology used by MRST ``plotNodeDataDeformed``.

    For 3D grids, ``entities`` are faces. For 2D grids, ``entities`` are cells.
    Indices in the returned face matrix are 1-based, matching MATLAB patch
    geometry. ``verts`` are 0-based PyMRST node indices.
    """
    if g.griddim == 3:
        faces = _boundary_faces(g) if entities is None else np.asarray(
            entities, dtype=np.int64
        )
        nn = np.diff(g.faces.nodePos)[faces]
        nodes = np.concatenate([
            g.faces.nodes[g.faces.nodePos[f]:g.faces.nodePos[f + 1]]
            for f in faces
        ]) if faces.size else np.zeros(0, dtype=np.int64)
        fmat, verts = _local_numbering(nodes, nn, g.nodes.num)
        for i, n in enumerate(nn):
            if n < fmat.shape[1]:
                fmat[i, n:] = fmat[i, n - 1]
        return fmat, verts

    cells = np.arange(g.cells.num) if entities is None else np.asarray(
        entities, dtype=np.int64
    )
    if getattr(g.cells, "nodes", None) is not None:
        nn = np.diff(g.cells.nodePos)[cells]
        per_cell = []
        for c in cells:
            cn = g.cells.nodes[g.cells.nodePos[c]:g.cells.nodePos[c + 1]]
            # MRST's createAugmentedGrid stores the same cyclic polygon but
            # starts one vertex before PyMRST's angle-sorted convention.
            per_cell.append(np.roll(cn, 1))
        nodes = np.concatenate(per_cell) if cells.size else np.zeros(
            0, dtype=np.int64
        )
    else:
        nn = np.diff(g.cells.facePos)[cells]
        nodes = np.concatenate([
            np.unique(np.concatenate([
                g.faces.nodes[g.faces.nodePos[f]:g.faces.nodePos[f + 1]]
                for f in g.cells.faces[g.cells.facePos[c]:g.cells.facePos[c + 1]]
            ]))
            for c in cells
        ]) if cells.size else np.zeros(0, dtype=np.int64)
    return _local_numbering(nodes, nn, g.nodes.num)


def plot_node_data_deformed_geometry(g, node_data, u):
    """Patch geometry counterpart of MRST ``plotNodeDataDeformed``."""
    gd = deformed_grid(g, u)
    entities = _boundary_faces(gd) if gd.griddim == 3 else np.arange(
        gd.cells.num
    )
    faces, verts = node_patch_topology(gd, entities)
    vertices = gd.nodes.coords[verts, :]
    if faces.shape[0] == vertices.shape[0]:
        vertices = np.vstack([vertices, np.full((1, vertices.shape[1]), np.nan)])
    cdata = np.asarray(node_data)[verts, ...]
    return {
        "faces": faces,
        "vertices": vertices,
        "verts": verts,
        "cdata": cdata,
        "face_color": "interp",
    }


def plot_faces_2d_geometry(
    g,
    faces: np.ndarray,
    uu: Optional[np.ndarray] = None,
    shift: Optional[np.ndarray] = None,
):
    """Line geometry counterpart of MRST ``plotFaces2D``."""
    if g.griddim != 2:
        raise ValueError("plot_faces_2d_geometry only supports 2D grids")
    gd = deformed_grid(g, uu) if uu is not None else g
    faces = np.asarray(faces, dtype=np.int64).ravel()
    shift_arr = None if shift is None else np.asarray(shift, dtype=float).ravel()
    lines = []
    for j, face in enumerate(faces):
        nodes = gd.faces.nodes[gd.faces.nodePos[face]:gd.faces.nodePos[face + 1]]
        pos = gd.nodes.coords[nodes, :2].copy()
        if shift_arr is not None:
            pos += gd.faces.normals[face, :2] * shift_arr[j]
        lines.append(pos)
    return np.asarray(lines)


def _array_or_none(value):
    return None if value is None else np.asarray(value)


def _render_grid_pyvista(handle):
    from pymrst.plotting import plot_cell_data, plot_grid

    gd = handle.geometry["grid"] if isinstance(handle.geometry, dict) else handle.geometry
    kwargs = dict(handle.backend_kwargs)
    plotter = kwargs.pop("plotter", handle.plotter)
    show = bool(kwargs.pop("show", False))
    filename = kwargs.pop("filename", None)
    if handle.kind == "cell_data_deformed":
        data = np.asarray(handle.geometry["data"])
        result = plot_cell_data(
            gd, data, plotter=plotter, show=show, filename=filename, **kwargs
        )
    else:
        result = plot_grid(
            gd, plotter=plotter, show=show, filename=filename, **kwargs
        )
    handle.plotter = result
    handle.actor = result


@dataclass
class VEMPlotHandle:
    """MATLAB-style handle for deformed VEM plotting payloads.

    The handle stores both the original inputs and the current geometry. In
    headless use, ``backend='geometry'`` only recomputes the payload. With
    ``backend='pyvista'``, grid and cell-data handles also render through
    PyMRST's PyVista plotting layer.
    """

    kind: str
    builder: Callable[..., Any]
    g: Any
    u: Optional[np.ndarray] = None
    data: Optional[np.ndarray] = None
    entities: Optional[np.ndarray] = None
    shift: Optional[np.ndarray] = None
    backend: str = "geometry"
    backend_kwargs: dict[str, Any] = field(default_factory=dict)
    geometry: Any = None
    plotter: Any = None
    actor: Any = None

    def __post_init__(self):
        self.refresh()

    def refresh(self):
        """Recompute geometry and update the optional backend object."""
        if self.kind == "faces_2d":
            self.geometry = self.builder(
                self.g, self.entities, uu=self.u, shift=self.shift
            )
        elif self.data is None:
            self.geometry = self.builder(self.g, self.u)
        else:
            self.geometry = self.builder(self.g, self.data, self.u)

        if self.backend == "geometry":
            return self
        if self.backend == "pyvista":
            if self.kind not in {"grid_deformed", "cell_data_deformed"}:
                return self
            _render_grid_pyvista(self)
            return self
        raise ValueError(f"unknown plotting backend {self.backend!r}")

    def update(
        self,
        *,
        u: Optional[np.ndarray] = None,
        data: Optional[np.ndarray] = None,
        entities: Optional[np.ndarray] = None,
        shift: Optional[np.ndarray] = None,
    ):
        """Update handle inputs and recompute the plotted payload."""
        if u is not None:
            self.u = np.asarray(u, dtype=float)
        if data is not None:
            self.data = np.asarray(data)
        if entities is not None:
            self.entities = np.asarray(entities, dtype=np.int64)
        if shift is not None:
            self.shift = np.asarray(shift, dtype=float)
        return self.refresh()

    def set_displacement(self, u):
        """Set nodal displacement and refresh the handle."""
        return self.update(u=u)

    def set_data(self, data):
        """Set scalar/vector data and refresh the handle."""
        return self.update(data=data)

    def show(self):
        """Show the underlying backend plotter when one is available."""
        if hasattr(self.plotter, "show"):
            return self.plotter.show()
        return self.geometry


def plot_grid_deformed(g, u, backend: str = "geometry", **backend_kwargs):
    """Return a handle for MRST ``plotGridDeformed``."""
    return VEMPlotHandle(
        "grid_deformed", plot_grid_deformed_geometry, g,
        u=np.asarray(u, dtype=float), backend=backend,
        backend_kwargs=backend_kwargs,
    )


def plot_cell_data_deformed(
    g, data, u, backend: str = "geometry", **backend_kwargs
):
    """Return a handle for MRST ``plotCellDataDeformed``."""
    return VEMPlotHandle(
        "cell_data_deformed", plot_cell_data_deformed_geometry, g,
        u=np.asarray(u, dtype=float), data=np.asarray(data), backend=backend,
        backend_kwargs=backend_kwargs,
    )


def plot_face_data_deformed(
    g, data, u, backend: str = "geometry", **backend_kwargs
):
    """Return a handle for MRST ``plotFaceDataDeformed``."""
    return VEMPlotHandle(
        "face_data_deformed", plot_face_data_deformed_geometry, g,
        u=np.asarray(u, dtype=float), data=np.asarray(data), backend=backend,
        backend_kwargs=backend_kwargs,
    )


def plot_node_data_deformed(
    g, node_data, u, backend: str = "geometry", **backend_kwargs
):
    """Return a handle for MRST ``plotNodeDataDeformed``."""
    return VEMPlotHandle(
        "node_data_deformed", plot_node_data_deformed_geometry, g,
        u=np.asarray(u, dtype=float), data=np.asarray(node_data),
        backend=backend, backend_kwargs=backend_kwargs,
    )


def plot_faces_2d(
    g,
    faces: np.ndarray,
    uu: Optional[np.ndarray] = None,
    shift: Optional[np.ndarray] = None,
    backend: str = "geometry",
    **backend_kwargs,
):
    """Return a handle for MRST ``plotFaces2D``."""
    if backend != "geometry":
        raise ValueError("plot_faces_2d currently supports backend='geometry'")
    return VEMPlotHandle(
        "faces_2d", plot_faces_2d_geometry, g,
        u=_array_or_none(uu), entities=np.asarray(faces, dtype=np.int64),
        shift=_array_or_none(shift), backend=backend,
        backend_kwargs=backend_kwargs,
    )


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.vemmech.vem import create_augmented_grid

    def test_deformed_grid():
        g = compute_geometry(cart_grid([2, 1], [2.0, 1.0]))
        u = np.column_stack([0.1 * g.nodes.coords[:, 0],
                             -0.2 * g.nodes.coords[:, 1]])
        gd = deformed_grid(g, u)
        assert np.allclose(gd.nodes.coords, g.nodes.coords + u)
        print("ok test_deformed_grid")

    def test_node_patch_2d():
        g = create_augmented_grid(compute_geometry(cart_grid([2, 1],
                                                             [2.0, 1.0])))
        out = plot_node_data_deformed_geometry(
            g, np.arange(g.nodes.num), np.zeros_like(g.nodes.coords)
        )
        assert out["faces"].shape == (g.cells.num, 4)
        assert out["cdata"].shape[0] == out["verts"].size
        print("ok test_node_patch_2d")

    def test_faces_2d():
        g = compute_geometry(cart_grid([1, 1], [1.0, 1.0]))
        lines = plot_faces_2d_geometry(g, np.arange(g.faces.num))
        assert lines.shape == (g.faces.num, 2, 2)
        print("ok test_faces_2d")

    test_deformed_grid()
    test_node_patch_2d()
    test_faces_2d()
    print("\nAll tests passed!")
