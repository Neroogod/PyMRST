"""Grid and cell-data visualization with PyVista (MRST plotting/).

Ports plotGrid, plotCellData, plotFaces, and plotWell. Any PyMRST grid -
Cartesian, corner-point with faults, or a VE top surface - is converted
to a VTK unstructured grid of polyhedral cells, so faulted and pinched
geometry renders exactly as it is.

Interactive by default; pass a filename to save a PNG headlessly:

    plot_cell_data(G, state.pressure, filename="p.png")

Depths increase downward in reservoir grids, so the camera is set up
with the z-axis pointing down unless flip_z=False.
"""

from typing import Optional, Sequence, Union

import numpy as np
import pyvista as pv

from pymrst.gridprocessing.grid import Grid
from pymrst.params.wells_and_bc import Well


def _scaled_coords(g: Grid, zscale: float = 1.0) -> np.ndarray:
    coords = np.asarray(g.nodes.coords, dtype=float).copy()
    if coords.shape[1] == 2:                       # lift 2D grids into 3D
        coords = np.column_stack([coords, np.zeros(coords.shape[0])])
    if not np.isclose(float(zscale), 1.0):
        z0 = float(np.min(coords[:, 2]))
        coords[:, 2] = z0 + (coords[:, 2] - z0) * float(zscale)
    return coords


def _scaled_points(points: np.ndarray, zscale: float = 1.0) -> np.ndarray:
    pts = np.asarray(points, dtype=float).copy()
    if pts.shape[1] == 2:
        pts = np.column_stack([pts, np.zeros(pts.shape[0])])
    if not np.isclose(float(zscale), 1.0):
        z0 = float(np.min(pts[:, 2]))
        pts[:, 2] = z0 + (pts[:, 2] - z0) * float(zscale)
    return pts


def to_pyvista(
    g: Grid,
    cells: Optional[np.ndarray] = None,
    zscale: float = 1.0,
) -> pv.UnstructuredGrid:
    """Convert a PyMRST grid to a VTK unstructured grid.

    3D cells become VTK_POLYHEDRON (so corner-point cells with any number
    of faces, including fault-split ones, are represented exactly); 2D
    cells become VTK_POLYGON.
    """
    sel = (np.arange(g.cells.num) if cells is None
           else np.asarray(cells, dtype=np.int64).ravel())

    coords = _scaled_coords(g, zscale)

    cell_arr: list[int] = []
    cell_types: list[int] = []

    if g.griddim == 3:
        for c in sel:
            faces = g.cells.faces[g.cells.facePos[c]:g.cells.facePos[c + 1]]
            stream = [len(faces)]
            for f in faces:
                fn = g.faces.nodes[g.faces.nodePos[f]:g.faces.nodePos[f + 1]]
                stream.append(len(fn))
                stream.extend(int(i) for i in fn)
            cell_arr.append(len(stream))
            cell_arr.extend(stream)
            cell_types.append(pv.CellType.POLYHEDRON)
    else:
        for c in sel:
            faces = g.cells.faces[g.cells.facePos[c]:g.cells.facePos[c + 1]]
            edges = [g.faces.nodes[g.faces.nodePos[f]:g.faces.nodePos[f + 1]]
                     for f in faces]
            ring = _order_ring(edges)
            cell_arr.append(len(ring))
            cell_arr.extend(int(i) for i in ring)
            cell_types.append(pv.CellType.POLYGON)

    grid = pv.UnstructuredGrid(np.array(cell_arr, dtype=np.int64),
                               np.array(cell_types, dtype=np.uint8),
                               coords)
    grid.cell_data["cell_index"] = sel
    return grid


def _order_ring(edges: Sequence[np.ndarray]) -> list[int]:
    """Chain 2-node edges of a 2D cell into a closed vertex ring."""
    remaining = [list(map(int, e)) for e in edges]
    ring = remaining.pop(0)
    while remaining:
        for k, e in enumerate(remaining):
            if e[0] == ring[-1]:
                ring.append(e[1])
                remaining.pop(k)
                break
            if e[1] == ring[-1]:
                ring.append(e[0])
                remaining.pop(k)
                break
        else:
            break                     # open chain: give up, use what we have
    if len(ring) > 1 and ring[0] == ring[-1]:
        ring.pop()
    return ring


def _finish(pl: pv.Plotter, filename: Optional[str], flip_z: bool,
            show: bool):
    if flip_z:
        pl.camera_position = "xz"
        pl.camera.up = (0.0, 0.0, -1.0)   # depth increases downward
        pl.camera.azimuth = 210
        pl.camera.elevation = 25
    pl.add_axes()
    if filename is not None:
        pl.screenshot(filename)
        pl.close()
        return filename
    if show:
        pl.show()
    return pl


def plot_grid(
    g: Grid,
    cells: Optional[np.ndarray] = None,
    color: str = "lightgray",
    show_edges: bool = True,
    opacity: float = 1.0,
    filename: Optional[str] = None,
    show: bool = True,
    plotter: Optional[pv.Plotter] = None,
    flip_z: bool = True,
    zscale: float = 1.0,
):
    """Draw the grid (plotGrid)."""
    pl = plotter or pv.Plotter(off_screen=filename is not None)
    pl.add_mesh(to_pyvista(g, cells, zscale=zscale), color=color, show_edges=show_edges,
                opacity=opacity)
    if plotter is not None:
        return pl
    return _finish(pl, filename, flip_z, show)


def plot_cell_data(
    g: Grid,
    data: np.ndarray,
    cells: Optional[np.ndarray] = None,
    label: str = "",
    cmap: str = "viridis",
    clim: Optional[tuple] = None,
    show_edges: bool = False,
    opacity: float = 1.0,
    filename: Optional[str] = None,
    show: bool = True,
    plotter: Optional[pv.Plotter] = None,
    flip_z: bool = True,
    zscale: float = 1.0,
):
    """Colour cells by a per-cell field (plotCellData)."""
    data = np.asarray(data, dtype=np.float64).ravel()
    assert data.size == g.cells.num, (
        f"expected one value per cell ({g.cells.num}), got {data.size}"
    )
    mesh = to_pyvista(g, cells, zscale=zscale)
    mesh.cell_data[label or "data"] = data[mesh.cell_data["cell_index"]]
    pl = plotter or pv.Plotter(off_screen=filename is not None)
    pl.add_mesh(mesh, scalars=label or "data", cmap=cmap, clim=clim,
                show_edges=show_edges, opacity=opacity,
                scalar_bar_args={"title": label or "data"})
    if plotter is not None:
        return pl
    return _finish(pl, filename, flip_z, show)


def plot_well(
    g: Grid,
    wells: Union[Well, Sequence[Well]],
    plotter: pv.Plotter,
    color: str = "red",
    radius: Optional[float] = None,
    height: float = 0.0,
    zscale: float = 1.0,
):
    """Draw well trajectories as tubes through their perforated cells."""
    if isinstance(wells, Well):
        wells = [wells]
    extent = float(np.ptp(g.cells.centroids[:, :2]))
    r = radius if radius is not None else 0.008 * extent
    for w in wells:
        pts = _scaled_points(g.cells.centroids[w.cells], zscale=zscale)
        top = pts[np.argmin(pts[:, 2])].copy()
        top[2] -= height if height else 0.15 * float(np.ptp(pts[:, 2]) or r)
        path = np.vstack([top, pts])
        if path.shape[0] == 1:
            plotter.add_mesh(pv.Sphere(radius=r, center=path[0]), color=color)
        else:
            plotter.add_mesh(pv.Spline(path, 100).tube(radius=r), color=color)
    return plotter


if __name__ == "__main__":
    import os
    import tempfile

    import numpy as np

    pv.OFF_SCREEN = True
    OUT = tempfile.mkdtemp(prefix="pymrst_plots_")

    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.gridprocessing.process_grdecl import process_grdecl
    from pymrst.modules.deckformat import read_grdecl
    from pymrst.params.rock import make_rock
    from pymrst.params.wells_and_bc import add_well
    from pymrst.utils.units import DARCY, MILLI

    def _png(name):
        return os.path.join(OUT, name)

    def test_to_pyvista_cartesian():
        """A Cartesian grid converts with the right cell count and volume."""
        g = compute_geometry(cart_grid([4, 3, 2], [40.0, 30.0, 20.0]))
        mesh = to_pyvista(g)
        assert mesh.n_cells == g.cells.num == 24
        assert mesh.n_points == g.nodes.num
        vol = mesh.compute_cell_sizes(length=False, area=False,
                                      volume=True)["Volume"]
        assert np.allclose(np.abs(vol), g.cells.volumes, rtol=1e-10)
        print("ok test_to_pyvista_cartesian")

    def test_to_pyvista_corner_point():
        """The real SPE1 corner-point grid converts and preserves volume."""
        from pathlib import Path
        p = (Path(__file__).parents[2]
             / "../mrst-2024a-Geomech/examples/data/SPE1/SPE1.GRDECL")
        g = compute_geometry(process_grdecl(read_grdecl(p.resolve())))
        mesh = to_pyvista(g)
        assert mesh.n_cells == g.cells.num == 300
        vol = mesh.compute_cell_sizes(length=False, area=False,
                                      volume=True)["Volume"]
        assert np.allclose(np.abs(vol), g.cells.volumes, rtol=1e-9)
        print("ok test_to_pyvista_corner_point")

    def test_to_pyvista_2d():
        """2D grids become polygons and preserve area."""
        g = compute_geometry(cart_grid([5, 4], [10.0, 8.0]))
        mesh = to_pyvista(g)
        assert mesh.n_cells == 20
        area = mesh.compute_cell_sizes(length=False, area=True,
                                       volume=False)["Area"]
        assert np.allclose(area, g.cells.volumes, rtol=1e-10)
        print("ok test_to_pyvista_2d")

    def test_plot_cell_data_writes_png():
        """plot_cell_data renders a non-trivial image headlessly."""
        g = compute_geometry(cart_grid([10, 10, 3], [100.0, 100.0, 30.0]))
        rock = make_rock(g, 100 * MILLI * DARCY, 0.2)
        w = add_well(None, g, rock, np.array([0, 100, 200]),
                     well_type="rate", val=1e-3)
        data = g.cells.centroids[:, 0] + g.cells.centroids[:, 2]

        pl = pv.Plotter(off_screen=True)
        plot_cell_data(g, data, label="pressure", plotter=pl,
                       show_edges=True)
        plot_well(g, w, pl)
        out = _png("cells_and_well.png")
        pl.screenshot(out)
        pl.close()
        assert os.path.getsize(out) > 5000, "image looks empty"
        print(f"ok test_plot_cell_data_writes_png -> {out}")

    def test_plot_grid_subset():
        """Plotting a cell subset draws only those cells."""
        g = compute_geometry(cart_grid([6, 6, 2]))
        sel = np.arange(0, g.cells.num, 2)
        mesh = to_pyvista(g, sel)
        assert mesh.n_cells == sel.size
        out = plot_grid(g, cells=sel, filename=_png("subset.png"), show=False)
        assert os.path.getsize(out) > 3000
        print("ok test_plot_grid_subset")

    test_to_pyvista_cartesian()
    test_to_pyvista_corner_point()
    test_to_pyvista_2d()
    test_plot_cell_data_writes_png()
    test_plot_grid_subset()
    print(f"\nAll tests passed!  (images in {OUT})")
