"""Visualization for CO2 storage: caprock, plumes, and structural traps.

Companions to co2lab's plotting: the top-surface grid drawn as a 3D
surface coloured by depth, the CO2 plume thickness under the caprock, and
the traps found by trap_analysis coloured by trap index or filled to
their spill depth.
"""

from typing import Optional

import numpy as np
import pyvista as pv

from pymrst.modules.co2lab_spillpoint.trapping import TrapAnalysis, volumes_of_traps
from pymrst.modules.co2lab_ve.ve import TopSurfaceGrid
from pymrst.plotting.plot_grid import _finish, to_pyvista


def _surface(ts: TopSurfaceGrid, z: Optional[np.ndarray] = None,
             exaggeration: float = 1.0) -> pv.UnstructuredGrid:
    """The top-surface grid lifted to depth z (default: the caprock).

    `ts.g` is the *projected* areal grid (z = 0), so the mesh must be
    lifted onto the caprock before anything is drawn on it. Structural
    relief is usually tiny next to the lateral extent, so `exaggeration`
    scales the depth variation about its mean to make domes visible.
    """
    mesh = to_pyvista(ts.g)
    zc = ts.z if z is None else np.asarray(z, float).ravel()
    # lift each node to the mean depth of the cells sharing it
    pts = mesh.points.copy()
    acc = np.zeros(pts.shape[0])
    cnt = np.zeros(pts.shape[0])
    g = ts.g
    for c in range(g.cells.num):
        faces = g.cells.faces[g.cells.facePos[c]:g.cells.facePos[c + 1]]
        nodes = np.unique(np.concatenate([
            g.faces.nodes[g.faces.nodePos[f]:g.faces.nodePos[f + 1]]
            for f in faces
        ]))
        acc[nodes] += zc[c]
        cnt[nodes] += 1.0
    zn = acc / np.maximum(cnt, 1.0)
    if exaggeration != 1.0:
        zn = zn.mean() + exaggeration * (zn - zn.mean())
    pts[:, 2] = zn
    mesh.points = pts
    return mesh


def plot_caprock(ts: TopSurfaceGrid, cmap: str = "terrain_r",
                 exaggeration: float = 10.0,
                 filename: Optional[str] = None, show: bool = True,
                 plotter: Optional[pv.Plotter] = None):
    """Draw the caprock surface coloured by depth."""
    mesh = _surface(ts, exaggeration=exaggeration)
    mesh.cell_data["depth (m)"] = ts.z
    pl = plotter or pv.Plotter(off_screen=filename is not None)
    pl.add_mesh(mesh, scalars="depth (m)", cmap=cmap, show_edges=False)
    if plotter is not None:
        return pl
    return _finish(pl, filename, True, show)


def plot_plume(ts: TopSurfaceGrid, h: np.ndarray, cmap: str = "inferno",
               min_h: float = 1e-3, exaggeration: float = 10.0,
               filename: Optional[str] = None,
               show: bool = True, plotter: Optional[pv.Plotter] = None):
    """Draw the caprock in grey with the CO2 plume painted onto it."""
    h = np.asarray(h, float).ravel()
    surf = _surface(ts, exaggeration=exaggeration)
    pl = plotter or pv.Plotter(off_screen=filename is not None)
    pl.add_mesh(surf, color="lightgray", opacity=0.35)

    sel = np.flatnonzero(h > min_h)
    if sel.size:
        plume = surf.extract_cells(sel)
        plume.cell_data["plume height (m)"] = h[sel]
        # lift slightly off the caprock so it is not z-fought
        plume.points[:, 2] -= 0.01 * float(np.ptp(surf.points[:, 2]) or 1.0)
        pl.add_mesh(plume, scalars="plume height (m)", cmap=cmap)
    if plotter is not None:
        return pl
    return _finish(pl, filename, True, show)


def plot_traps(ts: TopSurfaceGrid, res: TrapAnalysis,
               poro: Optional[np.ndarray] = None,
               show_catchments: bool = False,
               exaggeration: float = 10.0,
               filename: Optional[str] = None, show: bool = True,
               plotter: Optional[pv.Plotter] = None):
    """Draw the caprock with its structural traps, optionally with each
    trap's catchment and its capacity labelled at the crest."""
    surf = _surface(ts, exaggeration=exaggeration)
    pl = plotter or pv.Plotter(off_screen=filename is not None)

    if show_catchments:
        cat = surf.copy()
        cat.cell_data["catchment"] = res.trap_regions.astype(float)
        pl.add_mesh(cat, scalars="catchment", cmap="tab20", opacity=0.6,
                    show_scalar_bar=False)
    else:
        pl.add_mesh(surf, color="lightgray", opacity=0.4)

    zrel = float(np.ptp(surf.points[:, 2]) or 1.0)
    sel = np.flatnonzero(res.traps > 0)
    if sel.size:
        traps = surf.extract_cells(sel)
        traps.cell_data["trap"] = res.traps[sel].astype(float)
        traps.points[:, 2] -= 0.01 * zrel
        pl.add_mesh(traps, scalars="trap", cmap="tab20", show_edges=False,
                    scalar_bar_args={"title": "trap"})

    if poro is not None and res.n_traps:
        vols = volumes_of_traps(ts, res, poro=poro)
        zs = surf.cell_centers().points[:, 2]
        for i in range(1, res.n_traps + 1):
            cells = np.flatnonzero(res.traps == i)
            crest = cells[np.argmin(ts.z[cells])]
            c = ts.g.cells.centroids[crest]
            pos = (float(c[0]), float(c[1]), float(zs[crest] - 0.15 * zrel))
            pl.add_point_labels([pos], [f"{vols[i - 1] / 1e6:.2f} Mm3"],
                                font_size=13, point_size=1,
                                always_visible=True)
    if plotter is not None:
        return pl
    return _finish(pl, filename, True, show)


if __name__ == "__main__":
    import os
    import tempfile

    pv.OFF_SCREEN = True
    OUT = tempfile.mkdtemp(prefix="pymrst_ve_plots_")

    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.co2lab_spillpoint.trapping import trap_analysis
    from pymrst.modules.co2lab_ve.ve import top_surface_grid

    def _dome_surface():
        nx, ny = 50, 50
        g = cart_grid([nx, ny, 1], [1000.0, 1000.0, 20.0])
        x, y = g.nodes.coords[:, 0], g.nodes.coords[:, 1]
        z = (1000.0 - 0.015 * x - 0.01 * y
             - 15.0 * np.exp(-(((x - 250.0) / 80.0) ** 2
                               + ((y - 250.0) / 80.0) ** 2))
             - 18.0 * np.exp(-(((x - 500.0) / 70.0) ** 2
                               + ((y - 750.0) / 70.0) ** 2)))
        g.nodes.coords[:, 2] += z
        return top_surface_grid(compute_geometry(g))

    def test_surface_geometry():
        """The rendered caprock carries the right depths and cell count."""
        ts = _dome_surface()
        mesh = _surface(ts)
        assert mesh.n_cells == ts.g.cells.num
        # the mesh must sit ON the caprock, not on the projected plane
        assert mesh.points[:, 2].min() >= ts.z.min() - 1e-9
        assert mesh.points[:, 2].max() <= ts.z.max() + 1e-9
        assert mesh.points[:, 2].min() > 900.0, "surface not lifted to caprock"
        # vertical exaggeration stretches relief about its mean
        ex = _surface(ts, exaggeration=5.0)
        assert np.isclose(ex.points[:, 2].mean(), mesh.points[:, 2].mean())
        assert np.ptp(ex.points[:, 2]) > 4.9 * np.ptp(mesh.points[:, 2])
        print("ok test_surface_geometry")

    def test_plot_caprock_and_traps():
        """Caprock, traps, and catchments all render to non-empty images."""
        ts = _dome_surface()
        res = trap_analysis(ts)
        assert res.n_traps == 2, res.n_traps

        f1 = os.path.join(OUT, "caprock.png")
        plot_caprock(ts, filename=f1, show=False)
        assert os.path.getsize(f1) > 5000

        poro = np.full(ts.g.cells.num, 0.2)
        f2 = os.path.join(OUT, "traps.png")
        plot_traps(ts, res, poro=poro, filename=f2, show=False)
        assert os.path.getsize(f2) > 5000

        f3 = os.path.join(OUT, "catchments.png")
        plot_traps(ts, res, show_catchments=True, filename=f3, show=False)
        assert os.path.getsize(f3) > 5000
        print("ok test_plot_caprock_and_traps")

    def test_plot_plume():
        """A synthetic plume renders; empty plumes degrade gracefully."""
        ts = _dome_surface()
        res = trap_analysis(ts)
        # fill the traps to their spill depth -> a physical plume
        h = np.zeros(ts.g.cells.num)
        for i in range(1, res.n_traps + 1):
            sel = res.traps == i
            h[sel] = np.minimum(res.trap_z[i - 1] - ts.z[sel], ts.H[sel])
        assert h.max() > 1.0

        f = os.path.join(OUT, "plume.png")
        plot_plume(ts, h, filename=f, show=False)
        assert os.path.getsize(f) > 5000

        f0 = os.path.join(OUT, "plume_empty.png")
        plot_plume(ts, np.zeros(ts.g.cells.num), filename=f0, show=False)
        assert os.path.getsize(f0) > 3000
        print("ok test_plot_plume")

    test_surface_geometry()
    test_plot_caprock_and_traps()
    test_plot_plume()
    print(f"\nAll tests passed!  (images in {OUT})")
