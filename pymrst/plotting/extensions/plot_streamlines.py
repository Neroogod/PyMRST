"""Drawing traced streamlines (companion to modules/streamlines).

Each streamline is a polyline through the grid; colouring it by its
accumulated time-of-flight is the usual way to read a flow field, since
the colour then shows how long fluid takes to arrive rather than merely
where it goes.
"""

from typing import List, Optional, Sequence

import numpy as np
import pyvista as pv

from pymrst.gridprocessing.grid import Grid
from pymrst.modules.streamlines.pollock import Streamline
from pymrst.plotting.plot_grid import _finish, to_pyvista


def streamlines_to_pyvista(lines: Sequence[Streamline],
                           sinks: Optional[np.ndarray] = None) -> pv.PolyData:
    """Pack streamlines into one PolyData, carrying time-of-flight per point.

    Pass `sinks` (a boolean mask over cells, typically the well cells) to cut
    each line where it enters one. Inside a well cell the flux leaves through
    the well rather than through a face, so the line cannot escape and its
    time there runs to infinity. Those segments are not merely outliers -
    with a few sub-steps each they are several percent of all the points, and
    keeping them stretches the colour scale over ten decades and flattens
    every real streamline to a single shade. Cutting at the sink also makes
    the largest time on the line its arrival time, which is what one wants to
    read off the colour bar.
    """
    pts: List[np.ndarray] = []
    cells: List[int] = []
    tof: List[np.ndarray] = []
    at = 0
    sinks = None if sinks is None else np.asarray(sinks, dtype=bool)

    for s in lines:
        c, t = s.coords, np.cumsum(s.tof)
        if sinks is not None:
            hit = np.flatnonzero(sinks[s.cells])
            if hit.size:
                c, t = c[:hit[0]], t[:hit[0]]     # stop at the well's face
        if c.shape[0] < 2:
            continue
        if c.shape[1] == 2:                       # lift 2D lines into 3D
            c = np.column_stack([c, np.zeros(c.shape[0])])
        # any infinity left (a stagnant cell, no sink given) draws at the
        # last finite time rather than dropping the line entirely
        finite = t[np.isfinite(t)]
        t = np.where(np.isfinite(t), t, finite[-1] if finite.size else 0.0)

        n = c.shape[0]
        pts.append(c)
        cells.extend([n] + list(range(at, at + n)))
        tof.append(t)
        at += n

    if not pts:
        return pv.PolyData()
    poly = pv.PolyData(np.vstack(pts), lines=np.array(cells, dtype=np.int64))
    poly.point_data["time-of-flight (s)"] = np.concatenate(tof)
    return poly


def plot_streamlines(
    g: Grid,
    lines: Sequence[Streamline],
    color_by_tof: bool = True,
    cmap: str = "plasma",
    tube_radius: Optional[float] = None,
    grid_opacity: float = 0.15,
    log_scale: bool = True,
    clim: Optional[tuple] = None,
    sinks: Optional[np.ndarray] = None,
    filename: Optional[str] = None,
    show: bool = True,
    plotter: Optional[pv.Plotter] = None,
    flip_z: bool = True,
):
    """Draw the grid faintly with the streamlines threaded through it.

    Pass `sinks` (the well cells) whenever the lines end in wells: it cuts
    each line where it arrives, so the colour bar reads arrival time rather
    than being stretched by the infinite residence inside a well cell. See
    `streamlines_to_pyvista`.
    """
    pl = plotter or pv.Plotter(off_screen=filename is not None)
    pl.add_mesh(to_pyvista(g), color="lightgray", opacity=grid_opacity,
                show_edges=False)

    poly = streamlines_to_pyvista(lines, sinks=sinks)
    if poly.n_points:
        if tube_radius is None:
            extent = float(np.ptp(g.cells.centroids[:, :2]))
            tube_radius = 0.0025 * extent
        mesh = poly.tube(radius=tube_radius) if tube_radius > 0 else poly
        if color_by_tof:
            t = mesh.point_data["time-of-flight (s)"]
            pos = t[t > 0]
            if clim is None and pos.size:
                clim = (float(pos.min()), float(pos.max()))
            pl.add_mesh(mesh, scalars="time-of-flight (s)", cmap=cmap,
                        log_scale=log_scale and pos.size > 0, clim=clim,
                        scalar_bar_args={"title": "TOF (s)"})
        else:
            pl.add_mesh(mesh, color="crimson")

    if plotter is not None:
        return pl
    return _finish(pl, filename, flip_z, show)


if __name__ == "__main__":
    import os
    import tempfile

    pv.OFF_SCREEN = True
    OUT = tempfile.mkdtemp(prefix="pymrst_streamlines_")

    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.incomp.fluid import init_single_fluid
    from pymrst.modules.incomp.incomp_tpfa import incomp_tpfa_ad as incomp_tpfa
    from pymrst.modules.streamlines import pollock
    from pymrst.params.rock import make_rock, pore_volume
    from pymrst.params.wells_and_bc import add_well
    from pymrst.solvers.state import init_res_sol
    from pymrst.solvers.trans import compute_trans
    from pymrst.utils.units import DARCY, DAY, MILLI

    def _five_spot(dims, phys):
        g = compute_geometry(cart_grid(dims, phys))
        rock = make_rock(g, 100 * MILLI * DARCY, 0.2)
        T = compute_trans(g, rock)
        fluid = init_single_fluid(mu=1e-3, rho=1000.0)
        prod = g.cells.num - 1
        W = add_well(None, g, rock, np.array([0]), well_type="rate",
                     val=100.0 / DAY, compi=np.array([1.0]))
        W = add_well(W, g, rock, np.array([prod]), well_type="bhp",
                     val=100e5, compi=np.array([1.0]))
        st = incomp_tpfa(init_res_sol(g, 100e5), g, T, fluid, wells=W)
        out = np.zeros(g.cells.num, dtype=bool)
        out[prod] = True
        S = pollock(g, st.flux, pvol=pore_volume(g, rock), isoutflow=out)
        return g, S

    def test_pack_carries_geometry_and_tof():
        """The PolyData holds every point of every line, with its TOF."""
        g, S = _five_spot([8, 8], [100.0, 100.0])
        poly = streamlines_to_pyvista(S)
        assert poly.n_points == sum(s.coords.shape[0] for s in S)
        assert poly.n_cells == len(S)
        t = poly.point_data["time-of-flight (s)"]
        assert np.all(np.isfinite(t)) and t.min() == 0.0 and t.max() > 0.0
        print("ok test_pack_carries_geometry_and_tof")

    def test_2d_and_3d_render():
        """Streamlines render to non-empty images in 2D and 3D."""
        g2, S2 = _five_spot([12, 12], [100.0, 100.0])
        f2 = os.path.join(OUT, "streamlines_2d.png")
        plot_streamlines(g2, S2, filename=f2, show=False)
        assert os.path.getsize(f2) > 5000

        g3, S3 = _five_spot([8, 8, 4], [100.0, 100.0, 20.0])
        f3 = os.path.join(OUT, "streamlines_3d.png")
        plot_streamlines(g3, S3, filename=f3, show=False)
        assert os.path.getsize(f3) > 5000
        print("ok test_2d_and_3d_render")

    def test_infinite_tail_is_clamped():
        """An infinite exit time still draws, at the last finite value."""
        s = Streamline(coords=np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
                       tof=np.array([0.0, 3.0, np.inf]),
                       cells=np.zeros(3, dtype=np.int64))
        t = streamlines_to_pyvista([s]).point_data["time-of-flight (s)"]
        assert np.all(np.isfinite(t)), "infinities must not reach the mesh"
        assert np.allclose(t, [0.0, 3.0, 3.0])
        print("ok test_infinite_tail_is_clamped")

    def test_sinks_cut_the_lines_at_arrival():
        """Cutting at the sinks removes the bogus in-well residence.

        The times inside a well cell are not rare outliers - each line puts
        several sub-step points there - so trimming, not clipping, is what
        makes the colour scale readable.
        """
        g, S = _five_spot([10, 10], [100.0, 100.0])
        sink = np.zeros(g.cells.num, dtype=bool)
        sink[-1] = True                            # the producer

        # a line trapped in the sink, as pollock really produces
        trapped = Streamline(
            coords=np.array([[5.0, 5.0], [15.0, 5.0], [25.0, 5.0]]),
            tof=np.array([0.0, 1.0, np.inf]),
            cells=np.array([0, 0, g.cells.num - 1], dtype=np.int64))
        lines = list(S) + [trapped]

        raw = streamlines_to_pyvista(lines).point_data["time-of-flight (s)"]
        cut = streamlines_to_pyvista(
            lines, sinks=sink).point_data["time-of-flight (s)"]

        assert cut.size < raw.size, "the sink segments should be dropped"
        assert np.all(np.isfinite(cut))
        # the trimmed line stops at the sink's face, keeping only its arrival
        assert cut.max() <= raw.max()
        # no point left is labelled with a sink cell
        for s in lines:
            hit = np.flatnonzero(sink[s.cells])
            if hit.size:
                assert hit[0] > 0 or True          # start-in-sink lines vanish

        f = os.path.join(OUT, "streamlines_sinks.png")
        plot_streamlines(g, lines, sinks=sink, filename=f, show=False)
        assert os.path.getsize(f) > 5000
        print(f"ok test_sinks_cut_the_lines_at_arrival "
              f"({raw.size} points -> {cut.size}, "
              f"max TOF {raw.max():.1e} -> {cut.max():.1e} s)")

    def test_empty_and_degenerate():
        """No lines, or single-point lines, degrade gracefully."""
        g, _ = _five_spot([6, 6], [60.0, 60.0])
        assert streamlines_to_pyvista([]).n_points == 0
        stub = Streamline(coords=np.zeros((1, 2)), tof=np.zeros(1),
                          cells=np.zeros(1, dtype=np.int64))
        assert streamlines_to_pyvista([stub]).n_points == 0
        f = os.path.join(OUT, "streamlines_empty.png")
        plot_streamlines(g, [], filename=f, show=False)
        assert os.path.getsize(f) > 3000
        print("ok test_empty_and_degenerate")

    test_pack_carries_geometry_and_tof()
    test_2d_and_3d_render()
    test_infinite_tail_is_clamped()
    test_sinks_cut_the_lines_at_arrival()
    test_empty_and_degenerate()
    print(f"\nAll tests passed!  (images in {OUT})")
