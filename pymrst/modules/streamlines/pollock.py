"""Semi-analytic streamline tracing (MRST streamlines/pollock.m).

Pollock's method. Inside a logically Cartesian cell the velocity is taken
to vary linearly and independently in each of the d local coordinates, so
the trajectory through the cell can be integrated in closed form rather
than stepped numerically:

    u_i(xi) = v_i0 + xi * (v_i1 - v_i0)

with v_i0, v_i1 the (pore-volume-normalised) fluxes through the low- and
high-side faces in direction i. The exit time in each direction follows
analytically, the smallest one wins, and the streamline is handed to the
neighbouring cell across that face. Because the time is exact, the
accumulated time along the trajectory is a genuine time-of-flight, which
is what makes this the standard companion to flow diagnostics.

The local coordinate is the unit cube [0,1]^d; a local point is mapped to
physical space by multilinear interpolation of the cell's 2^d corners.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import numpy as np

from pymrst.gridprocessing.grid import Grid

_TOL = 100.0 * np.finfo(float).eps


@dataclass
class Streamline:
    """One traced streamline.

    Careful with the last cell. A line that ends in a well cell has an
    *infinite* final tof, because in a well cell nearly all the flux leaves
    through the well rather than through a face, so the line cannot escape
    across any face and its exit time is infinite. `isoutflow` stops the
    line only after that step has been taken (this is what pollock.m does).
    So `time[-1]` is not the breakthrough time - `arrival_time` is.
    """

    coords: np.ndarray       # (n, d) physical coordinates along the curve
    tof: np.ndarray          # (n,) time-of-flight increment at each point
    cells: np.ndarray        # (n,) cell each segment lies in

    @property
    def time(self) -> np.ndarray:
        """Cumulative time-of-flight at each point (s)."""
        return np.cumsum(self.tof)

    def arrival_time(self, sinks: np.ndarray) -> float:
        """Time to reach the first of `sinks` (a boolean mask over cells).

        The time at the last point still outside a sink, i.e. the moment the
        line crosses into the well cell. This is the breakthrough time; it
        excludes the infinite residence inside the sink itself.
        """
        sinks = np.asarray(sinks, dtype=bool)
        hit = np.flatnonzero(sinks[self.cells])
        if hit.size == 0:
            return float(self.time[-1])
        first = int(hit[0])
        return 0.0 if first == 0 else float(self.time[first - 1])


def cell_nodes(g: Grid) -> np.ndarray:
    """The 2^d corners of every cell, in tensor-product (bit) order.

    Corner j carries the high-side face in direction i when bit i of j is
    set, so corner j is the unique node shared by the d faces the bits of
    j select. Requires direction tags, i.e. a logically Cartesian grid.
    """
    if g.cells.tags is None:
        raise ValueError("pollock needs a logically Cartesian grid "
                         "(cells.tags is None)")
    d = g.griddim
    nc = g.cells.num
    out = np.zeros((nc, 2 ** d), dtype=np.int64)

    # nodes of each of the 2d tagged faces, per cell
    face_nodes = [[None] * (2 * d) for _ in range(nc)]
    for c in range(nc):
        lo, hi = g.cells.facePos[c], g.cells.facePos[c + 1]
        for f, t in zip(g.cells.faces[lo:hi], g.cells.tags[lo:hi]):
            face_nodes[c][int(t)] = set(
                int(n) for n in
                g.faces.nodes[g.faces.nodePos[f]:g.faces.nodePos[f + 1]]
            )

    for c in range(nc):
        for j in range(2 ** d):
            shared = None
            for i in range(d):
                tag = 2 * i + ((j >> i) & 1)      # low or high face in dim i
                ns = face_nodes[c][tag]
                if ns is None:
                    raise ValueError(f"cell {c} is missing face tag {tag}")
                shared = ns if shared is None else (shared & ns)
            if len(shared) != 1:
                raise ValueError(
                    f"cell {c}, corner {j}: expected 1 shared node, "
                    f"got {len(shared)} (not a hexahedral/quadrilateral cell)"
                )
            out[c, j] = shared.pop()
    return out


def interior_fluxes(g: Grid, flux: np.ndarray,
                    pvol: Optional[np.ndarray] = None,
                    reverse: bool = False) -> np.ndarray:
    """Face fluxes gathered per cell and direction, divided by pore volume.

    Returns (num_cells, 2d), column 2i / 2i+1 being the low- / high-side
    face in direction i. Fluxes keep their global sign (positive along the
    +axis), which is what makes u_i(xi) above a velocity.
    """
    flux = np.asarray(flux, dtype=np.float64)
    if flux.ndim > 1:
        flux = flux.sum(axis=1)               # total flux of a multiphase state
    if reverse:
        flux = -flux
    if g.cells.tags is None:
        raise ValueError("pollock needs a logically Cartesian grid "
                         "(cells.tags is None)")
    pv = (np.ones(g.cells.num) if pvol is None
          else np.asarray(pvol, dtype=np.float64).ravel())
    if not np.all(pv > 0):
        raise ValueError("pore volume must be strictly positive in all cells")

    d = g.griddim
    cell_no = np.repeat(np.arange(g.cells.num), np.diff(g.cells.facePos))
    out = np.zeros((g.cells.num, 2 * d))
    np.add.at(out, (cell_no, g.cells.tags), flux[g.cells.faces])
    return out / pv[:, None]


def streamline_neighbors(g: Grid) -> np.ndarray:
    """Neighbour cell across each tagged face, -1 on the boundary.

    (num_cells, 2d), laid out like `interior_fluxes`.
    """
    if g.cells.tags is None:
        raise ValueError("pollock needs a logically Cartesian grid "
                         "(cells.tags is None)")
    d = g.griddim
    cell_no = np.repeat(np.arange(g.cells.num), np.diff(g.cells.facePos))
    nb = g.faces.neighbors[g.cells.faces]          # (n_half_faces, 2)
    other = np.where(nb[:, 0] == cell_no, nb[:, 1], nb[:, 0])
    out = np.full((g.cells.num, 2 * d), -1, dtype=np.int64)
    out[cell_no, g.cells.tags] = other
    return out


def _global_coordinate(g: Grid, cn: np.ndarray, c: np.ndarray,
                       p: np.ndarray) -> np.ndarray:
    """Multilinear map from local p in [0,1]^d to physical coordinates."""
    d = g.griddim
    w = np.ones((p.shape[0], 2 ** d))
    j = np.arange(2 ** d)
    for i in range(d):
        high = ((j >> i) & 1).astype(bool)
        w[:, high] *= p[:, [i]]
        w[:, ~high] *= 1.0 - p[:, [i]]
    corners = g.nodes.coords[cn[c]]                # (n, 2^d, d)
    return np.einsum("nj,njk->nk", w, corners[:, :, :d])


def _compute_time(xi: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Time to reach xi = 0 or xi = 1 given end-face velocities v = [v0, v1].

    Linear velocity gives t = log(ue/ui) / (v1 - v0); a constant velocity
    degenerates to t = ds / ui. Infinite when the streamline cannot leave
    through either face in this direction.
    """
    du = v[:, 1] - v[:, 0]
    ui = v[:, 0] + xi * du
    ue = np.where(ui < 0, v[:, 0], v[:, 1])
    with np.errstate(divide="ignore", invalid="ignore"):
        arg = ue / ui
    t = np.full(xi.shape, np.inf)

    lin = np.abs(du) > _TOL * np.abs(v[:, 0])
    with np.errstate(divide="ignore", invalid="ignore"):
        t[lin] = np.log(arg[lin]) / du[lin]
        ds = np.where(ui > 0, 1.0 - xi, -xi)
        t[~lin] = ds[~lin] / ui[~lin]
    t[(arg <= 0) | np.isnan(arg)] = np.inf
    return t


def _compute_position(xi: np.ndarray, v: np.ndarray, t: np.ndarray):
    """Local position at time t, and which face (0 = low, 1 = high) it heads for."""
    du = v[:, 1] - v[:, 0]
    ui = v[:, 0] + xi * du
    side = (ui >= 0).astype(np.int64)          # 0 -> low face, 1 -> high face
    x = np.full(xi.shape, np.inf)

    lin = np.abs(du) > _TOL * np.abs(v[:, 0])
    with np.errstate(over="ignore", invalid="ignore"):
        x[lin] = xi[lin] + (ui[lin] * np.exp(du[lin] * t[lin])
                            - ui[lin]) / du[lin]
    x[~lin] = xi[~lin] + v[~lin, 0] * t[~lin]
    stuck = ~lin & ~np.isfinite(t)
    x[stuck] = xi[stuck]
    return x, side


def pollock(
    g: Grid,
    flux: np.ndarray,
    positions: Optional[Union[Sequence[int], np.ndarray]] = None,
    substeps: int = 5,
    maxsteps: int = 1000,
    reverse: bool = False,
    pvol: Optional[np.ndarray] = None,
    isoutflow: Optional[np.ndarray] = None,
) -> List[Streamline]:
    """Trace streamlines through a logically Cartesian grid.

    `positions` is either cell indices (each streamline then starts at the
    cell's local centre) or an (N, 1+d) array whose first column is the
    cell and whose remaining columns are local coordinates in [0,1]. It
    defaults to one streamline started at the centre of every cell.

    Pass `pvol` (pore volumes) to make the reported time-of-flight
    physical; without it the "time" is per unit bulk volume. `reverse`
    traces upstream, which is how one finds where fluid in a cell came
    from.
    """
    d = g.griddim
    if positions is None:
        pos_c = np.arange(g.cells.num, dtype=np.int64)
        pos_x = np.full((g.cells.num, d), 0.5)
    else:
        pos = np.atleast_2d(np.asarray(positions, dtype=np.float64))
        if pos.shape[0] == 1 and pos.shape[1] not in (1, 1 + d):
            pos = pos.T                        # a bare list of cell indices
        if pos.shape[1] == 1:
            pos_c = pos[:, 0].astype(np.int64)
            pos_x = np.full((pos.shape[0], d), 0.5)
        elif pos.shape[1] == 1 + d:
            pos_c = pos[:, 0].astype(np.int64)
            pos_x = pos[:, 1:].copy()
        else:
            raise ValueError(
                f"expected local positions of width 1 or {1 + d}, "
                f"got {pos.shape[1]}"
            )

    n = pos_c.size
    f = interior_fluxes(g, flux, pvol, reverse)
    nb = streamline_neighbors(g)
    cn = cell_nodes(g)
    outflow = (np.zeros(g.cells.num, dtype=bool) if isoutflow is None
               else np.asarray(isoutflow, dtype=bool).ravel())

    # Each streamline collects its own points; they die at different times.
    xyz: List[List[np.ndarray]] = [[] for _ in range(n)]
    tof: List[List[float]] = [[] for _ in range(n)]
    cid: List[List[int]] = [[] for _ in range(n)]

    start = _global_coordinate(g, cn, pos_c, pos_x)
    for k in range(n):
        xyz[k].append(start[k])
        tof[k].append(0.0)
        cid[k].append(int(pos_c[k]))

    active = np.arange(n)
    step = 1
    while active.size and step <= maxsteps:
        c = pos_c[active]
        x = pos_x[active]
        v = f[c]                                       # (m, 2d)

        # exit time in each direction; the earliest one is taken
        T = np.column_stack([_compute_time(x[:, i], v[:, 2 * i:2 * i + 2])
                             for i in range(d)])
        t_exit = T.min(axis=1)
        direction = T.argmin(axis=1)

        # trajectory inside the cell, sampled at `substeps` points
        m = active.size
        traj = np.zeros((m, d, substeps))
        side = np.zeros((m, d), dtype=np.int64)
        for s in range(1, substeps + 1):
            ts = t_exit * s / substeps
            for i in range(d):
                traj[:, i, s - 1], side[:, i] = _compute_position(
                    x[:, i], v[:, 2 * i:2 * i + 2], ts)

        for s in range(substeps):
            pts = _global_coordinate(g, cn, c, traj[:, :, s])
            for k, sl in enumerate(active):
                xyz[sl].append(pts[k])
                tof[sl].append(t_exit[k] / substeps)
                cid[sl].append(int(c[k]))

        # hand the streamline to the neighbour across the exit face
        exit_side = side[np.arange(m), direction]      # 0 = low, 1 = high
        col = 2 * direction + exit_side
        nxt = nb[c, col]

        stop = (nxt < 0) | ~np.isfinite(T).any(axis=1) | outflow[c]
        pos_x[active] = traj[:, :, -1]
        # crossing a face flips the local coordinate in that direction
        keep = ~stop
        ak = active[keep]
        pos_x[ak, direction[keep]] = 1.0 - exit_side[keep]
        pos_c[ak] = nxt[keep]

        active = ak
        step += substeps

    return [Streamline(coords=np.asarray(xyz[k]),
                       tof=np.asarray(tof[k]),
                       cells=np.asarray(cid[k], dtype=np.int64))
            for k in range(n)]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.incomp.incomp_tpfa import incomp_tpfa_ad as incomp_tpfa
    from pymrst.modules.incomp.fluid import init_single_fluid
    from pymrst.params.rock import make_rock, pore_volume
    from pymrst.params.wells_and_bc import add_well
    from pymrst.solvers.state import init_res_sol
    from pymrst.solvers.trans import compute_trans
    from pymrst.utils.units import DARCY, DAY, MILLI

    def _uniform_flow_grid():
        """Unit-speed flow in +x on a 10x1 grid: flux 1 through every x-face."""
        g = compute_geometry(cart_grid([10, 1], [10.0, 1.0]))
        flux = np.zeros(g.faces.num)
        # x-faces come first in a tensor grid
        nx_faces = 11
        flux[:nx_faces] = 1.0
        return g, flux

    def test_uniform_flow_straight_and_timed():
        """Uniform +x flow gives a straight line with exact time-of-flight."""
        g, flux = _uniform_flow_grid()
        pvol = np.ones(g.cells.num)             # unit pore volume per cell
        S = pollock(g, flux, positions=[0], pvol=pvol)
        assert len(S) == 1
        s = S[0]
        # starts at the centre of cell 0, exits the far end of the domain
        assert np.allclose(s.coords[0], [0.5, 0.5])
        assert np.isclose(s.coords[-1, 0], 10.0, atol=1e-9)
        assert np.allclose(s.coords[:, 1], 0.5)          # no drift in y
        # x advances monotonically
        assert np.all(np.diff(s.coords[:, 0]) > -1e-12)
        # cell has pore volume 1 and throughput 1, so 1 s per cell; the
        # streamline starts mid-cell, so 0.5 + 9 = 9.5 s to leave the domain
        assert np.isclose(s.time[-1], 9.5, rtol=1e-10), s.time[-1]
        assert s.cells[0] == 0 and s.cells[-1] == 9
        print("ok test_uniform_flow_straight_and_timed")

    def test_pore_volume_scales_time():
        """Doubling pore volume doubles time-of-flight, path unchanged."""
        g, flux = _uniform_flow_grid()
        a = pollock(g, flux, positions=[0], pvol=np.ones(g.cells.num))[0]
        b = pollock(g, flux, positions=[0], pvol=2 * np.ones(g.cells.num))[0]
        assert np.allclose(a.coords, b.coords)
        assert np.isclose(b.time[-1], 2 * a.time[-1], rtol=1e-10)
        print("ok test_pore_volume_scales_time")

    def test_reverse_traces_upstream():
        """Reversing the field traces back to the inflow boundary."""
        g, flux = _uniform_flow_grid()
        s = pollock(g, flux, positions=[9], reverse=True,
                    pvol=np.ones(g.cells.num))[0]
        assert np.isclose(s.coords[-1, 0], 0.0, atol=1e-9)
        assert np.isclose(s.time[-1], 9.5, rtol=1e-10)
        assert s.cells[-1] == 0
        print("ok test_reverse_traces_upstream")

    def test_quarter_five_spot_symmetry():
        """Quarter five-spot: streamlines are symmetric about the diagonal
        and every one of them ends at the producer."""
        n = 16
        g = compute_geometry(cart_grid([n, n], [100.0, 100.0]))
        rock = make_rock(g, 100 * MILLI * DARCY, 0.2)
        fluid = init_single_fluid(mu=1e-3, rho=1000.0)
        T = compute_trans(g, rock)

        inj, prod = 0, g.cells.num - 1
        W = add_well(None, g, rock, np.array([inj]), well_type="rate",
                     val=100.0 / DAY, compi=np.array([1.0]))
        W = add_well(W, g, rock, np.array([prod]), well_type="bhp",
                     val=100e5, compi=np.array([1.0]))
        st = incomp_tpfa(init_res_sol(g, 100e5), g, T, fluid, wells=W)

        pv = pore_volume(g, rock)
        outflow = np.arange(g.cells.num) == prod
        # a line in each cell of the first row, and in its mirror column.
        # Cell 0 is left out: it sits on the diagonal separatrix, where the
        # path is unstable to the solver's round-off. It is checked apart.
        row = np.arange(1, n)
        col = np.arange(1, n) * n
        Srow = pollock(g, st.flux, positions=row, pvol=pv, isoutflow=outflow)
        Scol = pollock(g, st.flux, positions=col, pvol=pv, isoutflow=outflow)

        # the flow field is invariant under (x, y) -> (y, x), so the two
        # families of streamlines must be reflections of one another
        for a, b in zip(Srow, Scol):
            assert np.allclose(a.coords, b.coords[:, ::-1], atol=1e-9), \
                "quarter five-spot lost its diagonal symmetry"
            assert np.isclose(a.time[-1], b.time[-1], rtol=1e-9)

        # everything drains to the producer
        for s in Srow:
            assert s.cells[-1] == prod, s.cells[-1]
            assert s.time[-1] > 0

        # the line launched at the injector runs up the diagonal to the
        # producer, so x and y stay equal along it
        diag = pollock(g, st.flux, positions=[inj], pvol=pv,
                       isoutflow=outflow)[0]
        assert diag.cells[-1] == prod
        assert np.abs(diag.coords[:, 0] - diag.coords[:, 1]).max() < 0.1

        # the diagonal is the shortest path, so it breaks through first
        assert diag.time[-1] < min(s.time[-1] for s in Srow)
        print(f"ok test_quarter_five_spot_symmetry "
              f"({len(Srow)} lines, breakthrough {diag.time[-1] / DAY:.1f} d, "
              f"slowest {max(s.time[-1] for s in Srow) / DAY:.1f} d)")

    def test_arrival_time_excludes_the_sink_cell():
        """A line ending in a well cell has an infinite last step.

        Almost all the flux leaves a well cell through the well, so the
        line cannot escape across a face and its exit time is infinite.
        arrival_time must report the finite time to *reach* the well.
        """
        g, flux = _uniform_flow_grid()
        pvol = np.ones(g.cells.num)
        # make the last cell a sink: cut the flux out of its far face
        flux[10] = 0.0
        sinks = np.zeros(g.cells.num, dtype=bool)
        sinks[9] = True

        s = pollock(g, flux, positions=[0], pvol=pvol, isoutflow=sinks)[0]
        assert not np.isfinite(s.time[-1]), "sink cell should trap the line"
        # 0.5 s to leave cell 0, then 1 s per cell up to the face of cell 9
        assert np.isclose(s.arrival_time(sinks), 8.5, rtol=1e-10), \
            s.arrival_time(sinks)
        # with no sink to reach, arrival_time falls back to the total
        none = np.zeros(g.cells.num, dtype=bool)
        assert s.arrival_time(none) == s.time[-1]
        print("ok test_arrival_time_excludes_the_sink_cell")

    def test_heterogeneous_1d_tof_is_exact():
        """With varying porosity, TOF is the analytic sum of pv_c / q.

        A streamline entering cell c at its inlet face spends pv_c / q in
        it, so the time to cross a column of cells is the running sum of
        the pore volumes divided by the (constant) throughput. This is the
        check that per-cell pore volumes enter the time correctly, which a
        uniform-porosity test cannot see.
        """
        nc = 12
        g = compute_geometry(cart_grid([nc, 1], [float(nc), 1.0]))
        q = 2.0
        flux = np.zeros(g.faces.num)
        flux[:nc + 1] = q                       # x-faces come first

        rng = np.random.default_rng(7)
        pv = 1.0 + 4.0 * rng.random(nc)         # strongly varying pore volume

        # enter cell 0 at its inlet face (local x = 0) and cross the column
        s = pollock(g, flux, positions=np.array([[0, 0.0, 0.5]]), pvol=pv)[0]
        assert np.isclose(s.time[-1], pv.sum() / q, rtol=1e-12), (
            s.time[-1], pv.sum() / q)

        # arrival time at each cell boundary must match the partial sums
        arrive = np.cumsum(pv) / q
        for c in range(nc):
            # the last sample taken inside cell c is its exit point
            last = np.flatnonzero(s.cells == c)[-1]
            assert np.isclose(s.time[last], arrive[c], rtol=1e-12)
            assert np.isclose(s.coords[last, 0], c + 1.0, atol=1e-9)
        print("ok test_heterogeneous_1d_tof_is_exact")

    def test_cell_nodes_and_neighbors():
        """The corner map and neighbour map are self-consistent in 3D."""
        g = compute_geometry(cart_grid([3, 4, 2], [3.0, 4.0, 2.0]))
        cn = cell_nodes(g)
        assert cn.shape == (24, 8)
        assert (np.sort(cn, axis=1)[:, :-1] != np.sort(cn, axis=1)[:, 1:]).all()

        # corner 0 is the low corner, corner 7 the high corner
        lo = g.nodes.coords[cn[:, 0]]
        hi = g.nodes.coords[cn[:, 7]]
        assert np.all(hi > lo)
        assert np.allclose(0.5 * (lo + hi), g.cells.centroids)

        nb = streamline_neighbors(g)
        assert nb.shape == (24, 6)
        # a neighbour relation is reciprocal: crossing the high-x face and
        # then the low-x face of that cell brings you home
        for c in range(g.cells.num):
            e = nb[c, 1]
            if e >= 0:
                assert nb[e, 0] == c
        print("ok test_cell_nodes_and_neighbors")

    def test_3d_gravity_free_column():
        """Streamlines trace correctly through a 3D grid."""
        g = compute_geometry(cart_grid([4, 4, 4], [4.0, 4.0, 4.0]))
        nfx = 5 * 4 * 4
        flux = np.zeros(g.faces.num)
        flux[:nfx] = 1.0                       # uniform +x flow
        s = pollock(g, flux, positions=[0], pvol=np.ones(g.cells.num))[0]
        assert np.isclose(s.coords[-1, 0], 4.0, atol=1e-9)
        assert np.allclose(s.coords[:, 1], 0.5)
        assert np.allclose(s.coords[:, 2], 0.5)
        assert np.isclose(s.time[-1], 3.5, rtol=1e-10)
        print("ok test_3d_gravity_free_column")

    test_uniform_flow_straight_and_timed()
    test_pore_volume_scales_time()
    test_reverse_traces_upstream()
    test_cell_nodes_and_neighbors()
    test_3d_gravity_free_column()
    test_quarter_five_spot_symmetry()
    test_heterogeneous_1d_tof_is_exact()
    print("\nAll tests passed!")
