"""Vertical-equilibrium CO2 storage (MRST co2lab-ve).

A 3D columnar aquifer is collapsed onto its top surface: each column of
3D cells becomes one 2D cell carrying a thickness H and a top depth z.
Under vertical equilibrium the CO2 in a column segregates instantly into
a plume of height h below the caprock (a sharp interface), so the 2D
"saturation" sG is just a proxy for h (upscaledSat2height):

    h_max = sGmax H / (1 - s_wr)
    h     = (H sG - h_max s_gr) / (1 - s_gr - s_wr)

and the upscaled properties are (addVERelpermSharpInterface):

    krG = h krG_max / H
    krW = [H - h_max + (h_max - h) krW_max] / H
    pc  = (rho_w - rho_g) g h

The 2D flow problem uses vertically integrated permeability K H, pore
volume poro A H, and the top-surface depth for gravity, exactly as
CO2VEBlackOilTypeModel.setupOperators does. Solving in 2D makes
long-term migration studies orders of magnitude cheaper than 3D.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from pymrst.gridprocessing.geometry import compute_geometry
from pymrst.gridprocessing.grid import Cells, Faces, Grid, Nodes
from pymrst.params.gravity_const import NORM_GRAVITY
from pymrst.params.rock import Rock
from pymrst.utils.adi import (
    ad_max,
    ad_min,
    ad_sum,
    combine_equations,
    init_variables_adi,
    value,
)


@dataclass
class TopSurfaceGrid:
    """2D top-surface grid: a Grid embedded in 3D plus column data."""

    g: Grid                      # 2D grid, nodes carry (x, y, z_top)
    H: np.ndarray                # (nc,) column thickness
    z: np.ndarray                # (nc,) top-surface depth
    area: np.ndarray             # (nc,) areal footprint
    column_cells: List[np.ndarray]   # 3D cells of each column, top -> bottom
    parent: Grid
    node_z: Optional[np.ndarray] = None   # (n_nodes,) caprock depth at the
                                          # areal grid's nodes - needed by the
                                          # node-based trap analysis, which
                                          # resolves traps between cells


def top_surface_grid(g3: Grid) -> TopSurfaceGrid:
    """Collapse a Cartesian columnar 3D grid onto its top surface."""
    assert g3.cartDims is not None and len(g3.cartDims) == 3, (
        "top_surface_grid needs a logically Cartesian 3D grid"
    )
    nx, ny, nz = g3.cartDims
    assert g3.cells.centroids is not None, "run compute_geometry first"

    # Columns of active cells, keyed by their areal (i, j). A real formation
    # is a patch inside a rectangular raster, so most of the box is inactive
    # and whole columns are missing; the areal grid therefore has holes and
    # cannot be a plain tensor grid.
    if g3.cells.num == nx * ny * nz:
        col_index = np.arange(nx * ny)                 # every column present
        idx = np.arange(nx * ny * nz).reshape((nx, ny, nz), order="F")
        columns = [idx[i, j, :] for j in range(ny) for i in range(nx)]
    else:
        assert g3.cells.indexMap is not None, (
            "a grid with inactive cells needs cells.indexMap")
        gid = np.asarray(g3.cells.indexMap, dtype=np.int64)
        i = gid % nx
        j = (gid // nx) % ny
        k = gid // (nx * ny)
        areal = i + j * nx                             # 2D Cartesian index
        order = np.lexsort((k, areal))                 # by column, then depth
        areal_sorted = areal[order]
        col_index, starts = np.unique(areal_sorted, return_index=True)
        columns = [np.asarray(order[a:b], dtype=np.int64)
                   for a, b in zip(starts, list(starts[1:]) + [order.size])]

    from pymrst.gridprocessing.grid import BOTTOM, TOP

    def _face_z(cell: int, tag: int) -> float:
        """Depth of the centroid of a cell's top or bottom face."""
        s, e = g3.cells.facePos[cell], g3.cells.facePos[cell + 1]
        f = g3.cells.faces[s:e][g3.cells.tags[s:e] == tag]
        return float(g3.faces.centroids[f[0], 2])

    def _face_nodes(cell: int, tag: int) -> np.ndarray:
        s, e = g3.cells.facePos[cell], g3.cells.facePos[cell + 1]
        f = g3.cells.faces[s:e][g3.cells.tags[s:e] == tag][0]
        return g3.faces.nodes[g3.faces.nodePos[f]:g3.faces.nodePos[f + 1]]

    nc2 = len(columns)
    H = np.zeros(nc2)
    z_top = np.zeros(nc2)
    area = np.zeros(nc2)
    for c, col in enumerate(columns):
        # MRST's computeGeometryVE_2D sets cells.z to the arithmetic mean of
        # the cell's corner-node depths (not the face centroid).
        z_top[c] = float(g3.nodes.coords[_face_nodes(col[0], TOP), 2].mean())
        z_bot = float(
            g3.nodes.coords[_face_nodes(col[-1], BOTTOM), 2].mean()
        )
        H[c] = z_bot - z_top[c]      # vertical thickness (VE: vertical pillars)
        area[c] = g3.cells.volumes[col].sum() / H[c]

    # Build the areal 2D grid (nodes carry the top-surface depth).
    #
    # The axes cannot simply be read off the nodes: with inactive cells the
    # grid only carries the nodes of the mapped region, so whole rows and
    # columns of the raster are missing from them. Recover the full axes from
    # the spacing and one cell's areal index instead - a cell with areal index
    # i has its centre at x0 + (i + 1/2) h, which pins x0 exactly.
    xs_seen = np.unique(np.round(g3.nodes.coords[:, 0], 6))
    ys_seen = np.unique(np.round(g3.nodes.coords[:, 1], 6))
    if xs_seen.size == nx + 1 and ys_seen.size == ny + 1:
        xs, ys = xs_seen, ys_seen
    else:
        hx = float(np.median(np.diff(xs_seen)))
        hy = float(np.median(np.diff(ys_seen)))
        assert hx > 0 and hy > 0, "areal grid is not rectilinear"
        c0 = int(columns[0][0])
        i0 = int(col_index[0] % nx)
        j0 = int(col_index[0] // nx)
        lo, hi = g3.cells.facePos[c0], g3.cells.facePos[c0 + 1]
        cn = np.unique(np.concatenate([
            g3.faces.nodes[g3.faces.nodePos[f]:g3.faces.nodePos[f + 1]]
            for f in g3.cells.faces[lo:hi]]))
        x0 = float(g3.nodes.coords[cn, 0].min()) - i0 * hx
        y0 = float(g3.nodes.coords[cn, 1].min()) - j0 * hy
        xs = x0 + hx * np.arange(nx + 1)
        ys = y0 + hy * np.arange(ny + 1)
        # every node the grid does carry must sit on the reconstructed axes
        assert np.abs(xs[:, None] - xs_seen[None, :]).min(axis=0).max() < 1e-6, \
            "reconstructed x-axis does not match the grid's nodes"
        assert np.abs(ys[:, None] - ys_seen[None, :]).min(axis=0).max() < 1e-6, \
            "reconstructed y-axis does not match the grid's nodes"
    # The VE grid geometry is the *projected* (areal) one: MRST's
    # topSurfaceGrid computes face areas and cell-to-face distances in the
    # horizontal plane, so a sloping caprock does not inflate the
    # transmissibilities. The caprock depth enters only through the
    # gravity term (Gt.cells.z), not the geometry.
    from pymrst.gridprocessing.grid import extract_subgrid
    from pymrst.gridprocessing.tensor_grid import tensor_grid
    g2 = tensor_grid(xs, ys)
    if col_index.size != g2.cells.num:
        g2 = extract_subgrid(g2, col_index)     # drop the unmapped columns
    g2 = compute_geometry(g2)

    # Depth of the caprock at each areal *node*. The cell-based trap analysis
    # never needs it, but the node-based one lives on it: it is what lets a
    # trap boundary fall between cells rather than on a cell edge. Take, for
    # each areal node, the shallowest 3D node standing over it.
    xy3 = np.round(g3.nodes.coords[:, :2], 6)
    xy2 = np.round(g2.nodes.coords[:, :2], 6)
    order = np.lexsort((xy3[:, 1], xy3[:, 0]))
    keys3 = xy3[order]
    z3 = g3.nodes.coords[order, 2]
    uniq, first = np.unique(keys3, axis=0, return_index=True)
    bounds = np.append(first, keys3.shape[0])
    zmin = np.array([z3[a:b].min() for a, b in zip(bounds[:-1], bounds[1:])])
    lut = {tuple(k): float(v) for k, v in zip(map(tuple, uniq), zmin)}
    node_z = np.array([lut.get(tuple(xy2[n]), np.nan)
                       for n in range(g2.nodes.num)])
    assert not np.any(np.isnan(node_z)), (
        "some areal nodes have no 3D node above them")

    return TopSurfaceGrid(g=g2, H=H, z=z_top, area=area,
                          column_cells=columns, parent=g3, node_z=node_z)


def average_rock(rock: Rock, ts: TopSurfaceGrid) -> Rock:
    """Thickness-weighted vertical average of poro and perm (averageRock)."""
    nc2 = ts.g.cells.num
    poro = np.zeros(nc2)
    perm = np.zeros(nc2)
    vol = ts.parent.cells.volumes
    for c, col in enumerate(ts.column_cells):
        dz = vol[col] / (vol[col].sum() / ts.H[c])   # cell thickness
        w = dz / dz.sum()
        poro[c] = float(rock.poro[col] @ w)
        perm[c] = float(rock.perm[col, 0] @ w)
    return Rock(perm=perm[:, None], poro=poro)


@dataclass
class VEFluid:
    """Sharp-interface VE fluid."""

    rhoWS: float
    rhoGS: float
    mu_w: float
    mu_g: float
    srw: float                    # residual water
    srg: float                    # residual gas
    krW_max: float
    krG_max: float
    c_water: float = 0.0
    c_gas: float = 0.0
    p_ref: float = 1e7

    def b_w(self, p):
        return 1.0 + self.c_water * (p - self.p_ref)

    def b_g(self, p):
        return 1.0 + self.c_gas * (p - self.p_ref)

    def heights(self, sg, sg_max0, H):
        """Plume height h and historical maximum h_max.

        sGmax is max(sGmax0, sG) evaluated at the *current* sG (as MRST's
        get_sGmax does), so that h -> H exactly as sG -> 1 - s_wr while
        drainage is in progress.
        """
        sg_max = ad_max(sg, np.asarray(sg_max0, dtype=np.float64))
        h_max = sg_max * (H / (1.0 - self.srw))
        h = (H * sg - h_max * self.srg) / (1.0 - self.srg - self.srw)
        return ad_min(ad_max(h, 0.0), H), ad_min(h_max, H)

    def kr_g(self, h, H):
        return h * (self.krG_max / H)

    def kr_w(self, h, h_max, H):
        return (H - h_max + (h_max - h) * self.krW_max) / H

    def pc(self, h, p, gravity: float = NORM_GRAVITY):
        drho = self.rhoWS * self.b_w(p) - self.rhoGS * self.b_g(p)
        return drho * h * gravity


@dataclass
class VEState:
    """VE state: water pressure at the caprock, upscaled sG, and sGmax."""

    pressure: np.ndarray
    sG: np.ndarray
    sGmax: np.ndarray
    qGs: np.ndarray
    bhp: np.ndarray

    def copy(self) -> "VEState":
        return VEState(*(v.copy() for v in (self.pressure, self.sG,
                                            self.sGmax, self.qGs, self.bhp)))


class VEModel:
    """Vertical-equilibrium two-phase model on a top-surface grid."""

    def __init__(
        self,
        ts: TopSurfaceGrid,
        rock2d: Rock,
        fluid: VEFluid,
        well_cells: Optional[np.ndarray] = None,
        well_rate: float = 0.0,
        well_WI: Optional[np.ndarray] = None,
        bc_cells: Optional[np.ndarray] = None,
        bc_trans: Optional[np.ndarray] = None,
        bc_pressure: Optional[np.ndarray] = None,
        gravity: float = NORM_GRAVITY,
        tol: float = 1e-9,
        max_iterations: int = 40,
    ):
        self.ts = ts
        self.fluid = fluid
        self.grav = gravity
        self.tol = tol
        self.max_iterations = max_iterations
        g, H = ts.g, ts.H

        self.pv = rock2d.poro * ts.area * H
        # vertically integrated permeability -> face transmissibility
        cell_no = g.cell_no()
        cf = g.cells.faces
        cvec = g.faces.centroids[cf] - g.cells.centroids[cell_no]
        sgn = np.where(g.faces.neighbors[cf, 0] == cell_no, 1.0, -1.0)
        nvec = sgn[:, None] * g.faces.normals[cf]
        kh = (rock2d.perm[:, 0] * H)[cell_no]
        t_half = kh * np.sum(cvec * nvec, axis=1) / np.sum(cvec**2, axis=1)
        t_face = 1.0 / np.bincount(cf, weights=1.0 / np.abs(t_half),
                                   minlength=g.faces.num)

        internal = np.all(g.faces.neighbors >= 0, axis=1)
        self.N = g.faces.neighbors[internal]
        self.T = t_face[internal]
        nif, nc = self.N.shape[0], g.cells.num
        self.C = sp.csr_array(
            (np.tile([-1.0, 1.0], nif),
             (np.repeat(np.arange(nif), 2), self.N.ravel())),
            shape=(nif, nc),
        )
        self.gdz = gravity * (ts.z[self.N[:, 1]] - ts.z[self.N[:, 0]])

        self.well_cells = (np.zeros(0, dtype=np.int64) if well_cells is None
                           else np.asarray(well_cells, np.int64))
        self.well_rate = well_rate
        self.well_WI = well_WI

        # Open boundaries: a hydrostatic brine pressure is imposed.
        self.bc_cells = (np.zeros(0, dtype=np.int64) if bc_cells is None
                         else np.asarray(bc_cells, np.int64))
        self.bc_trans = (np.zeros(0) if bc_trans is None
                         else np.asarray(bc_trans, float))
        self.bc_press = (np.zeros(0) if bc_pressure is None
                         else np.asarray(bc_pressure, float))

    def _grad(self, x):
        from pymrst.utils.adi import ad_matmul
        return ad_matmul(self.C, x)

    def _div(self, x):
        from pymrst.utils.adi import ad_matmul
        return ad_matmul(-self.C.T, x)

    def _upw(self, flag, x):
        from pymrst.utils.adi import ad_matmul
        m = sp.csr_array(
            (np.ones(self.N.shape[0]),
             (np.arange(self.N.shape[0]),
              np.where(flag, self.N[:, 0], self.N[:, 1]))),
            shape=(self.N.shape[0], self.ts.g.cells.num),
        )
        return ad_matmul(m, x)

    def assemble(self, st0: VEState, p, sg, qgs, bhp, dt: float):
        f, H, pv = self.fluid, self.ts.H, self.pv
        h, h_max = f.heights(sg, st0.sGmax, H)
        pc = f.pc(h, p, self.grav)

        b_w, b_g = f.b_w(p), f.b_g(p + pc)
        mob_w = f.kr_w(h, h_max, H) / f.mu_w
        mob_g = f.kr_g(h, H) / f.mu_g
        rho_w = b_w * f.rhoWS
        rho_g = b_g * f.rhoGS

        h0, hmax0 = f.heights(st0.sG, st0.sGmax, H)
        pc0 = value(f.pc(h0, st0.pressure, self.grav))
        b_w0 = value(f.b_w(st0.pressure))
        b_g0 = value(f.b_g(st0.pressure + pc0))

        # water phase uses p; gas phase uses the gas pressure p + pc
        eqs = []
        for press, mob, rho, b in ((p, mob_w, rho_w, b_w),
                                   (p + pc, mob_g, rho_g, b_g)):
            pot = self._grad(press) - 0.5 * (
                value(rho)[self.N[:, 0]] + value(rho)[self.N[:, 1]]
            ) * self.gdz
            flag = value(pot) <= 0
            v = -self.T * self._upw(flag, mob) * pot
            eqs.append(self._div(self._upw(flag, b) * v))

        water = (pv / dt) * (b_w * (1.0 - sg)
                             - b_w0 * (1.0 - st0.sG)) + eqs[0]
        gas = (pv / dt) * (b_g * sg - b_g0 * st0.sG) + eqs[1]

        # Open boundaries: brine flows in/out; CO2 may leave if it arrives.
        if self.bc_cells.size:
            bc = self.bc_cells
            tb = self.bc_trans
            dpw = self.bc_press - p[bc]
            inflow = value(dpw) > 0
            mob_w_bc = np.where(inflow, f.krW_max / f.mu_w, 0.0) \
                + (~inflow) * mob_w[bc]
            q_w = tb * mob_w_bc * dpw
            water[bc] = water[bc] - b_w[bc] * q_w
            # gas only leaves (no CO2 in the far-field aquifer)
            dpg = self.bc_press - (p + pc)[bc]
            q_g = tb * ((~inflow) * mob_g[bc]) * dpg
            gas[bc] = gas[bc] - b_g[bc] * q_g

        ctrls = []
        if self.well_cells.size:
            wc = self.well_cells
            wi = self.well_WI
            dd = -bhp[0:1] + p[wc]
            mt = mob_w[wc] + mob_g[wc]
            cq_g = -(wi * mt) * dd          # pure CO2 injection
            gas[wc] = gas[wc] - b_g[wc] * cq_g
            ctrls.append(qgs[0:1] - ad_sum(b_g[wc] * cq_g))
            ctrls.append(qgs[0:1] - self.well_rate)

        return [water, gas] + ctrls

    def step(self, state: VEState, dt: float) -> Tuple[VEState, bool, int]:
        st0 = state
        st = state.copy()
        nc = self.ts.g.cells.num
        for it in range(self.max_iterations):
            p, sg, qgs, bhp = init_variables_adi(
                st.pressure, st.sG, st.qGs, st.bhp
            )
            eqs = self.assemble(st0, p, sg, qgs, bhp, dt)
            res = np.concatenate([np.atleast_1d(value(e)) for e in eqs])
            scale = max(np.abs(self.pv / dt).max(), 1.0)
            if np.max(np.abs(res)) <= self.tol * scale:
                st.sGmax = np.maximum(st0.sGmax, st.sG)
                return st, True, it
            eq = combine_equations(*eqs)
            upd = spla.spsolve(eq.jac[0].tocsc(), -eq.val)
            st.pressure = st.pressure + upd[:nc]
            st.sG = np.clip(st.sG + np.clip(upd[nc:2 * nc], -0.2, 0.2),
                            0.0, 1.0 - self.fluid.srw)
            if self.well_cells.size:
                st.qGs = st.qGs + upd[2 * nc:2 * nc + 1]
                st.bhp = st.bhp + upd[2 * nc + 1:2 * nc + 2]
        st.sGmax = np.maximum(st0.sGmax, st.sG)
        return st, False, self.max_iterations

    def plume_height(self, st: VEState) -> np.ndarray:
        """CO2 plume thickness under the caprock (m)."""
        h, _ = self.fluid.heights(st.sG, st.sGmax, self.ts.H)
        return value(h)

    def stored_co2(self, st: VEState) -> float:
        """CO2 in place as a surface volume (the conserved quantity).

        The accumulation term uses the gas-phase pressure p + pc, so the
        b-factor must be evaluated there, not at the water pressure.
        """
        f = self.fluid
        h, _ = f.heights(st.sG, st.sGmax, self.ts.H)
        pc = value(f.pc(h, st.pressure, self.grav))
        return float(self.pv @ (value(f.b_g(st.pressure + pc)) * st.sG))


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid
    from pymrst.params.rock import make_rock
    from pymrst.utils.units import DARCY, DAY, MILLI, YEAR

    def _aquifer(nz=10):
        """A gently sloping 2 km x 20 m aquifer at ~1000 m depth."""
        g = cart_grid([50, 1, nz], [2000.0, 100.0, 20.0])
        x = g.nodes.coords[:, 0]
        g.nodes.coords[:, 2] += 1000.0 - 0.04 * x      # 4% slope upward in x
        return compute_geometry(g)

    def test_top_surface_grid():
        """Columns collapse correctly: H, area, and top depth."""
        g3 = _aquifer()
        ts = top_surface_grid(g3)
        assert ts.g.cells.num == 50
        assert np.allclose(ts.H, 20.0, rtol=1e-12)
        assert np.allclose(ts.area, 40.0 * 100.0, rtol=1e-12)
        # top depth decreases with x (the aquifer rises to the right)
        assert np.all(np.diff(ts.z) < 0)
        assert np.isclose(ts.z[0], 1000.0 - 0.04 * 20.0, atol=1e-9)
        # total pore volume is preserved
        assert np.isclose((ts.area * ts.H).sum(), g3.cells.volumes.sum(),
                          rtol=1e-12)
        print("ok test_top_surface_grid")

    def test_sharp_interface_heights():
        """h and the upscaled relperms follow the sharp-interface formulas."""
        f = VEFluid(rhoWS=1050.0, rhoGS=700.0, mu_w=8e-4, mu_g=6e-5,
                    srw=0.2, srg=0.1, krW_max=1.0, krG_max=1.0)
        H = np.array([20.0])
        # fully swept column at its historical maximum
        sg = np.array([0.4])
        h, hmax = f.heights(sg, sg, H)
        # sGmax == sG: h_max = sg H/(1-srw), h = (H sg - h_max srg)/(1-srg-srw)
        hm_exp = 0.4 * 20.0 / 0.8
        h_exp = (20.0 * 0.4 - hm_exp * 0.1) / 0.7
        assert np.isclose(value(hmax)[0], hm_exp)
        assert np.isclose(value(h)[0], min(h_exp, 20.0))
        assert np.isclose(value(f.kr_g(h, H))[0], value(h)[0] / 20.0)
        # capillary pressure is the buoyancy head over the plume
        pc = value(f.pc(h, np.array([1e7])))
        assert np.isclose(pc[0], 350.0 * value(h)[0] * NORM_GRAVITY)
        print("ok test_sharp_interface_heights")

    def test_ve_injection_and_updip_migration():
        """CO2 injected into a sloping aquifer is stored (exact surface-volume
        balance) and migrates updip under buoyancy."""
        g3 = _aquifer()
        rock3 = make_rock(g3, 400 * MILLI * DARCY, 0.2)
        ts = top_surface_grid(g3)
        rock2 = average_rock(rock3, ts)
        assert np.allclose(rock2.poro, 0.2)
        assert np.allclose(rock2.perm, 400 * MILLI * DARCY)

        # Closed aquifer with slightly compressible fluids: the pressure
        # system is well posed and CO2 cannot leave, so the surface-volume
        # balance must close exactly.
        f = VEFluid(rhoWS=1050.0, rhoGS=700.0, mu_w=8e-4, mu_g=6e-5,
                    srw=0.2, srg=0.1, krW_max=1.0, krG_max=1.0,
                    c_water=4.3e-10, c_gas=1e-9, p_ref=1e7)
        # z decreases with x: cell 0 is deepest (downdip), cell 49 shallowest.
        wc = np.array([10])
        wi = np.array([1e-11])
        rate = 2e-3                          # surface m3/s of CO2
        model = VEModel(ts, rock2, f, well_cells=wc, well_rate=rate,
                        well_WI=wi)

        nc = ts.g.cells.num
        p0 = 1e5 + f.rhoWS * NORM_GRAVITY * ts.z
        st = VEState(pressure=p0, sG=np.zeros(nc), sGmax=np.zeros(nc),
                     qGs=np.array([rate]), bhp=np.array([p0[wc[0]] + 1e5]))

        dts = np.full(10, 30.0 * DAY)
        for i, dt in enumerate(dts):
            st, ok, its = model.step(st, dt)
            assert ok, f"step {i} failed after {its} iterations"
            assert np.isclose(st.qGs[0], rate, rtol=1e-10)

        stored = model.stored_co2(st)
        injected = rate * dts.sum()
        assert np.isclose(stored, injected, rtol=1e-9), (stored, injected)

        h = model.plume_height(st)
        assert 0.0 < h.max() <= ts.H.max() + 1e-9
        assert st.sG[wc[0]] > 0
        # buoyancy carries the plume updip (+x, where the caprock is shallower)
        updip = h[wc[0] + 1:].sum()
        downdip = h[:wc[0]].sum()
        assert updip > downdip, "plume must migrate updip"
        print(f"ok test_ve_injection_and_updip_migration "
              f"(max plume height {h.max():.2f} m of {ts.H[0]:.0f} m)")

    test_top_surface_grid()
    test_sharp_interface_heights()
    test_ve_injection_and_updip_migration()
    print("\nAll tests passed!")
