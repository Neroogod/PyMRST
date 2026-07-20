"""Dual-porosity flow models (MRST dual-porosity/ad_models).

Two overlapping continua on the same grid. The fractures carry the flow
(they have the divergence term); the matrix blocks only store fluid and
exchange it with the fractures through the transfer term. So each cell
carries two pressures - and, in the two-phase case, two saturations - and
the matrix equations have no flux between neighbours at all.

    fracture:  d/dt (pv_f b s) + div(b v) + T = q_bc
    matrix:    d/dt (pv_m b s)              - T = 0

Forced by pressure/flux boundary conditions, as MRST's own dual-porosity
examples are.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse.linalg as spla

from pymrst.gridprocessing.grid import Grid
from pymrst.modules.ad_core.utils.operators import setup_operators_tpfa
from pymrst.modules.ad_props.simple_adi_fluid import ADFluid
from pymrst.modules.dual_porosity.transfer import (
    KazemiSinglePhaseTransfer,
    KazemiTwoPhaseTransfer,
)
from pymrst.params.rock import Rock, pore_volume
from pymrst.params.wells_and_bc import BoundaryCondition
from pymrst.solvers.trans import compute_trans
from pymrst.utils.adi import combine_equations, init_variables_adi, value


def _face_trans(g: Grid, rock: Rock) -> np.ndarray:
    """Full face transmissibility (MRST's operators.T_all).

    On a boundary face there is only one half-face, so this is just that
    half-face's transmissibility - which is what the BC term needs.
    """
    t_half = compute_trans(g, rock)
    return 1.0 / np.bincount(g.cells.faces, weights=1.0 / t_half,
                             minlength=g.faces.num)


def _bc_faces(g: Grid, bc: BoundaryCondition) -> Tuple[np.ndarray, np.ndarray]:
    """The boundary faces and the cells behind them."""
    faces = np.asarray(bc.face, dtype=np.int64).ravel()
    nb = g.faces.neighbors[faces]
    assert not np.any((nb >= 0).all(axis=1)), "boundary condition on an interior face"
    return faces, nb.max(axis=1)


@dataclass
class DPWaterState:
    """Single-phase dual-porosity state: one pressure per continuum."""

    pressure: np.ndarray             # fracture
    pressure_matrix: np.ndarray      # matrix

    def copy(self) -> "DPWaterState":
        return DPWaterState(self.pressure.copy(), self.pressure_matrix.copy())


@dataclass
class DPOilWaterState:
    """Two-phase dual-porosity state: pressure and saturation per continuum."""

    pressure: np.ndarray
    sW: np.ndarray
    pressure_matrix: np.ndarray
    sW_matrix: np.ndarray

    def copy(self) -> "DPOilWaterState":
        return DPOilWaterState(self.pressure.copy(), self.sW.copy(),
                               self.pressure_matrix.copy(),
                               self.sW_matrix.copy())


class WaterModelDP:
    """Single-phase dual-porosity water model (MRST WaterModelDP)."""

    def __init__(
        self,
        g: Grid,
        rock_fracture: Rock,
        rock_matrix: Rock,
        fluid_fracture: ADFluid,
        fluid_matrix: ADFluid,
        transfer: KazemiSinglePhaseTransfer,
        bc: Optional[BoundaryCondition] = None,
        tol: float = 1e-9,
        max_iterations: int = 25,
    ):
        self.g = g
        self.rock_fracture = rock_fracture
        self.rock_matrix = rock_matrix
        self.fluid = fluid_fracture
        self.fluid_matrix = fluid_matrix
        self.transfer = transfer
        self.bc = bc

        # flow happens in the fractures, so the operators use the fracture rock
        self.ops = setup_operators_tpfa(g, rock_fracture)
        self.pv = pore_volume(g, rock_fracture)
        self.pv_matrix = pore_volume(g, rock_matrix)
        self.vb = g.cells.volumes
        self.tol = tol
        self.max_iterations = max_iterations

        if bc is not None and bc.face.size:
            self.t_all = _face_trans(g, rock_fracture)
            self.bc_faces, self.bc_cells = _bc_faces(g, bc)
            assert all(t == "pressure" for t in bc.type), \
                "WaterModelDP supports pressure boundary conditions"

    def initial_state(self, p0: float) -> DPWaterState:
        p = np.full(self.g.cells.num, float(p0))
        return DPWaterState(pressure=p, pressure_matrix=p.copy())

    def assemble(self, st0: DPWaterState, p, pm, dt: float):
        f, fm, ops = self.fluid, self.fluid_matrix, self.ops

        b_w = f.b_w(p)
        b_w0 = f.b_w(st0.pressure)
        b_wm = fm.b_w(pm)
        b_wm0 = fm.b_w(st0.pressure_matrix)
        mob_w = 1.0 / f.mu[0]                      # constant viscosity

        pot = ops.grad(p)
        flag = value(pot) <= 0
        v_w = -ops.T * mob_w * pot
        b_wvw = ops.upw(flag, b_w) * v_w

        t_w = self.vb * self.transfer.transfer(
            self.rock_matrix, fm, p, pm)

        water = (self.pv / dt) * (b_w - b_w0) + ops.div(b_wvw) + t_w
        matrix = (self.pv_matrix / dt) * (b_wm - b_wm0) - t_w

        if self.bc is not None and self.bc.face.size:
            t = self.t_all[self.bc_faces]
            c = self.bc_cells
            dp = np.asarray(self.bc.value, float).ravel() - p[c]
            # mobility and b are taken inside the cell in both directions,
            # which is what MRST does when the bc carries no outside state
            q = f.b_w(p[c]) * mob_w * t * dp
            water[c] = water[c] - q

        return water, matrix

    def step(self, state: DPWaterState, dt: float
             ) -> Tuple[DPWaterState, bool, int]:
        st0, st = state, state.copy()
        nc = self.g.cells.num
        scale = self.pv.sum() / dt

        for it in range(self.max_iterations):
            p, pm = init_variables_adi(st.pressure, st.pressure_matrix)
            eqs = self.assemble(st0, p, pm, dt)
            res = max(np.abs(value(e)).max() for e in eqs) / scale
            if res <= self.tol:
                return st, True, it
            eq = combine_equations(*eqs)
            upd = spla.spsolve(eq.jac[0].tocsc(), -eq.val)
            st.pressure = st.pressure + upd[:nc]
            st.pressure_matrix = st.pressure_matrix + upd[nc:]

        p, pm = init_variables_adi(st.pressure, st.pressure_matrix)
        eqs = self.assemble(st0, p, pm, dt)
        res = max(np.abs(value(e)).max() for e in eqs) / scale
        return st, res <= self.tol, self.max_iterations


class TwoPhaseOilWaterModelDP:
    """Two-phase dual-porosity oil-water model (MRST TwoPhaseOilWaterModelDP)."""

    def __init__(
        self,
        g: Grid,
        rock_fracture: Rock,
        rock_matrix: Rock,
        fluid_fracture: ADFluid,
        fluid_matrix: ADFluid,
        transfer: KazemiTwoPhaseTransfer,
        bc: Optional[BoundaryCondition] = None,
        pc_ow_fracture: Optional[Callable] = None,
        pc_ow_matrix: Optional[Callable] = None,
        ds_max: float = 0.2,
        tol: float = 1e-9,
        max_iterations: int = 30,
    ):
        self.g = g
        self.rock_fracture = rock_fracture
        self.rock_matrix = rock_matrix
        self.fluid = fluid_fracture
        self.fluid_matrix = fluid_matrix
        self.transfer = transfer
        self.bc = bc
        self.pc_ow_fracture = pc_ow_fracture
        self.pc_ow_matrix = pc_ow_matrix

        self.ops = setup_operators_tpfa(g, rock_fracture)
        self.pv = pore_volume(g, rock_fracture)
        self.pv_matrix = pore_volume(g, rock_matrix)
        self.vb = g.cells.volumes
        self.ds_max = ds_max
        self.tol = tol
        self.max_iterations = max_iterations

        if bc is not None and bc.face.size:
            self.t_all = _face_trans(g, rock_fracture)
            self.bc_faces, self.bc_cells = _bc_faces(g, bc)
            assert all(t == "pressure" for t in bc.type), \
                "TwoPhaseOilWaterModelDP supports pressure boundary conditions"
            assert bc.sat.shape[1] >= 2, "pressure bc needs a 2-phase sat"

    def initial_state(self, p0: float, sw0: float) -> DPOilWaterState:
        nc = self.g.cells.num
        return DPOilWaterState(
            pressure=np.full(nc, float(p0)), sW=np.full(nc, float(sw0)),
            pressure_matrix=np.full(nc, float(p0)),
            sW_matrix=np.full(nc, float(sw0)))

    def assemble(self, st0: DPOilWaterState, p, sw, pm, swm, dt: float):
        f, fm, ops = self.fluid, self.fluid_matrix, self.ops
        so = 1.0 - sw
        som = 1.0 - swm

        b_w, b_o = f.b_w(p), f.b_o(p)
        b_w0 = f.b_w(st0.pressure)
        b_o0 = f.b_o(st0.pressure)

        mob_w = f.kr_w(sw) / f.mu[0]
        mob_o = f.kr_o(so) / f.mu[1]

        # fracture flow (no gravity; capillary pressure in the fracture is
        # zero in MRST's examples, so both phases see the same potential)
        pot = ops.grad(p)
        flag = value(pot) <= 0
        v_w = -ops.T * ops.upw(flag, mob_w) * pot
        v_o = -ops.T * ops.upw(flag, mob_o) * pot
        b_wvw = ops.upw(flag, b_w) * v_w
        b_ovo = ops.upw(flag, b_o) * v_o

        # matrix: pressures carry the capillary pressure, and there is no flux
        pc_m = (0.0 if self.pc_ow_matrix is None
                else self.pc_ow_matrix(swm))
        pc_m0 = (0.0 if self.pc_ow_matrix is None
                 else self.pc_ow_matrix(st0.sW_matrix))
        pwm = pm - pc_m
        pwm0 = st0.pressure_matrix - pc_m0
        b_wm, b_wm0 = fm.b_w(pwm), fm.b_w(pwm0)
        b_om, b_om0 = fm.b_o(pm), fm.b_o(st0.pressure_matrix)

        t_w, t_o = self.transfer.transfer(
            self.rock_matrix, f, fm, p, sw, pm, swm)
        t_w = self.vb * t_w
        t_o = self.vb * t_o

        pv, pvm = self.pv, self.pv_matrix
        water = (1.0 / dt) * (pv * b_w * sw - pv * b_w0 * st0.sW) + t_w
        oil = (1.0 / dt) * (pv * b_o * so
                            - pv * b_o0 * (1.0 - st0.sW)) + t_o
        water = water + ops.div(b_wvw)
        oil = oil + ops.div(b_ovo)

        water_m = (1.0 / dt) * (pvm * b_wm * swm
                                - pvm * b_wm0 * st0.sW_matrix) - t_w
        oil_m = (1.0 / dt) * (pvm * b_om * som
                              - pvm * b_om0 * (1.0 - st0.sW_matrix)) - t_o

        if self.bc is not None and self.bc.face.size:
            t = self.t_all[self.bc_faces]
            c = self.bc_cells
            sat = np.asarray(self.bc.sat, float)
            dp = np.asarray(self.bc.value, float).ravel() - p[c]
            inflow = value(dp) > 0

            mob_wc, mob_oc = mob_w[c], mob_o[c]
            tot_mob = mob_wc + mob_oc
            # inflowing fluid arrives with the bc's saturation and the total
            # mobility; outflowing fluid leaves with the cell's own mobility
            fin = inflow.astype(np.float64)
            mob_w_bc = fin * (tot_mob * sat[:, 0]) + (1.0 - fin) * mob_wc
            mob_o_bc = fin * (tot_mob * sat[:, 1]) + (1.0 - fin) * mob_oc

            q_w = f.b_w(p[c]) * mob_w_bc * t * dp
            q_o = f.b_o(p[c]) * mob_o_bc * t * dp
            water[c] = water[c] - q_w
            oil[c] = oil[c] - q_o

        return water, oil, water_m, oil_m

    def step(self, state: DPOilWaterState, dt: float
             ) -> Tuple[DPOilWaterState, bool, int]:
        st0, st = state, state.copy()
        nc = self.g.cells.num
        scale = self.pv.sum() / dt

        for it in range(self.max_iterations):
            p, sw, pm, swm = init_variables_adi(
                st.pressure, st.sW, st.pressure_matrix, st.sW_matrix)
            eqs = self.assemble(st0, p, sw, pm, swm, dt)
            res = max(np.abs(value(e)).max() for e in eqs) / scale
            if res <= self.tol:
                return st, True, it

            eq = combine_equations(*eqs)
            upd = spla.spsolve(eq.jac[0].tocsc(), -eq.val)
            st.pressure = st.pressure + upd[:nc]
            st.sW = np.clip(st.sW + np.clip(upd[nc:2 * nc],
                                            -self.ds_max, self.ds_max), 0.0, 1.0)
            st.pressure_matrix = st.pressure_matrix + upd[2 * nc:3 * nc]
            st.sW_matrix = np.clip(
                st.sW_matrix + np.clip(upd[3 * nc:], -self.ds_max,
                                       self.ds_max), 0.0, 1.0)

        p, sw, pm, swm = init_variables_adi(
            st.pressure, st.sW, st.pressure_matrix, st.sW_matrix)
        eqs = self.assemble(st0, p, sw, pm, swm, dt)
        res = max(np.abs(value(e)).max() for e in eqs) / scale
        return st, res <= self.tol, self.max_iterations


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_props.simple_adi_fluid import init_simple_adi_fluid
    from pymrst.modules.dual_porosity.shape_factors import KazemiShapeFactor
    from pymrst.params.rock import make_rock
    from pymrst.params.wells_and_bc import add_bc, pside
    from pymrst.utils.units import BARSA, CENTI, DARCY, DAY, MILLI, POISE

    def _grid(n=40, L=60.0):
        return compute_geometry(cart_grid([n, 1, 1], [L, 1.0, 1.0]))

    def _rocks(g, k_m, phi_m, k_f, phi_f):
        nc = g.cells.num
        rm = make_rock(g, np.full((nc, 1), k_m), np.full((nc, 1), phi_m))
        rf = make_rock(g, np.full((nc, 1), k_f), np.full((nc, 1), phi_f))
        return rf, rm

    def _water_model(bc=True):
        # MRST's own example_1ph: a very compressible fluid, so the two
        # continua have enough storage to drain on different timescales.
        # With a stiffer fluid the whole model empties in a single step and
        # there is no dual-porosity behaviour left to see.
        g = _grid(n=100)
        rf, rm = _rocks(g, 1 * DARCY, 0.3, 1000 * DARCY, 0.01)
        f = init_simple_adi_fluid(mu=[1.0, 1.0], rho=[1.0, 1.0],
                                  n=[1.0, 1.0], c=[1e-6, 1e-6], p_ref=0.0)
        tf = KazemiSinglePhaseTransfer(
            shape_factor=KazemiShapeFactor(block_dimension=[10.0, 10.0, 10.0]))
        b = pside(None, g, "xmin", 0.0) if bc else None
        return g, WaterModelDP(g, rf, rm, f, f, tf, bc=b)

    def test_water_dp_equilibrates_with_no_forcing():
        """With no boundary condition, an off-balance state relaxes.

        The fracture and matrix start at different pressures; the transfer
        term must drive them together and conserve mass while doing it.
        """
        g, model = _water_model(bc=False)
        st = DPWaterState(pressure=np.full(g.cells.num, 120 * BARSA),
                          pressure_matrix=np.full(g.cells.num, 80 * BARSA))

        def mass(s):
            return float((model.pv * model.fluid.b_w(s.pressure)).sum()
                         + (model.pv_matrix
                            * model.fluid_matrix.b_w(s.pressure_matrix)).sum())

        m0 = mass(st)
        gap0 = abs(st.pressure[0] - st.pressure_matrix[0])
        for _ in range(20):
            st, ok, _ = model.step(st, 1.0 * DAY)
            assert ok
        gap = abs(st.pressure[0] - st.pressure_matrix[0])

        assert gap < 0.01 * gap0, f"pressures did not equilibrate ({gap0} -> {gap})"
        assert abs(mass(st) - m0) / m0 < 1e-10, "closed system lost mass"
        # equilibrium is the volume-weighted mean, not the midpoint
        assert st.pressure[0] > 80 * BARSA and st.pressure[0] < 120 * BARSA
        print(f"ok test_water_dp_equilibrates_with_no_forcing "
              f"(gap {gap0 / BARSA:.1f} -> {gap / BARSA:.2e} bar)")

    def test_water_dp_matrix_lags_the_fracture():
        """Depleting the fractures, the matrix must follow behind, not with.

        This is the whole point of dual porosity: the fracture drains fast,
        the matrix feeds it slowly. If the matrix tracked the fracture
        exactly, a single-porosity model would do.
        """
        g, model = _water_model(bc=True)
        p0 = 70 * BARSA
        st = model.initial_state(p0)             # bc holds x = 0 at 0 bar

        for _ in range(10):
            st, ok, _ = model.step(st, 0.1 * DAY)
            assert ok

        # the fracture has drawn down hard at the outlet while the matrix,
        # which can only drain through the transfer term, has barely moved
        assert st.pressure[0] < 0.3 * p0, "the fracture should drain fast"
        assert st.pressure_matrix[0] > 0.9 * p0, "the matrix should lag"
        assert np.all(st.pressure < st.pressure_matrix), \
            "the fracture must lead the matrix everywhere"
        # and the far end has not felt the outlet much yet
        assert st.pressure[-1] > 0.75 * p0
        print(f"ok test_water_dp_matrix_lags_the_fracture "
              f"(at the outlet after 1 day: fracture "
              f"{st.pressure[0] / p0:.0%} of initial, matrix "
              f"{st.pressure_matrix[0] / p0:.0%})")

    def test_water_dp_reduces_to_single_porosity():
        """Turning the transfer up makes the two continua move as one."""
        g, _ = _water_model(bc=True)
        rf, rm = _rocks(g, 1 * DARCY, 0.3, 1000 * DARCY, 0.01)
        f = init_simple_adi_fluid(mu=[1.0, 1.0], rho=[1.0, 1.0],
                                  n=[1.0, 1.0], c=[1e-6, 1e-6], p_ref=0.0)
        bc = pside(None, g, "xmin", 0.0)

        # a tiny block: enormous shape factor, so transfer is instantaneous
        tf = KazemiSinglePhaseTransfer(
            shape_factor=KazemiShapeFactor(block_dimension=[0.02] * 3))
        model = WaterModelDP(g, rf, rm, f, f, tf, bc=bc)

        st = model.initial_state(70 * BARSA)
        for _ in range(5):
            st, ok, _ = model.step(st, 0.1 * DAY)
            assert ok
        gap = np.abs(st.pressure - st.pressure_matrix).max()
        assert gap / (70 * BARSA) < 1e-3, \
            f"strong coupling should collapse the two continua (gap {gap})"
        print("ok test_water_dp_reduces_to_single_porosity")

    def test_two_phase_dp_imbibition():
        """Capillary imbibition pulls water from the fractures into the matrix.

        Water floods the fracture network; the matrix, which no flow ever
        touches directly, must fill up purely through the transfer term.
        """
        g = _grid(n=30, L=30.0)
        rf, rm = _rocks(g, 10 * MILLI * DARCY, 0.1, 100 * MILLI * DARCY, 0.01)
        f = init_simple_adi_fluid(
            mu=[1 * CENTI * POISE, 1 * CENTI * POISE], rho=[1000.0, 800.0],
            n=[1.0, 1.0], c=[1e-12, 1e-12], p_ref=0.0)

        # 20 m matrix blocks: transfer is slower than the fracture flood, so
        # the matrix visibly lags. With small blocks (MRST's example uses 5 m)
        # the capillary suction is strong enough to outrun the flood and the
        # matrix can lead the fracture at the far end - real, but not the
        # regime this test is about.
        Pe = 7 * BARSA                          # matrix capillary entry
        tf = KazemiTwoPhaseTransfer(
            shape_factor=KazemiShapeFactor(block_dimension=[20.0] * 3),
            pc_ow_matrix=lambda sw: Pe)

        bc = pside(None, g, "xmin", 70 * BARSA, sat=[1.0, 0.0])
        bc = pside(bc, g, "xmax", 0.0, sat=[1.0, 0.0])

        model = TwoPhaseOilWaterModelDP(g, rf, rm, f, f, tf, bc=bc,
                                        pc_ow_matrix=lambda sw: Pe)
        st = model.initial_state(0.0, 0.0)      # both continua full of oil

        for k in range(6):
            st, ok, _ = model.step(st, 0.5 * DAY)
            assert ok, f"step {k} failed"

        # the fracture floods from the inlet
        assert st.sW[0] > 0.5, st.sW[0]
        assert st.sW[0] > st.sW[-1], "fracture front should run inlet -> outlet"
        # the matrix, which no flux ever reaches, fills only by imbibition
        assert st.sW_matrix[0] > 0.05, "matrix never imbibed"
        # and it lags the fracture, which is the whole point of dual porosity
        assert st.sW_matrix[0] < 0.5 * st.sW[0], "matrix should lag the fracture"
        assert st.sW_matrix[-1] < st.sW_matrix[0], "imbibition follows the front"
        # saturations stay physical
        for s in (st.sW, st.sW_matrix):
            assert s.min() >= -1e-12 and s.max() <= 1.0 + 1e-12
        print(f"ok test_two_phase_dp_imbibition "
              f"(at the inlet after 3 d: fracture sW {st.sW[0]:.3f}, "
              f"matrix sW {st.sW_matrix[0]:.3f})")

    def test_capillarity_is_what_fills_the_matrix():
        """Capillary pressure, not the flood, is what drives imbibition.

        The control case. Without pc the matrix still takes a trace of water
        - the flood raises the fracture pressure above the matrix, so a
        little transfers on the pressure difference alone - but it is orders
        of magnitude less than capillary imbibition. Comparing the two runs
        is the honest test; an absolute bound on the pc-free case would just
        be pinning that transient.
        """
        g = _grid(n=30, L=30.0)
        rf, rm = _rocks(g, 10 * MILLI * DARCY, 0.1, 100 * MILLI * DARCY, 0.01)
        f = init_simple_adi_fluid(
            mu=[1 * CENTI * POISE, 1 * CENTI * POISE], rho=[1000.0, 800.0],
            n=[1.0, 1.0], c=[1e-12, 1e-12], p_ref=0.0)

        bc = pside(None, g, "xmin", 70 * BARSA, sat=[1.0, 0.0])
        bc = pside(bc, g, "xmax", 0.0, sat=[1.0, 0.0])

        def run(pc):
            tf = KazemiTwoPhaseTransfer(
                shape_factor=KazemiShapeFactor(block_dimension=[20.0] * 3),
                pc_ow_matrix=pc)
            model = TwoPhaseOilWaterModelDP(g, rf, rm, f, f, tf, bc=bc,
                                            pc_ow_matrix=pc)
            st = model.initial_state(0.0, 0.0)
            for _ in range(6):
                st, ok, _ = model.step(st, 0.5 * DAY)
                assert ok
            return st

        wet = run(lambda sw: 7 * BARSA)
        dry = run(None)

        # the fracture floods either way - the flood does not need pc
        assert wet.sW[0] > 0.5 and dry.sW[0] > 0.5
        assert abs(wet.sW[0] - dry.sW[0]) < 0.5, "the flood is not pc-driven"
        # but the matrix only fills when capillarity pulls water in
        assert wet.sW_matrix[0] > 1000 * dry.sW_matrix[0], (
            f"capillary imbibition {wet.sW_matrix[0]:.2e} is not decisively "
            f"above the pressure-driven trace {dry.sW_matrix[0]:.2e}")
        print(f"ok test_capillarity_is_what_fills_the_matrix "
              f"(matrix sW: {wet.sW_matrix[0]:.3f} with pc, "
              f"{dry.sW_matrix[0]:.1e} without)")

    test_water_dp_equilibrates_with_no_forcing()
    test_water_dp_matrix_lags_the_fracture()
    test_water_dp_reduces_to_single_porosity()
    test_two_phase_dp_imbibition()
    test_capillarity_is_what_fills_the_matrix()
    print("\nAll tests passed!")
