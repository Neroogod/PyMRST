"""MRST `ad-blackoil/models/WaterModel.m`."""

from typing import Dict, List, Sequence, Tuple

import numpy as np

from pymrst.gridprocessing.grid import Grid
from pymrst.modules.ad_blackoil.utils.equations_water import equations_water
from pymrst.modules.ad_core.models.facilities.facility_model import FacilityModel
from pymrst.modules.ad_core.utils.convergence import (
    cnv_mb_convergence,
    well_control_residual,
)
from pymrst.modules.ad_core.models.reservoir_model import ReservoirModel
from pymrst.modules.ad_core.utils.operators import setup_operators_tpfa
from pymrst.modules.ad_core.statefunctions.pvt_property_functions import (
    PhaseFluid,
    water_phase_fluid,
)
from pymrst.modules.ad_core.statefunctions.state_function import State
from pymrst.params.rock import Rock, pore_volume
from pymrst.params.wells_and_bc import Well
from pymrst.utils.adi import init_variables_adi, value


def _as_water_phase_fluid(fluid) -> PhaseFluid:
    if isinstance(fluid, PhaseFluid):
        if fluid.n_phases != 1:
            raise ValueError("WaterModel requires a single-phase fluid")
        return fluid
    if hasattr(fluid, "b_w"):
        rho = float(np.asarray(getattr(fluid, "rhoS", [1000.0]))[0])
        mu = float(np.asarray(getattr(fluid, "mu", [1.0e-3]))[0])
        return water_phase_fluid(fluid.b_w, mu, rho)
    if callable(fluid):
        return water_phase_fluid(fluid, 1.0e-3, 1000.0)
    raise TypeError(
        "WaterModel fluid must be a PhaseFluid, an ADFluid-like object with "
        "b_w/mu/rhoS, or a callable b(p)"
    )


class WaterModel(ReservoirModel):
    """Single-phase water model assembled through the AD-core graph."""

    def __init__(
        self,
        g: Grid,
        rock: Rock,
        fluid,
        wells: Sequence[Well] = (),
        gravity: float = 0.0,
        dp_max_rel: float = 0.2,
        tol_cnv: float = 1e-8,
        tol_mb: float = 1e-12,
        tol_well: float = 1e-10,
        bc=None,
        src=None,
        mass_basis: bool = False,
    ):
        phase_fluid = _as_water_phase_fluid(fluid)
        operators = setup_operators_tpfa(g, rock)
        if gravity:
            z = g.cells.centroids[:, 2]
            gdz = gravity * (z[operators.N[:, 1]] - z[operators.N[:, 0]])
        else:
            gdz = np.zeros(operators.T.size)

        super().__init__(
            g, rock, phase_fluid, operators, pore_volume(g, rock), gdz,
            gravity=gravity, mass_basis=mass_basis,
        )
        self.raw_fluid = fluid
        self.bc = bc
        self.src = src
        self.facility = FacilityModel(wells, disgas=False, vapoil=False)
        self.dp_max_rel = float(dp_max_rel)
        self.tol_cnv = float(tol_cnv)
        self.tol_mb = float(tol_mb)
        self.tol_well = float(tol_well)
        self.validate_model()

    @property
    def wells(self) -> List[Well]:
        return self.facility.wells

    def initial_state(self, p0: float) -> State:
        nw = self.facility.n_wells
        qws = np.array([
            w.val * w.compi[0] if w.type == "rate" else 0.0
            for w in self.wells
        ], dtype=float)
        bhp = np.array([
            w.val if w.type == "bhp" else p0
            for w in self.wells
        ], dtype=float)
        return State(
            pressure=np.full(self.G.cells.num, float(p0)),
            qWs=qws if nw else np.zeros(0),
            bhp=bhp if nw else np.zeros(0),
        )

    def init_state_ad(self, st: State) -> State:
        p, qws, bhp = init_variables_adi(st["pressure"], st["qWs"], st["bhp"])
        return State(pressure=p, qWs=qws, bhp=bhp)

    def _state0_props(self, st0: State) -> State:
        out = State(pressure=st0["pressure"])
        if "well_flux" in st0:
            out["well_flux"] = st0["well_flux"]
        return out

    def get_equations(self, state0: State, state: State, dt: float):
        return equations_water(state0, state, self, dt)

    def _convergence_scaling(self):
        st = getattr(self, "_conv_state", None)
        if st is None:
            return 1.0, self.pv_ref
        b = self.get_props(st, "ShrinkageFactors")[0]
        pv = np.asarray(value(self.get_props(st, "PoreVolume")), dtype=float)
        return float(np.mean(1.0 / np.asarray(value(b), float))), pv

    def get_convergence(self, eqs, dt: float) -> Tuple[bool, Dict[str, float]]:
        cell_eq, well_eq, ctrl = (value(eqs[0]), value(eqs[1]), value(eqs[2]))
        b_avg, pv = self._convergence_scaling()
        cell_eq = cell_eq / self._basis_scale()[0]
        cnv, mb = cnv_mb_convergence([cell_eq], np.array([b_avg]), pv, dt)
        well, ctrl_res = well_control_residual([well_eq], ctrl, self.wells)

        converged = (
            cnv <= self.tol_cnv
            and mb <= self.tol_mb
            and well <= self.tol_well
            and ctrl_res <= self.tol_well
            and self.facility.within_limits
        )
        return converged, {"CNV": cnv, "MB": mb, "well": well,
                           "control": ctrl_res}

    def update_state(self, st: State, increment: np.ndarray) -> State:
        nc, nw = self.G.cells.num, self.facility.n_wells
        new = st.copy()
        cap = self.dp_max_rel * np.maximum(np.abs(st["pressure"]), 1.0)
        new["pressure"] = st["pressure"] + np.clip(increment[:nc], -cap, cap)
        off = nc
        new["qWs"] = st["qWs"] + increment[off:off + nw]
        new["bhp"] = st["bhp"] + increment[off + nw:off + 2 * nw]
        self.facility.update_switched_controls(new, ["qWs"])
        new.clear_cache()
        return new

    def after_convergence(self, state: State, state0: State,
                          dt: float) -> State:
        if self.facility.last_flux is not None:
            state["well_flux"] = [q.copy() for q in self.facility.last_flux]
        return state


__all__ = ["WaterModel"]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_core.simulators.nonlinear_solver import NonLinearSolver
    from pymrst.modules.ad_props.simple_adi_fluid import init_simple_adi_fluid
    from pymrst.params.rock import make_rock
    from pymrst.params.wells_and_bc import add_well
    from pymrst.utils.units import BARSA, CENTI, DARCY, DAY, MILLI, POISE

    def _model():
        g = compute_geometry(cart_grid([8, 1, 1], [80.0, 10.0, 10.0]))
        rock = make_rock(g, 100 * MILLI * DARCY, 0.25)
        f = init_simple_adi_fluid(
            mu=np.array([1.0, 5.0]) * CENTI * POISE,
            rho=[1000.0, 800.0],
            n=[1, 1],
            c=[1e-6 / BARSA, 0.0],
            p_ref=200 * BARSA,
        )
        w = add_well(None, g, rock, np.array([0]), well_type="rate",
                     val=1e-4, compi=np.array([1.0]), sign=1)
        w = add_well(w, g, rock, np.array([7]), well_type="bhp",
                     val=190 * BARSA, compi=np.array([1.0]), sign=-1)
        return WaterModel(g, rock, f, w)

    def test_water_model_runs_and_injects():
        m = _model()
        st0 = m.initial_state(200 * BARSA)
        solver = NonLinearSolver(max_iterations=20)
        st, rep = solver.solve_timestep(m, st0, DAY)
        assert rep.converged
        assert np.all(np.isfinite(st["pressure"]))
        assert np.isclose(st["qWs"][0], 1e-4, rtol=1e-10)
        assert np.isclose(st["bhp"][1], 190 * BARSA, rtol=1e-12)
        print("ok test_water_model_runs_and_injects")

    def test_mass_basis_matches_surface_basis():
        m1 = _model()
        m2 = _model()
        m2.mass_basis = True
        s1 = m1.initial_state(200 * BARSA)
        s2 = m2.initial_state(200 * BARSA)
        solver = NonLinearSolver(max_iterations=20)
        r1, rep1 = solver.solve_timestep(m1, s1, DAY)
        r2, rep2 = solver.solve_timestep(m2, s2, DAY)
        assert rep1.converged and rep2.converged
        assert np.allclose(r1["pressure"], r2["pressure"], rtol=1e-11)
        print("ok test_mass_basis_matches_surface_basis")

    test_water_model_runs_and_injects()
    test_mass_basis_matches_surface_basis()
    print("\nAll tests passed!")
