"""Multiphase poroelasticity: black-oil coupled to mechanics.

MRST's `MechOilWaterModel` and `MechBlackOilModel`. Here they are the same
class, because the only thing poroelasticity changes about a flow model is
its pore volume - so `MechBlackOilModel` is `GenericBlackOilModel` with
`PoroMechanicalPoreVolume` installed and the displacement added as a primary
variable. Two phases or three, live oil or not, wells and all, are inherited
unchanged from a model already validated against MATLAB to 1e-10.

That is the third time the state-function graph has paid for itself, and the
cheapest: the multiphase physics needed no re-derivation at all.

## A warning about MRST's oil-water version

`equationsOilWaterMech.m` computes

    effPorVol = rock.poro .* (G.cells.volumes .* pvMult) + rock.alpha .* mechTerm

where `mechTerm` is `div(u)`, a *dimensionless* strain. Its sibling equations
do not:

    equationsWaterMech.m:    effPorVol = G.cells.volumes .* (poro.*pvMult + alpha.*mechTerm)
    equationsBlackOilMech.m: effPorVol = G.cells.volumes .* (poro.*pvMult + alpha.*mechTerm)

So the oil-water model is **missing the cell volume on the Biot term**: it adds
a bare strain to a volume. The coupling is therefore too weak by a factor of
the cell volume, and - being dimensionally inconsistent - its strength depends
on the grid's unit of length. On a grid of unit cells the bug is invisible,
which is presumably how it survived.

PyMRST computes it correctly. Pass `mrst_oilwater_bug=True` to reproduce
MRST's number, exactly as `LimAzizShapeFactor` does for its own MRST defect.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from pymrst.gridprocessing.grid import Grid
from pymrst.modules.ad_blackoil.models.generic_black_oil_model import GenericBlackOilModel
from pymrst.modules.ad_core.statefunctions.state_function import State
from pymrst.modules.ad_mechanics.mechanics import MechanicModel
from pymrst.modules.ad_mechanics.poromech import PoroMechanicalPoreVolume
from pymrst.params.rock import Rock
from pymrst.params.wells_and_bc import Well
from pymrst.utils.adi import init_variables_adi, value


class MechBlackOilModel(GenericBlackOilModel):
    """Black-oil (2- or 3-phase) fully coupled to poroelastic mechanics."""

    def __init__(
        self,
        g: Grid,
        rock: Rock,
        fluid,
        wells: Sequence[Well],
        mech: MechanicModel,
        disgas: bool = False,
        gravity: float = 0.0,
        tol_mech: float = 1e-9,
        mrst_oilwater_bug: bool = False,
        **kwargs,
    ):
        super().__init__(g, rock, fluid, wells, disgas=disgas,
                         gravity=gravity, **kwargs)
        self.mech = mech
        self.alpha = mech.alpha
        self.tol_mech = tol_mech
        self.PVTPropertyFunctions.set_state_function(
            "PoreVolume",
            PoroMechanicalPoreVolume(
                self, scale_mech_by_volume=not mrst_oilwater_bug
            ),
        )
        self.validate_model()

    def _q_names(self) -> List[str]:
        return ["qWs", "qOs", "qGs"][:self.n_phases]

    # -- primary variables ---------------------------------------------------
    def init_state_ad(self, st: State) -> State:
        """As the flow model, plus the displacement as the last variable."""
        q_names = self._q_names()
        if not self.disgas:
            prim = init_variables_adi(
                st["pressure"], st["sW"], *[st[q] for q in q_names],
                st["bhp"], st["xd"],
            )
            out = State(pressure=prim[0], sW=prim[1])
            rest = prim[2:]
        else:
            undersat = st["status"] == 1
            x_val = np.where(undersat, st["rs"], st["sG"])
            prim = init_variables_adi(
                st["pressure"], st["sW"], x_val,
                *[st[q] for q in q_names], st["bhp"], st["xd"],
            )
            p, x = prim[0], prim[2]
            rs_sat = self.raw_fluid.rs_sat(p)
            out = State(
                pressure=p, sW=prim[1],
                rs=(x * undersat.astype(float)
                    + rs_sat * (~undersat).astype(float)),
                sG=x * (~undersat).astype(float),
                status=st["status"],
            )
            rest = prim[3:]

        for i, q in enumerate(q_names):
            out[q] = rest[i]
        out["bhp"] = rest[len(q_names)]
        out["xd"] = rest[len(q_names) + 1]
        return out

    def _state0_props(self, st0: State) -> State:
        out = super()._state0_props(st0)
        out["xd"] = st0["xd"]
        out["u"] = st0["u"]
        return out

    # -- equations -----------------------------------------------------------
    def get_equations(self, state0: State, state: State, dt: float):
        st = self.init_state_ad(state)
        # remembered for `get_convergence`, which needs b and pv to form
        # MRST's CNV norm and is handed only the residuals
        self._conv_state = st
        st0 = self._state0_props(state0)

        eqs = self.conservation_equations(st0, st, dt)
        q_s = [st[q] for q in self._q_names()]
        cq_s, well_eqs, ctrl = self.facility.well_equations(
            self, st, q_s, st["bhp"], state0=st0
        )
        eqs = self.facility.apply_to_reservoir(eqs, cq_s)
        mech = self.mech.equations(st["xd"], st["pressure"])
        return eqs + well_eqs + [ctrl, mech]

    def get_convergence(self, eqs, dt: float) -> Tuple[bool, Dict[str, float]]:
        nph = self.n_phases
        ok, res = super().get_convergence(eqs[:2 * nph + 1], dt)
        res["mech"] = float(np.max(np.abs(value(eqs[-1]))))
        return (ok and res["mech"] <= self.tol_mech), res

    # -- Newton update -------------------------------------------------------
    def update_state(self, st: State, increment: np.ndarray) -> State:
        n_free = self.mech.n_free
        new = super().update_state(st, increment[:-n_free])
        new["xd"] = st["xd"] + increment[-n_free:]
        new["u"] = self.mech.full_displacement(new["xd"])
        new.clear_cache()
        return new

    def seed_state(self, st: State) -> State:
        """Give a flow state the mechanics fields it now needs."""
        st = st.copy()
        st["xd"] = self.mech.solve(st["pressure"])
        st["u"] = self.mech.full_displacement(st["xd"])
        return st


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_core.simulators.nonlinear_solver import NonLinearSolver
    from pymrst.modules.ad_props.init_simple_adi_fluid import init_simple_adi_fluid
    from pymrst.modules.vemmech.vem import (
        ElasticBC,
        create_augmented_grid,
        enu2c,
    )
    from pymrst.params.rock import make_rock
    from pymrst.params.wells_and_bc import add_well
    from pymrst.utils.units import BARSA, CENTI, DARCY, DAY, MILLI, POISE

    def build(cell_size=(10.0, 10.0, 5.0), nx=5, ny=5, nz=3, **kw):
        """A poroelastic box on rollers, injector and producer at the corners."""
        dims = [nx, ny, nz]
        phys = [nx * cell_size[0], ny * cell_size[1], nz * cell_size[2]]
        g = create_augmented_grid(compute_geometry(cart_grid(dims, phys)))
        rock = make_rock(g, 100 * MILLI * DARCY, 0.25)
        fluid = init_simple_adi_fluid(
            mu=np.array([1.0, 5.0]) * CENTI * POISE, rho=[1000.0, 800.0],
            n=[2, 2], c=[1e-6 / BARSA, 1e-4 / BARSA], p_ref=200 * BARSA,
        )

        pts = g.nodes.coords
        tol = 1e-9
        xf = np.flatnonzero((pts[:, 0] < tol) | (pts[:, 0] > phys[0] - tol))
        yf = np.flatnonzero((pts[:, 1] < tol) | (pts[:, 1] > phys[1] - tol))
        zf = np.flatnonzero(pts[:, 2] > phys[2] - tol)
        nodes = np.concatenate([xf, yf, zf])
        mask = np.vstack([
            np.tile([True, False, False], (xf.size, 1)),
            np.tile([False, True, False], (yf.size, 1)),
            np.tile([False, False, True], (zf.size, 1)),
        ])
        el_bc = ElasticBC(disp_nodes=nodes, disp_mask=mask,
                          disp_vals=np.zeros((nodes.size, 3)))
        C = enu2c(np.full(g.cells.num, 1e9), np.full(g.cells.num, 0.25))
        mech = MechanicModel(g, C, alpha=1.0, el_bc=el_bc)

        w = add_well(None, g, rock, np.array([0]), well_type="rate",
                     val=1e-3, compi=np.array([1.0, 0.0]), sign=1)
        w = add_well(w, g, rock, np.array([g.cells.num - 1]), well_type="bhp",
                     val=150 * BARSA, compi=np.array([0.0, 1.0]), sign=-1)

        model = MechBlackOilModel(g, rock, fluid, w, mech, **kw)
        st = State(
            pressure=np.full(g.cells.num, 200.0 * BARSA),
            sW=np.full(g.cells.num, 0.2),
            qWs=np.array([1e-3, 0.0]), qOs=np.array([0.0, -1e-3]),
            bhp=np.array([200.0 * BARSA, 150.0 * BARSA]),
        )
        return g, model, model.seed_state(st)

    def test_coupled_two_phase_runs():
        """Two-phase flow with wells, coupled to mechanics, converges."""
        g, m, st = build()
        solver = NonLinearSolver()
        for _ in range(3):
            st, rep = solver.solve_ministep(m, st, st.copy(), 30 * DAY)
            assert rep.converged, rep.residuals[-1]
        assert np.isclose(st["qWs"][0] + st["qOs"][0], 1e-3, rtol=1e-8)
        assert st["sW"][0] > 0.5, "the injector must flood its cell"
        assert "mech" in rep.residuals[-1]
        print(f"ok test_coupled_two_phase_runs ({rep.iterations} its)")

    def test_injection_dilates_the_rock():
        """Injecting raises pressure, which pushes the rock open - the pore
        volume must exceed the flow-only one where pressure has risen."""
        g, m, st = build()
        solver = NonLinearSolver()
        for _ in range(3):
            st, rep = solver.solve_ministep(m, st, st.copy(), 30 * DAY)
            assert rep.converged
        pv = m.get_props(st, "PoreVolume")
        flow_only = g.cells.volumes * m.rock.poro.ravel()
        divu = m.mech.volumetric_strain_full(st["u"])
        assert np.allclose(pv - flow_only, g.cells.volumes * divu, rtol=1e-10)
        assert divu[0] > 0, "the pressurised injector cell must dilate"
        assert pv[0] > flow_only[0]
        print("ok test_injection_dilates_the_rock")

    def test_mrst_oilwater_bug_is_a_missing_cell_volume():
        """MRST's equationsOilWaterMech omits G.cells.volumes on the Biot
        term. Reproduce it, and show the two agree only when the cells have
        unit volume - which is exactly what a missing volume factor predicts,
        and what makes the defect invisible on a unit grid."""
        vol = 10.0 * 10.0 * 5.0

        _, m_ok, st = build()
        _, m_bug, _ = build(mrst_oilwater_bug=True)
        pv_ok = m_ok.get_props(st, "PoreVolume")
        pv_bug = m_bug.get_props(st.copy(), "PoreVolume")
        divu = m_ok.mech.volumetric_strain_full(st["u"])
        # the difference is exactly (V - 1) * alpha * div(u)
        assert np.allclose(pv_ok - pv_bug, (vol - 1.0) * divu, rtol=1e-10)

        # on unit cells the bug vanishes
        _, u_ok, st_u = build(cell_size=(1.0, 1.0, 1.0))
        _, u_bug, _ = build(cell_size=(1.0, 1.0, 1.0), mrst_oilwater_bug=True)
        assert np.allclose(u_ok.get_props(st_u, "PoreVolume"),
                           u_bug.get_props(st_u.copy(), "PoreVolume"))
        print("ok test_mrst_oilwater_bug_is_a_missing_cell_volume")

    test_coupled_two_phase_runs()
    test_injection_dilates_the_rock()
    test_mrst_oilwater_bug_is_a_missing_cell_volume()
    print("\nAll tests passed!")
