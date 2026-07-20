"""A tracer, added to black-oil without touching the flow model.

This module exists to prove one claim: that the component framework does what
it is for. `TracerBlackOilModel` is `GenericBlackOilModel` plus

  - one entry in `self.components` (a `ConcentrationComponent` carried by the
    water), and
  - one primary variable (its concentration).

Nothing else. The conservation equation, the flux, the upwinding, the wells,
the Newton solver and the linear solve are all untouched - they never knew how
many components there were, because they loop over `model.components`.

A passive tracer is polymer with the interesting parts removed: it rides in the
water, it is conserved, and it does not change the flow. That makes it exactly
the right thing to test the *machinery* with, because its answer is known in
advance - a tracer must not perturb the water it rides in, and it must be
conserved to round-off. `ad-eor` adds the viscosity multiplier and the
adsorption; it does not add any of the plumbing below.
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np

from pymrst.gridprocessing.grid import Grid
from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
    GenericBlackOilModel,
)
from pymrst.modules.ad_core.statefunctions.components.components import (
    ConcentrationComponent,
    blackoil_components,
)
from pymrst.modules.ad_core.statefunctions.state_function import State
from pymrst.params.rock import Rock
from pymrst.params.wells_and_bc import Well
from pymrst.utils.adi import init_variables_adi, value, vertcat_adi


class TracerBlackOilModel(GenericBlackOilModel):
    """Black-oil plus a passive tracer dissolved in the water."""

    def __init__(self, g: Grid, rock: Rock, fluid, wells: Sequence[Well],
                 tracer_name: str = "tracer", inj_concentration=None,
                 **kwargs):
        super().__init__(g, rock, fluid, wells, **kwargs)

        # The entire addition: one more component, carried by the water.
        self.components = self.components + [
            ConcentrationComponent(tracer_name, phase_index=0, state_key="cT")
        ]
        self.tracer_name = tracer_name
        # What each well injects, per well (0 for producers).
        self.inj_concentration = (
            np.zeros(self.facility.n_wells) if inj_concentration is None
            else np.asarray(inj_concentration, dtype=float).ravel()
        )

        # The groupings were built before the component existed, so rebuild
        # them: the mass and flux functions must now see it.
        self._rebuild_groupings()
        self.validate_model()

    def _rebuild_groupings(self) -> None:
        from pymrst.modules.ad_core.statefunctions.flow_discretization import (
            FlowDiscretization,
        )
        from pymrst.modules.ad_core.statefunctions.flow_property_functions import (
            FlowPropertyFunctions,
        )
        from pymrst.modules.ad_core.statefunctions.pvt_property_functions import (
            PVTPropertyFunctions,
        )
        self.PVTPropertyFunctions = PVTPropertyFunctions(self)
        self.FlowPropertyFunctions = FlowPropertyFunctions(self)
        self.FlowDiscretization = FlowDiscretization(self)

    # -- one extra primary variable ------------------------------------------
    def init_state_ad(self, st: State) -> State:
        """As black-oil, with the tracer concentration appended."""
        nph = self.n_phases
        q_names = ["qWs", "qOs", "qGs"][:nph]
        assert not self.disgas, "the tracer demo is two- or three-phase dead oil"

        prim = init_variables_adi(
            st["pressure"], st["sW"], *[st[q] for q in q_names], st["bhp"],
            st["cT"],
        )
        out = State(pressure=prim[0], sW=prim[1])
        for i, q in enumerate(q_names):
            out[q] = prim[2 + i]
        out["bhp"] = prim[2 + nph]
        out["cT"] = prim[3 + nph]
        return out

    def _state0_props(self, st0: State) -> State:
        out = super()._state0_props(st0)
        out["cT"] = st0["cT"]
        return out

    # -- equations -----------------------------------------------------------
    def get_equations(self, state0: State, state: State, dt: float):
        st = self.init_state_ad(state)
        # remembered for `get_convergence`, which needs b and pv to form
        # MRST's CNV norm and is handed only the residuals
        self._conv_state = st
        st0 = self._state0_props(state0)
        nph = self.n_phases

        # The conservation equations already include the tracer: there are
        # n_components of them, not n_phases, and nothing here said so.
        eqs = self.conservation_equations(st0, st, dt)
        assert len(eqs) == self.n_components

        q_s = [st[q] for q in ["qWs", "qOs", "qGs"][:nph]]
        cq_s, well_eqs, ctrl = self.facility.well_equations(
            self, st, q_s, st["bhp"], state0=st0
        )
        # The tracer enters and leaves with the water it rides in: injected at
        # the well's concentration, produced at the cell's.
        eqs = self.facility.apply_to_reservoir(eqs[:nph], cq_s) + eqs[nph:]
        for k, w in enumerate(self.wells):
            wc = w.cells
            c_in = (self.inj_concentration[k] if w.sign > 0
                    else st["cT"][wc])
            eqs[nph][wc] = eqs[nph][wc] - c_in * cq_s[k][0]

        return eqs + well_eqs + [ctrl]

    def get_convergence(self, eqs, dt: float):
        nph = self.n_phases
        ok, res = super().get_convergence(
            [eqs[i] for i in range(nph)] + list(eqs[nph + 1:]), dt)
        res["tracer"] = float(np.max(np.abs(value(eqs[nph])))
                              ) * dt / self.pv_ref.mean()
        return (ok and res["tracer"] <= self.tol_cnv), res

    # -- update --------------------------------------------------------------
    def update_state(self, st: State, increment: np.ndarray) -> State:
        nc = self.G.cells.num
        new = super().update_state(st, increment[:-nc])
        new["cT"] = np.clip(st["cT"] + increment[-nc:], 0.0, None)
        new.clear_cache()
        return new


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_core.simulators.nonlinear_solver import NonLinearSolver
    from pymrst.modules.ad_props.simple_adi_fluid import init_simple_adi_fluid
    from pymrst.params.rock import make_rock, pore_volume
    from pymrst.params.wells_and_bc import add_well
    from pymrst.utils.units import BARSA, CENTI, DARCY, DAY, MILLI, POISE

    def _case(with_tracer: bool, c_inj: float = 1.0):
        g = compute_geometry(cart_grid([10, 10, 1], [200.0, 200.0, 10.0]))
        rock = make_rock(g, 100 * MILLI * DARCY, 0.25)
        fluid = init_simple_adi_fluid(
            mu=np.array([1.0, 5.0]) * CENTI * POISE, rho=[1000.0, 800.0],
            n=[2, 2], c=[1e-6 / BARSA, 1e-4 / BARSA], p_ref=200 * BARSA,
        )
        w = add_well(None, g, rock, np.array([0]), well_type="rate", val=2e-3,
                     compi=np.array([1.0, 0.0]), sign=1)
        w = add_well(w, g, rock, np.array([g.cells.num - 1]), well_type="bhp",
                     val=150 * BARSA, compi=np.array([0.0, 1.0]), sign=-1)

        base = dict(
            pressure=np.full(g.cells.num, 200.0 * BARSA),
            sW=np.full(g.cells.num, 0.2),
            qWs=np.array([2e-3, 0.0]), qOs=np.array([0.0, -2e-3]),
            bhp=np.array([200.0 * BARSA, 150.0 * BARSA]),
        )
        if with_tracer:
            m = TracerBlackOilModel(g, rock, fluid, w,
                                    inj_concentration=[c_inj, 0.0])
            st = State(cT=np.zeros(g.cells.num), **base)
        else:
            m = GenericBlackOilModel(g, rock, fluid, w)
            st = State(**base)
        return g, m, st

    DTS = [30 * DAY] * 5

    def test_the_model_gained_a_component_and_an_equation():
        """The claim, stated as an assertion."""
        _, m, _ = _case(True)
        assert m.n_components == 3 and m.n_phases == 2
        assert m.component_names() == ["water", "oil", "tracer"]
        assert "cT" in m.component_state_keys()
        print(f"ok test_the_model_gained_a_component_and_an_equation "
              f"({m.n_phases} phases, {m.n_components} components)")

    def test_the_flow_equations_cannot_see_the_tracer():
        """THE test, and it is exact rather than statistical.

        A passive tracer must not perturb the water it rides in *at all*. Not
        "to solver tolerance" - at all: the flow residual does not depend on
        `cT`, so changing `cT` must leave it bit-identical. Anything else means
        the component framework is leaking into the flow.
        """
        _, m, st = _case(True, c_inj=1.0)
        st["sW"] = np.linspace(0.2, 0.8, m.G.cells.num)   # a real front

        st_a = st.copy()
        st_a["cT"] = np.zeros(m.G.cells.num)
        st_b = st.copy()
        st_b["cT"] = np.linspace(0.0, 1.0, m.G.cells.num)  # wildly different

        eq_a = m.get_equations(st_a, st_a.copy(), 30 * DAY)
        eq_b = m.get_equations(st_b, st_b.copy(), 30 * DAY)

        nph = m.n_phases
        for i in range(nph):                      # the flow equations
            d = np.abs(value(eq_a[i]) - value(eq_b[i])).max()
            assert d == 0.0, (
                f"the tracer moved flow equation {i} by {d:.2e} - it must not "
                "move it at all"
            )
        # ...while the tracer's own equation of course did change
        assert np.abs(value(eq_a[nph]) - value(eq_b[nph])).max() > 0
        print("ok test_the_flow_equations_cannot_see_the_tracer "
              "(flow residual bit-identical under any tracer field)")

    def test_a_passive_tracer_does_not_disturb_the_flow():
        """And end to end: the same schedule with and without the tracer gives
        the same flow, to the solver's tolerance - which is all a converged
        Newton iterate can promise."""
        _, m0, st0 = _case(False)
        _, m1, st1 = _case(True)
        for m in (m0, m1):
            m.tol_cnv, m.tol_mb, m.tol_well = 1e-13, 1e-15, 1e-13
        solver = NonLinearSolver(max_iterations=40)

        worst = 0.0
        for dt in DTS:
            st0, r0 = solver.solve_ministep(m0, st0, st0.copy(), dt)
            st1, r1 = solver.solve_ministep(m1, st1, st1.copy(), dt)
            assert r0.converged and r1.converged
            for key in ("pressure", "sW", "bhp"):
                scale = max(np.abs(st0[key]).max(), 1e-30)
                worst = max(worst, np.abs(st1[key] - st0[key]).max() / scale)
        assert worst < 1e-11, f"the tracer changed the flow by {worst:.2e}"
        print(f"ok test_a_passive_tracer_does_not_disturb_the_flow "
              f"(max rel diff {worst:.1e})")

    def test_the_tracer_is_conserved():
        """What went in is in the rock or came back out."""
        g, m, st = _case(True, c_inj=1.0)
        solver = NonLinearSolver()

        injected = 0.0
        produced = 0.0
        states = [st]
        for dt in DTS:
            st, rep = solver.solve_ministep(m, st, st.copy(), dt)
            assert rep.converged
            states.append(st)
            injected += float(m.inj_concentration[0] * st["qWs"][0]) * dt
            produced += -float(st["cT"][m.wells[1].cells[0]]
                               * st["qWs"][1]) * dt

        def stored(s):
            pv = m.get_props(s, "PoreVolume")
            b = m.get_props(s, "ShrinkageFactors")[0]
            return float(np.sum(pv * value(b) * s["sW"] * s["cT"]))

        change = stored(states[-1]) - stored(states[0])
        err = abs(injected - produced - change) / max(injected, 1e-30)
        assert err < 1e-8, (
            f"tracer not conserved: in {injected:.4e} - out {produced:.4e} "
            f"!= stored {change:.4e} (rel {err:.2e})"
        )
        print(f"ok test_the_tracer_is_conserved (rel error {err:.1e})")

    def test_the_tracer_follows_the_water():
        """It floods out from the injector and never gets ahead of the water."""
        g, m, st = _case(True, c_inj=1.0)
        solver = NonLinearSolver()
        for dt in DTS:
            st, rep = solver.solve_ministep(m, st, st.copy(), dt)
            assert rep.converged

        assert st["cT"][0] > 0.5, "the injector cell must be full of tracer"
        assert st["cT"].max() <= 1.0 + 1e-10, "it cannot exceed what was injected"
        assert np.all(st["cT"] >= 0.0)
        # nowhere the water has not reached can hold tracer
        untouched = st["sW"] < 0.2 + 1e-9
        assert np.all(st["cT"][untouched] < 1e-9), (
            "tracer appeared where the water did not"
        )
        print(f"ok test_the_tracer_follows_the_water "
              f"(front: c={st['cT'][0]:.3f} at the injector, "
              f"{st['cT'][-1]:.2e} at the producer)")

    test_the_model_gained_a_component_and_an_equation()
    test_the_flow_equations_cannot_see_the_tracer()
    test_a_passive_tracer_does_not_disturb_the_flow()
    test_the_tracer_is_conserved()
    test_the_tracer_follows_the_water()
    print("\nAll tests passed!")
