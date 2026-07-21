"""ExplicitFlowStateBuilder (MRST ad-core/statefunctions/flux/flow_state).

Ports MRST's `ExplicitFlowStateBuilder.m`: an IMPES-style flow state where the
named flux terms are frozen at the previous (explicit) state, while pressure
stays implicit. The default frozen terms are the face mobilities and the
gravity potential - the hyperbolic parts whose implicit treatment couples every
cell to its neighbours.

Freezing is done by evaluating those terms on ``state0`` (whose primaries are
numeric, so the result carries no derivative w.r.t. the new unknowns) and
seeding them into the flow state's property cache. When the graph then assembles
the flux it reads the frozen mobility and the implicit potential - exactly the
IMPES split.

The stable time step for an explicit scheme is CFL-limited. That estimate
(`estimateSaturationCFL` / `estimateCompositionCFL`) is AC21 and not yet ported,
so `get_maximum_timestep` returns the initial step on the first step and,
absent an injected estimator, does not otherwise cap the step. The *hybridizing*
mechanism - the actual explicit flux state - is complete and validated here.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

import numpy as np

from pymrst.modules.ad_core.statefunctions.flux.flow_state.flow_state_builder import (  # noqa: E501
    FlowStateBuilder,
    owning_grouping,
    seed_property,
)
from pymrst.modules.ad_core.statefunctions.state_function import State
from pymrst.utils.adi import value

DAY = 86400.0


class ExplicitFlowStateBuilder(FlowStateBuilder):
    """IMPES-style flow state: named flux terms frozen at the old state."""

    def __init__(
        self,
        verbose: bool = False,
        saturation_cfl: float = 0.9,
        composition_cfl: float = 0.9,
        explicit_flux: Optional[List[str]] = None,
        explicit_flow: Optional[List[str]] = None,
        explicit_pvt: Optional[List[str]] = None,
        initial_step: float = 1.0 * DAY,
        change_timestep: bool = True,
        cfl_estimator: Optional[Callable] = None,
    ):
        super().__init__(verbose=verbose)
        self.saturation_cfl = saturation_cfl
        self.composition_cfl = composition_cfl
        # FaceComponentMobility is in MRST's default set but is not yet ported
        # (AC1/AC6); it is skipped below if the model does not define it.
        self.explicit_flux = (["FaceMobility", "FaceComponentMobility",
                               "GravityPotentialDifference"]
                              if explicit_flux is None else list(explicit_flux))
        self.explicit_flow = list(explicit_flow or [])
        self.explicit_pvt = list(explicit_pvt or [])
        self.initial_step = initial_step
        self.change_timestep = change_timestep
        self.cfl_estimator = cfl_estimator

    # -- the explicit terms, grouped ----------------------------------------
    def explicit_groups(self, model: Any):
        """Yield ``(grouping, [prop, ...])`` for every explicit term the model
        actually defines, skipping any that are not present."""
        wishlist = [
            ("PVTPropertyFunctions", self.explicit_pvt),
            ("FlowPropertyFunctions", self.explicit_flow),
            ("FlowDiscretization", self.explicit_flux),
        ]
        for _gname, props in wishlist:
            present = []
            for p in props:
                try:
                    grp = owning_grouping(model, p)
                except KeyError:
                    continue                     # not ported / not in this model
                present.append((grp, p))
            for grp, p in present:
                yield grp, p

    def is_first_timestep(self, state: State) -> bool:
        f = state.get("flux") if hasattr(state, "get") else None
        return f is None or not np.any(np.abs(np.asarray(value(f))) > 0)

    def get_maximum_timestep(self, model: Any, state: State, state0: State,
                             dt: float, forces: Any = None) -> float:
        if not self.change_timestep:
            return float("inf")
        if self.is_first_timestep(state):
            return self.initial_step
        if self.cfl_estimator is None:
            # CFL estimation is AC21; without it we cannot cap the step here.
            return float("inf")
        cfl_s, cfl_c = self.cfl_estimator(model, state, dt, forces)
        dt_s = np.min(1.0 / np.maximum(np.max(cfl_s, axis=-1) / self.saturation_cfl,
                                       1e-30))
        dt_c = np.min(1.0 / np.maximum(np.max(cfl_c, axis=-1) / self.composition_cfl,
                                       1e-30))
        return float(min(dt_s, dt_c))

    def build(self, model: Any, state: State, state0: State, dt: float,
              fd: Any = None) -> State:
        """A flow state whose explicit terms are frozen at ``state0``."""
        flow_state = state.copy()               # shared AD primaries, fresh cache
        for grp, prop in self.explicit_groups(model):
            frozen = grp.get(model, state0, prop)          # numeric -> explicit
            seed_property(flow_state, grp, prop, _detach(frozen))
        return flow_state


def _detach(x: Any) -> Any:
    """A pure-numeric copy: a frozen term must carry no AD derivative."""
    if isinstance(x, list):
        return [None if xi is None else _detach(xi) for xi in x]
    return np.asarray(value(x)).copy()


__all__ = ["ExplicitFlowStateBuilder"]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
        GenericBlackOilModel,
    )
    from pymrst.modules.ad_props import init_simple_adi_fluid
    from pymrst.params import add_well, make_rock
    from pymrst.utils.units import BARSA, CENTI, DARCY, MILLI, POISE

    def _model_and_states():
        g = compute_geometry(cart_grid([6, 5, 2], [300.0, 250.0, 20.0]))
        rock = make_rock(g, 100 * MILLI * DARCY, 0.25)
        fluid = init_simple_adi_fluid(
            mu=np.array([1.0, 5.0]) * CENTI * POISE, rho=[1000.0, 800.0],
            n=[2, 2], c=[1e-6 / BARSA, 1e-4 / BARSA], p_ref=200 * BARSA)
        w = add_well(None, g, rock, np.array([0]), well_type="rate", val=2e-3,
                     compi=np.array([1.0, 0.0]), sign=1)
        w = add_well(w, g, rock, np.array([g.cells.num - 1]), well_type="bhp",
                     val=150 * BARSA, compi=np.array([0.0, 1.0]), sign=-1)
        model = GenericBlackOilModel(g, rock, fluid, w, gravity=9.80665)
        st0 = State(pressure=np.linspace(230, 180, g.cells.num) * BARSA,
                    sW=np.full(g.cells.num, 0.4),
                    qWs=np.array([2e-3]), qOs=np.array([-2e-3]),
                    bhp=np.array([150 * BARSA]))
        # a perturbed *new* implicit state, so state0 and state differ
        st_new = State(pressure=np.linspace(228, 178, g.cells.num) * BARSA,
                       sW=np.full(g.cells.num, 0.55),
                       qWs=np.array([2e-3]), qOs=np.array([-2e-3]),
                       bhp=np.array([150 * BARSA]))
        return model, model.init_state_ad(st_new), st0

    def test_frozen_face_mobility_is_explicit():
        """FaceMobility in the flow state equals FaceMobility(state0) and
        carries no Jacobian - it is frozen at the old time level."""
        model, st, st0 = _model_and_states()
        fs = ExplicitFlowStateBuilder().build(model, st, st0, dt=30.0)
        frozen = model.get_props(fs, "FaceMobility")
        old = model.get_props(st0, "FaceMobility")
        for ph in range(model.n_phases):
            assert np.allclose(value(frozen[ph]), value(old[ph]))
            assert not hasattr(frozen[ph], "jac"), "frozen term must be numeric"
        print("ok test_frozen_face_mobility_is_explicit")

    def test_pressure_stays_implicit():
        """The pressure gradient is NOT frozen - it keeps its Jacobian, so the
        IMPES split (mobility explicit, pressure implicit) holds."""
        model, st, st0 = _model_and_states()
        fs = ExplicitFlowStateBuilder().build(model, st, st0, dt=30.0)
        pg = model.get_props(fs, "PressureGradient")
        assert hasattr(pg[0], "jac"), "pressure gradient must stay implicit"
        # and it matches the implicit pressure gradient of the new state
        pg_imp = model.get_props(st, "PressureGradient")
        assert np.allclose(value(pg[0]), value(pg_imp[0]))
        print("ok test_pressure_stays_implicit")

    def test_impes_flux_matches_manual_split():
        """The explicit flow-state PhaseFlux equals the manual IMPES flux:
        -T * FaceMobility(state0) * PhasePotentialDifference(state)."""
        model, st, st0 = _model_and_states()
        fs = ExplicitFlowStateBuilder(
            explicit_flux=["FaceMobility"]).build(model, st, st0, dt=30.0)
        flux = model.get_props(fs, "PhaseFlux")
        t = model.operators.T
        mob0 = model.get_props(st0, "FaceMobility")
        pot = model.get_props(st, "PhasePotentialDifference")
        for ph in range(model.n_phases):
            manual = -t * value(mob0[ph]) * pot[ph]
            assert np.allclose(value(flux[ph]), value(manual))
            assert np.allclose((flux[ph] - manual).jac[0].toarray(), 0.0)
        print("ok test_impes_flux_matches_manual_split")

    def test_missing_terms_are_skipped():
        """Requesting a not-ported term (FaceComponentMobility) is skipped,
        not an error - the default set stays usable before AC6."""
        model, st, st0 = _model_and_states()
        # default explicit_flux includes FaceComponentMobility, absent here
        fs = ExplicitFlowStateBuilder().build(model, st, st0, dt=30.0)
        assert "FaceComponentMobility" not in fs.containers.get("FluxDisc", {})
        assert "FaceMobility" in fs.containers.get("FluxDisc", {})
        print("ok test_missing_terms_are_skipped")

    def test_first_step_uses_initial_step():
        """With no flux on the state, the max step is the configured initial."""
        model, st, st0 = _model_and_states()
        b = ExplicitFlowStateBuilder(initial_step=3.0 * DAY)
        assert b.get_maximum_timestep(model, st, st0, dt=99.0) == 3.0 * DAY
        print("ok test_first_step_uses_initial_step")

    test_frozen_face_mobility_is_explicit()
    test_pressure_stays_implicit()
    test_impes_flux_matches_manual_split()
    test_missing_terms_are_skipped()
    test_first_step_uses_initial_step()
    print("\nAll tests passed!")
