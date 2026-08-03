"""Fully coupled black-oil poromechanics (MRST layout owner)."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from pymrst.gridprocessing.grid import Grid
from pymrst.modules.ad_blackoil.models.generic_black_oil_model import GenericBlackOilModel
from pymrst.modules.ad_core.statefunctions.state_function import State
from pymrst.modules.ad_mechanics.models.mechanic_model import MechanicModel
from pymrst.modules.ad_mechanics.models.poromechanics import PoroMechanicalPoreVolume
from pymrst.params.rock import Rock
from pymrst.params.wells_and_bc import Well
from pymrst.utils.adi import init_variables_adi, value


class MechBlackOilModel(GenericBlackOilModel):
    """Black-oil phases and wells fully coupled to VEM mechanics."""

    primary_variable_names = ("pressure", "sW", "phase_state", "well_variables", "xd")

    def __init__(
        self, g: Grid, rock: Rock, fluid, wells: Sequence[Well],
        mech: MechanicModel, disgas: bool = False, gravity: float = 0.0,
        tol_mech: float = 1e-9, mrst_oilwater_bug: bool = False, **kwargs,
    ):
        super().__init__(g, rock, fluid, wells, disgas=disgas,
                         gravity=gravity, **kwargs)
        self.mech = mech
        self.alpha = mech.alpha
        self.tol_mech = tol_mech
        self.PVTPropertyFunctions.set_state_function(
            "PoreVolume", PoroMechanicalPoreVolume(
                self, scale_mech_by_volume=not mrst_oilwater_bug
            ),
        )
        self.validate_model()

    def _q_names(self) -> List[str]:
        return ["qWs", "qOs", "qGs"][:self.n_phases]

    def init_state_ad(self, st: State) -> State:
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
                rs=x * undersat.astype(float) + rs_sat * (~undersat).astype(float),
                sG=x * (~undersat).astype(float), status=st["status"],
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

    def get_equations(self, state0: State, state: State, dt: float):
        st = self.init_state_ad(state)
        self._conv_state = st
        st0 = self._state0_props(state0)
        eqs = self.conservation_equations(st0, st, dt)
        q_s = [st[q] for q in self._q_names()]
        cq_s, well_eqs, ctrl = self.facility.well_equations(
            self, st, q_s, st["bhp"], state0=st0
        )
        eqs = self.facility.apply_to_reservoir(eqs, cq_s)
        return eqs + well_eqs + [ctrl, self.mech.equations(st["xd"], st["pressure"])]

    def get_convergence(self, eqs, dt: float) -> Tuple[bool, Dict[str, float]]:
        ok, residuals = super().get_convergence(eqs[:2 * self.n_phases + 1], dt)
        residuals["mech"] = float(np.max(np.abs(value(eqs[-1]))))
        return ok and residuals["mech"] <= self.tol_mech, residuals

    def update_state(self, st: State, increment: np.ndarray) -> State:
        n_free = self.mech.n_free
        new = super().update_state(st, increment[:-n_free])
        new["xd"] = st["xd"] + increment[-n_free:]
        new["u"] = self.mech.full_displacement(new["xd"])
        new.clear_cache()
        return new

    def seed_state(self, st: State) -> State:
        st = st.copy()
        st["xd"] = self.mech.solve(st["pressure"])
        st["u"] = self.mech.full_displacement(st["xd"])
        return st

    def compute_coupling_terms(self, state0: Any, state: Any):
        if "u" not in state0 or "u" not in state:
            raise KeyError("black-oil mechanics states require full displacement 'u'")
        return {
            "old": self.mech.volumetric_strain_full(state0["u"]),
            "new": self.mech.volumetric_strain_full(state["u"]),
        }, state["pressure"]

    computeCouplingTerms = compute_coupling_terms
    getEquations = get_equations
    updateState = update_state


__all__ = ["MechBlackOilModel"]
