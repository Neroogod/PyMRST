"""Carter-Tracy convolution aquifer support.

MRST 2024a in this repository only instantiates Fetkovich aquifers from decks.
This module supplies the missing Carter-Tracy style convolution model in the
same interface shape as :mod:`pymrst.modules.ad_core.aquifer`.
"""

from __future__ import annotations

import numpy as np

from pymrst.modules.ad_core.aquifer import AquiferConnections
from pymrst.modules.ad_core.statefunctions.state_function import State
from pymrst.utils.adi import value


def van_everdingen_hurst_pd(td):
    """Approximate infinite-acting radial aquifer dimensionless pressure."""
    td = np.asarray(td, dtype=float)
    out = np.empty_like(td, dtype=float)
    early = td <= 1.0
    out[early] = 0.5 * np.sqrt(np.pi * np.maximum(td[early], 0.0))
    out[~early] = 0.5 * (np.log(td[~early]) + 0.80907)
    return out


class CarterTracyAquiferModel:
    """Carter-Tracy water influx model with convolution history."""

    water_index = 0
    state_arg_keys = ("aquifer_ct_t", "aquifer_ct_W")

    def __init__(
        self,
        connections: AquiferConnections,
        pressures,
        productivity_index,
        time_constant,
        history_time=None,
        cumulative_influx=None,
    ):
        self.connections = connections
        self.pressures = np.asarray(pressures, dtype=float).ravel()
        self.productivity_index = np.asarray(productivity_index, dtype=float).ravel()
        self.time_constant = np.asarray(time_constant, dtype=float).ravel()
        self.history_time = (
            np.zeros_like(self.pressures)
            if history_time is None
            else np.asarray(history_time, dtype=float).ravel()
        )
        self.cumulative_influx = (
            np.zeros_like(self.pressures)
            if cumulative_influx is None
            else np.asarray(cumulative_influx, dtype=float).ravel()
        )

    def init_state(self, state: State) -> State:
        state["aquifer_ct_t"] = self.history_time.copy()
        state["aquifer_ct_W"] = self.cumulative_influx.copy()
        return state

    def _water_props(self, model, state: State):
        cells = self.connections.cells
        p_phase = model.get_props(state, "PhasePressures")
        b = model.get_props(state, "ShrinkageFactors")
        return p_phase[self.water_index][cells], b[self.water_index][cells]

    def fluxes(self, model, state: State, history_time, cumulative_influx, dt):
        p_w, b_w = self._water_props(model, state)
        rho_ws = float(model.fluid.rho_s[self.water_index])
        rho_w = b_w * rho_ws
        head = rho_w * model.gravity_magnitude * (
            self.connections.depth_conn - self.connections.depth_aq
        )
        p_aq = self.connections.map_to_conn(self.pressures)
        drive = p_aq - p_w + head
        td0 = self.connections.map_to_conn(history_time / self.time_constant)
        td1 = self.connections.map_to_conn((history_time + dt) / self.time_constant)
        pd0 = van_everdingen_hurst_pd(td0)
        pd1 = van_everdingen_hurst_pd(td1)
        d_pd = np.maximum(pd1 - pd0, 1e-30)
        j = self.connections.j
        q_inst = self.connections.alpha * j * drive
        # Carter-Tracy convolution correction: cumulative historical influx
        # reduces the current drive over the dimensionless-pressure increment.
        hist = self.connections.map_to_conn(cumulative_influx)
        corr = hist / d_pd / np.maximum(dt, 1e-30)
        return q_inst - self.connections.alpha * corr

    def add_contribution(self, model, eqs, ad_state, history_time, cumulative_influx, dt):
        cells = self.connections.cells
        q = self.fluxes(model, ad_state, history_time, cumulative_influx, dt)
        q = model._basis_scale()[self.water_index] * q
        eqs[self.water_index][cells] = eqs[self.water_index][cells] - q
        return eqs

    def after_convergence(self, model, state: State, history_time, cumulative_influx, dt):
        q = value(self.fluxes(model, state, history_time, cumulative_influx, dt))
        total = dt * self.connections.sum_to_aquifer(q)
        return history_time + dt, cumulative_influx + total


__all__ = ["CarterTracyAquiferModel", "van_everdingen_hurst_pd"]


if __name__ == "__main__":
    def test_dimensionless_pressure_monotone():
        td = np.array([0.0, 0.1, 1.0, 10.0])
        pd = van_everdingen_hurst_pd(td)
        assert np.all(np.diff(pd) >= 0.0)
        print("ok test_dimensionless_pressure_monotone")

    test_dimensionless_pressure_monotone()
