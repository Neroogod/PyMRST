"""MRST ``ad-core/models/aquifers/AquiferModel.m``: Fetkovich aquifers attached
to a reservoir model.

The model carries `processAquifer`'s connection table (`aquifers`, with its
column index `aquind`), the aquifers' properties and initial pressures and
volumes. Its state lives on the reservoir state as `aquiferSol`, one entry
per aquifer with `pressure`, `volume` and `flux`. Each connection exchanges

    q = alpha J (p_aq - p_W + rho_W g (z_conn - z_aq)) (1 - exp(-dt/Tc))/(dt/Tc),

`Tc = C V_aq/J`, with the water equation (scaled by the water surface
density in a generic model), and after each converged step the aquifer's
pressure and volume are depleted by what flowed: `p = p0 - Q/(C V0)`,
`V = V0 - Q`.

The connection table's aquifer and cell numbers are 0-based here (MRST's are
1-based).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, List, Optional, Tuple

import numpy as np

from pymrst.modules.ad_core.ad_oo.physical_model import PhysicalModel
from pymrst.utils.adi import value


class AquiferModel(PhysicalModel):
    """MRST `AquiferModel(reservoirModel, aquifers, aquind, aquiferprops, initvals)`."""

    def __init__(self, reservoirModel: Any, aquifers: Any, aquind: Any, aquiferprops: Any,
                 initvals: Any, **options: Any):
        super().__init__(None)
        self.ReservoirModel = reservoirModel
        self.aquifers = np.atleast_2d(np.asarray(aquifers, dtype=float))
        self.aquind = aquind
        self.aquiferprops = aquiferprops
        self.initvals = initvals

    def validate_state(self, state: Any) -> Any:
        if "aquiferSol" not in state:
            state = self.init_state_aquifer(state)
        return state

    def validate_model(self, reservoirModel: Any = None, *args: Any) -> "AquiferModel":
        self.ReservoirModel = reservoirModel
        return self

    def init_state_aquifer(self, state: Any) -> Any:
        p = np.atleast_1d(_init(self.initvals, "pressures"))
        vol = np.atleast_1d(_init(self.initvals, "volumes"))
        state["aquiferSol"] = [SimpleNamespace(pressure=float(p[k]), volume=float(vol[k]),
                                               flux=np.zeros(0)) for k in range(p.size)]
        return state

    def update_after_convergence(self, state0: Any, state: Any, dt: float,
                                 drivingForces: Any) -> Tuple[Any, Optional[Any]]:
        report = None
        p0 = np.array([a.pressure for a in state["aquiferSol"]], dtype=float)
        vol0 = np.array([a.volume for a in state["aquiferSol"]], dtype=float)
        q = np.asarray(value(self.compute_aquifer_fluxes(state, dt)), dtype=float)
        aquid = self.aquifers[:, self.aquind.aquid].astype(np.int64)
        aquid2conn = _aquid2conn(aquid)
        Q = aquid2conn.T @ q
        Q = dt*Q
        C = np.asarray(_field(self.aquiferprops, "C"), dtype=float)
        p = p0 - Q/(C*vol0)
        vol = vol0 - Q
        # MATLAB's struct array is a value: the states before this one keep
        # their own aquifer solution.
        state["aquiferSol"] = [SimpleNamespace(pressure=float(p[k]), volume=float(vol[k]),
                                               flux=q[aquid2conn[:, k].astype(bool)])
                               for k in range(p.size)]
        return state, report

    def compute_aquifer_fluxes(self, state: Any, dt: float) -> Any:
        rmodel = self.ReservoirModel
        p_aq = np.array([a.pressure for a in state["aquiferSol"]], dtype=float)
        V_aq = np.array([a.volume for a in state["aquiferSol"]], dtype=float)
        aq, ix = self.aquifers, self.aquind
        alpha = aq[:, ix.alpha]
        J = aq[:, ix.J]
        conn = aq[:, ix.conn].astype(np.int64)
        depthconn = aq[:, ix.depthconn]
        depthaq = aq[:, ix.depthaq]
        C = aq[:, ix.C]
        aquid = aq[:, ix.aquid].astype(np.int64)
        aquid2conn = _aquid2conn(aquid)
        p_aq = aquid2conn @ p_aq
        V_aq = aquid2conn @ V_aq
        p = rmodel.get_props(state, "PhasePressures")
        isw = [name == "W" for name in rmodel.get_phase_names()].index(True)
        pW = _rows(p[isw], conn)
        b = rmodel.get_props(state, "ShrinkageFactors")
        bW = _rows(b[isw], conn)
        rhoW = bW*np.asarray(_field(rmodel.fluid, "rhoWS"), dtype=float)
        Tc = C*V_aq/J
        if dt == 0:
            coef = 1
        else:
            coef = (1 - np.exp(-dt/Tc))/(dt/Tc)
        g = np.ravel(rmodel.gravity)[2]
        return alpha*J*(p_aq - pW + rhoW*g*(depthconn - depthaq))*coef

    def add_aquifers_contribution(self, eqs: List[Any], names: List[str], state: Any,
                                  dt: float) -> List[Any]:
        q = self.compute_aquifer_fluxes(state, dt)
        wind = [n.lower() == "water" for n in names].index(True)
        scale = 1
        if getattr(self.ReservoirModel, "isGenericReservoirModel", False):
            rhoS = self.ReservoirModel.get_surface_densities()
            scale = np.ravel(value(rhoS))[wind]
        conn = self.aquifers[:, self.aquind.conn].astype(np.int64)
        eqs = list(eqs)
        eqs[wind] = _assign_rows(eqs[wind], conn, _rows(eqs[wind], conn) - q*scale)
        return eqs

    def update_aquifer_fluxes(self, state: Any, dt: float = 0) -> Any:
        # MRST loops `for k = 1:naq` with `naq` never defined in this method,
        # so calling it is an error there; it is not called anywhere in MRST.
        self.compute_aquifer_fluxes(state, dt)
        raise NameError("Undefined function or variable 'naq'.")

    validateState = validate_state
    validateModel = validate_model
    initStateAquifer = init_state_aquifer
    updateAfterConvergence = update_after_convergence
    computeAquiferFluxes = compute_aquifer_fluxes
    addAquifersContribution = add_aquifers_contribution
    updateAquiferFluxes = update_aquifer_fluxes


def add_aquifers_contribution(model: AquiferModel, eqs: List[Any], names: List[str],
                              state: Any, dt: float) -> List[Any]:
    """MRST's call form, `eqs = addAquifersContribution(model.AquiferModel, ...)`."""
    return model.add_aquifers_contribution(eqs, names, state, dt)


addAquifersContribution = add_aquifers_contribution


def _aquid2conn(aquid: np.ndarray) -> np.ndarray:
    """MRST's `sparse(aquid, 1:nconn, 1, naq, nconn)'`: connection x aquifer."""
    out = np.zeros((aquid.size, int(aquid.max()) + 1))
    out[np.arange(aquid.size), aquid] = 1.0
    return out


def _rows(x: Any, rows: np.ndarray) -> Any:
    return x[rows]


def _assign_rows(x: Any, rows: np.ndarray, values: Any) -> Any:
    x = x.copy() if hasattr(x, "copy") else x
    x[rows] = values
    return x


def _field(obj: Any, name: str) -> Any:
    return obj[name] if isinstance(obj, dict) else getattr(obj, name)


def _init(initvals: Any, name: str) -> Any:
    return np.asarray(_field(initvals, name), dtype=float)


__all__ = ["AquiferModel", "add_aquifers_contribution", "addAquifersContribution"]


if __name__ == "__main__":
    from pymrst.modules.deckformat.params.rock.process_aquifer import AquiferIndices

    IX = AquiferIndices()
    TABLE = np.array([[0, 0, 1, 2.0, 1e-9, 1.0, 10.0, 12.0],
                      [1, 2, 1, 3.0, 1e-9, 1.0, 10.0, 10.0]])
    PROPS = SimpleNamespace(C=np.array([1e-9, 2e-9]))
    INIT = {"pressures": np.array([2e7, 3e7]), "volumes": np.array([1e6, 2e6])}

    class _Reservoir:
        isGenericReservoirModel = False
        gravity = np.array([0.0, 0.0, 10.0])
        fluid = SimpleNamespace(rhoWS=1000.0)

        def get_phase_names(self):
            return ["W", "O"]

        def get_props(self, state, name):
            if name == "PhasePressures":
                return [np.array([1e7, 1.5e7, 2e7]), None]
            return [np.ones(3), None]

    def test_state_is_one_entry_per_aquifer():
        model = AquiferModel(_Reservoir(), TABLE, IX, PROPS, INIT)
        state = model.validate_state({})
        assert [a.pressure for a in state["aquiferSol"]] == [2e7, 3e7]
        print("ok test_state_is_one_entry_per_aquifer")

    def test_flux_is_the_fetkovich_rate():
        model = AquiferModel(_Reservoir(), TABLE, IX, PROPS, INIT)
        state = model.validate_state({})
        q = model.compute_aquifer_fluxes(state, 0)
        expected = [2.0*(2e7 - 1e7 + 1000*10*(10.0 - 12.0)), 3.0*(3e7 - 2e7)]
        np.testing.assert_allclose(q, expected)
        print("ok test_flux_is_the_fetkovich_rate")

    def test_convergence_depletes_by_what_flowed():
        model = AquiferModel(_Reservoir(), TABLE, IX, PROPS, INIT)
        state = model.validate_state({})
        dt = 10.0
        q = np.asarray(model.compute_aquifer_fluxes(state, dt))
        state, _ = model.update_after_convergence(None, state, dt, None)
        np.testing.assert_allclose([a.volume for a in state["aquiferSol"]],
                                   INIT["volumes"] - dt*q)
        np.testing.assert_allclose(state["aquiferSol"][0].pressure,
                                   2e7 - dt*q[0]/(1e-9*1e6))
        print("ok test_convergence_depletes_by_what_flowed")

    def test_the_water_equation_receives_the_flux():
        model = AquiferModel(_Reservoir(), TABLE, IX, PROPS, INIT)
        state = model.validate_state({})
        eqs = model.add_aquifers_contribution([np.zeros(3), np.zeros(3)], ["water", "oil"],
                                              state, 0)
        q = model.compute_aquifer_fluxes(state, 0)
        np.testing.assert_allclose(eqs[0], [-q[0], 0.0, -q[1]])
        np.testing.assert_array_equal(eqs[1], 0.0)
        print("ok test_the_water_equation_receives_the_flux")

    test_state_is_one_entry_per_aquifer()
    test_flux_is_the_fetkovich_rate()
    test_convergence_depletes_by_what_flowed()
    test_the_water_equation_receives_the_flux()
    print("\nAll tests passed!")
