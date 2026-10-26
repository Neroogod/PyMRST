"""MRST ``ad-core/initialization/initStateDeck.m``.

The initial state of a model from a unit-converted deck, by one of two
schemes:
- **EQUIL**: hydrostatic equilibration over the deck's initialisation
  regions (`getInitializationRegionsDeck`, `initStateBlackOilAD`), with the
  water saturation optionally taken from SWATINIT and the oil-water
  capillary curve's scaling stored as `pcowScale`;
- **direct assignment**: PRESSURE (or a PRVD table), SWAT/SOIL/SGAS with the
  missing phase filled in, and RS/RV (or PBUB/PDEW through the PVT tables).

Any aquifer then initialises its own state. The state is MRST's struct:
`pressure`, the saturation matrix `s` and `rs`/`rv`. `init_state_deck_ad`
in `deck.py` is PyMRST's earlier version for the legacy models.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from pymrst.utils.matlab_interp1 import matlab_interp1

from pymrst.solvers.init_res_sol import init_res_sol


def init_state_deck(model: Any, deck: Any) -> dict:
    """MRST `initStateDeck(model, deck)`."""
    assert isinstance(deck, dict) and "RUNSPEC" in deck and "SOLUTION" in deck, (
        "Parameter 'deck' does not appear to be a valid ECLIPSE deck")
    if "SI" not in deck["RUNSPEC"]:
        raise ValueError("Non-SI units detected. Call 'convertDeckUnits(deck)' first.")
    solution = deck["SOLUTION"]
    if "EQUIL" in solution:
        state = _equilibrated(model, deck)
    elif any(k in solution for k in ("PRESSURE", "PRVD")) and \
            any(k in solution for k in ("SGAS", "SOIL", "SWAT")):
        state = _direct_assignment(model, deck)
    else:
        raise ValueError("Initialisation scheme specified in 'deck' is not supported.")
    if model.AquiferModel is not None:
        state = model.AquiferModel.init_state_aquifer(state)
    return state


def _equilibrated(model: Any, deck: Any) -> dict:
    from pymrst.modules.ad_core.ad_oo.physical_model import State
    from pymrst.modules.ad_core.initialization.get_initialization_regions_deck import (
        get_initialization_regions_deck,
    )
    from pymrst.modules.ad_core.initialization.init_state_black_oil_ad import (
        init_state_black_oil_ad,
    )
    regions = get_initialization_regions_deck(model, deck)
    state, p = init_state_black_oil_ad(model, regions)
    if "SWATINIT" in deck.get("PROPS", {}):
        from pymrst.modules.ad_core.ad_oo.wrapper_model import value_copy_model
        model = value_copy_model(model).validate_model()
        imap = np.asarray(model.G.cells.indexMap, dtype=np.int64).ravel()
        swat = np.asarray(deck["PROPS"]["SWATINIT"], dtype=float).ravel()[imap]
        state = State(state)
        state = model.set_prop(state, "sw", swat)
        so = 1 - swat
        if model.gas:
            so = so - np.asarray(model.get_prop(state, "sg"), dtype=float)
        if np.any(so < 0):
            warnings.warn(f"SWATINIT resulted in {int(np.sum(so < 0))} negative oil saturations. "
                          f"Setting to zero...")
        so = np.maximum(so, 0)
        state = model.set_prop(state, "so", so)
        pc = model.get_prop(state, "capillarypressure")
        wix, oix = model.get_phase_index("W"), model.get_phase_index("O")
        pcow = None if pc[wix] is None else -np.asarray(pc[wix], dtype=float)
        if pcow is None or pcow.size == 0:
            warnings.warn("Oil-Water capillary pressure must be present for scaling of "
                          "pc-curve with SWATINIT.")
        else:
            p[:, oix] = state["pressure"]       # May be wrong where oil is immobile
            dp = p[:, oix] - p[:, wix]
            bad = dp <= 0
            with np.errstate(divide="ignore", invalid="ignore"):
                scale = dp/pcow
            scale[bad] = 1
            state["pcowScale"] = scale
        state = dict(state)
    return state


def _direct_assignment(model: Any, deck: Any) -> dict:
    G = model.G
    solution = deck["SOLUTION"]
    phases = model.get_phase_names()
    nph = len(phases)
    state = {k: v for k, v in vars(init_res_sol(G, 0.0)).items() if not k.startswith("_")}
    state["s"] = np.zeros((G.cells.num, nph))
    imap = np.asarray(G.cells.indexMap, dtype=np.int64).ravel()
    pvtreg = _get_pvtnum(model, deck)
    z = np.asarray(G.cells.centroids)[:, 2]
    if "PRVD" in solution:
        pressure = np.asarray(state["pressure"], dtype=float).copy()
        for reg, table in enumerate(solution["PRVD"], start=1):
            table = np.asarray(table, dtype=float)
            act = pvtreg == reg
            # `griddedInterpolant(..., 'linear', 'nearest')`
            pressure[act] = matlab_interp1(table[:, 0], table[:, 1], z[act], "nearest")
        state["pressure"] = pressure
    else:
        state["pressure"] = np.asarray(solution["PRESSURE"], dtype=float).ravel()[imap]
    missing = None
    for i, letter in enumerate(phases):
        fld = {"W": "SWAT", "O": "SOIL", "G": "SGAS"}[letter.upper()]
        if fld in solution:
            state["s"][:, i] = np.asarray(solution[fld], dtype=float).ravel()[imap]
        elif missing is None:
            missing = i
        else:
            raise ValueError("Too few phase saturations specified")
    if missing is not None:
        state["s"][:, missing] = 1 - state["s"].sum(axis=1)
    if getattr(model, "disgas", False):
        if "RS" in solution:
            rs = np.asarray(solution["RS"], dtype=float).ravel()[imap]
        elif "PBUB" in solution:
            pbub = np.asarray(solution["PBUB"], dtype=float).ravel()[imap]
            rs = np.full(G.cells.num, np.nan)
            for reg, pvto in enumerate(deck["PROPS"]["PVTO"], start=1):
                cells = pvtreg == reg
                rs[cells] = _interp_key(pvto, np.minimum(pbub[cells], state["pressure"][cells]))
            assert np.all(np.isfinite(rs)), "Some cells not covered by Solution GOR initialisation."
        else:
            rs = np.zeros(G.cells.num)
        state["rs"] = rs
    if getattr(model, "vapoil", False):
        if "RV" in solution:
            rv = np.asarray(solution["RV"], dtype=float).ravel()[imap]
        elif "PDEW" in solution:
            pdew = np.asarray(solution["PDEW"], dtype=float).ravel()[imap]
            rv = np.full(G.cells.num, np.nan)
            p_gas = np.asarray(state["pressure"], dtype=float)
            if getattr(model.fluid, "pcOG", None) is not None:
                p_gas = p_gas + np.asarray(model.fluid.pcOG(model.get_prop(state, "sG")))
            for reg, pvtg in enumerate(deck["PROPS"]["PVTG"], start=1):
                cells = pvtreg == reg
                rv[cells] = _interp_key(pvtg, np.minimum(pdew[cells], p_gas[cells]))
            assert np.all(np.isfinite(rv)), "Some cells not covered by Solution OGR initialisation."
        else:
            rv = np.zeros(G.cells.num)
        state["rv"] = rv
    if hasattr(model, "polymer"):
        state["cp"] = np.zeros(G.cells.num)
    if hasattr(model, "surfactant"):
        state["cs"] = np.zeros(G.cells.num)
    return state


def _get_pvtnum(model: Any, deck: Any) -> np.ndarray:
    regions = deck.get("REGIONS", {})
    if "PVTNUM" in regions:
        imap = np.asarray(model.G.cells.indexMap, dtype=np.int64).ravel()
        return np.asarray(regions["PVTNUM"], dtype=np.int64).ravel()[imap]
    return np.ones(model.G.cells.num, dtype=np.int64)


def _interp_key(table: Any, p: np.ndarray) -> np.ndarray:
    """MRST `interp1(PVT.data(PVT.pos(1:end-1), 1), PVT.key, p, 'linear',
    'extrap')`: the key (Rs or Rv) at which pressure `p` is saturated."""
    x = np.asarray(table.data, dtype=float)[np.asarray(table.pos)[:-1], 0]
    y = np.asarray(table.key, dtype=float)
    return matlab_interp1(x, y, p, "linear")


initStateDeck = init_state_deck

__all__ = ["init_state_deck", "initStateDeck"]


if __name__ == "__main__":
    from pymrst.modules.ad_core.utils.select_model_from_deck import select_model_from_deck
    from pymrst.modules.test_suite.dataset_functions.spe1 import setup_spe1
    from pymrst.params.gravity.gravity import isolated_gravity

    def test_spe1_is_assigned_directly_from_its_solution_arrays():
        """SPE1 gives PRESSURE, SWAT, SGAS and RS; oil fills the rest."""
        with isolated_gravity():
            G, rock, fluid, deck, _ = setup_spe1()
            model = select_model_from_deck(G, rock, fluid, deck)
            state = init_state_deck(model, deck)
        sol = deck["SOLUTION"]
        np.testing.assert_allclose(state["pressure"], sol["PRESSURE"])
        np.testing.assert_allclose(state["s"][:, 0], sol["SWAT"])
        np.testing.assert_allclose(state["s"].sum(axis=1), 1.0)
        np.testing.assert_allclose(state["rs"], sol["RS"])
        assert "rv" not in state, "SPE1 has no vaporised oil"
        print("ok test_spe1_is_assigned_directly_from_its_solution_arrays")

    def test_bubble_point_pressure_gives_the_saturated_ratio():
        """PBUB above the cell pressure is capped at the cell pressure."""
        from types import SimpleNamespace
        table = SimpleNamespace(key=np.array([10.0, 20.0, 30.0]),
                                pos=np.array([0, 1, 2, 3]),
                                data=np.array([[1e6, 1.0], [2e6, 1.1], [3e6, 1.2]]))
        np.testing.assert_allclose(_interp_key(table, np.array([1.5e6, 4e6])),
                                   [15.0, 40.0])
        print("ok test_bubble_point_pressure_gives_the_saturated_ratio")

    def test_a_deck_in_field_units_is_refused():
        try:
            init_state_deck(None, {"RUNSPEC": {}, "SOLUTION": {}})
        except ValueError as refused:
            assert "convertDeckUnits" in str(refused)
        else:
            raise AssertionError("expected ValueError")
        print("ok test_a_deck_in_field_units_is_refused")

    test_spe1_is_assigned_directly_from_its_solution_arrays()
    test_bubble_point_pressure_gives_the_saturated_ratio()
    test_a_deck_in_field_units_is_refused()
    print("\nAll tests passed!")
