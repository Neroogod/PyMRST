"""MRST ``diagnostics/utils/validateStateForDiagnostics.m``: a state reduced
to total fluxes - the face fluxes summed over phases, and each well's
connection fluxes (or, lacking those, its `cqs`) summed likewise."""

import copy
from typing import Any

import numpy as np


def validate_state_for_diagnostics(state: Any) -> Any:
    """MRST `state = validateStateForDiagnostics(state)`; returns a copy."""
    flux = _get(state, "flux")
    if flux is None:
        raise ValueError("Reservoir state must provide total Darcy flux")
    state = dict(state) if isinstance(state, dict) else copy.copy(state)
    _set(state, "flux", _row_sum(flux))
    well_sols = _get(state, "wellSol")
    if well_sols:
        well_sols = [copy.copy(ws) for ws in well_sols]
        if all(getattr(ws, "flux", None) is not None for ws in well_sols):
            for ws in well_sols:
                ws.flux = _row_sum(ws.flux)
        elif all(getattr(ws, "cqs", None) is not None for ws in well_sols):
            for ws in well_sols:
                ws.flux = _row_sum(ws.cqs)
        _set(state, "wellSol", well_sols)
    return state


def _row_sum(x: Any) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x.sum(axis=1) if x.ndim > 1 else x


def _get(state: Any, name: str) -> Any:
    return state.get(name) if isinstance(state, dict) else getattr(state, name, None)


def _set(state: Any, name: str, value: Any) -> None:
    if isinstance(state, dict):
        state[name] = value
    else:
        setattr(state, name, value)


validateStateForDiagnostics = validate_state_for_diagnostics
__all__ = ["validate_state_for_diagnostics", "validateStateForDiagnostics"]


if __name__ == "__main__":
    from types import SimpleNamespace

    def test_phase_fluxes_are_summed():
        ws = SimpleNamespace(flux=np.array([[1.0, 2.0], [3.0, 4.0]]))
        state = {"flux": np.array([[1.0, -1.0], [2.0, 0.5]]), "wellSol": [ws]}
        out = validate_state_for_diagnostics(state)
        assert np.array_equal(out["flux"], [0.0, 2.5])
        assert np.array_equal(out["wellSol"][0].flux, [3.0, 7.0])
        assert state["flux"].ndim == 2 and ws.flux.ndim == 2
        print("ok test_phase_fluxes_are_summed")

    def test_cqs_when_there_is_no_flux():
        ws = SimpleNamespace(flux=None, cqs=np.array([[1.0, 1.0]]))
        out = validate_state_for_diagnostics({"flux": np.zeros(2), "wellSol": [ws]})
        assert np.array_equal(out["wellSol"][0].flux, [2.0])
        print("ok test_cqs_when_there_is_no_flux")

    def test_a_state_without_flux_is_an_error():
        try:
            validate_state_for_diagnostics({"pressure": np.zeros(2)})
        except ValueError:
            print("ok test_a_state_without_flux_is_an_error")
            return
        raise AssertionError("expected ValueError")

    test_phase_fluxes_are_summed()
    test_cqs_when_there_is_no_flux()
    test_a_state_without_flux_is_an_error()
    print("\nAll tests passed!")
