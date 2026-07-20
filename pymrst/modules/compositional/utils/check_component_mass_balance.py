"""Component mass-balance diagnostic (MRST ``checkComponentMassBalance.m``)."""

from __future__ import annotations

from typing import Any

import numpy as np

from pymrst.modules.compositional.utils.format_mass_string import formatMassString


def _get(obj: Any, name: str, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _call(obj: Any, camel: str, snake: str | None = None, default=None):
    for name in (camel, snake):
        if not name:
            continue
        value = _get(obj, name, None)
        if value is not None:
            return value() if callable(value) else value
    return default


def _state_field(state: Any, *names: str):
    for name in names:
        value = _get(state, name, None)
        if value is not None:
            return value
    raise KeyError(f"state is missing all of {names!r}")


def _molar_mass(model: Any) -> np.ndarray:
    eos = _get(model, "EOSModel", model)
    mixture = _get(eos, "CompositionalMixture", None)
    if mixture is None:
        mixture = _get(eos, "mixture", None)
    if mixture is None:
        mixture = _get(eos, "fluid", None)
    if mixture is None:
        raise ValueError("model must expose EOSModel/CompositionalMixture with molarMass")
    return np.asarray(mixture.molarMass, dtype=float)


def _component_names(model: Any, ncomp: int) -> tuple[str, ...]:
    names = _call(model, "getComponentNames", "get_component_names", None)
    if names is None:
        eos = _get(model, "EOSModel", model)
        names = _call(eos, "getComponentNames", "get_component_names", None)
    if names is None:
        mixture = _get(_get(model, "EOSModel", model), "CompositionalMixture", None)
        names = _get(mixture, "names", None)
    if names is None:
        return tuple(f"Component {i + 1}" for i in range(ncomp))
    return tuple(str(v) for v in names)


def _mass_fraction(model: Any, mole_fraction) -> np.ndarray:
    arr = np.asarray(mole_fraction, dtype=float)
    mw = _molar_mass(model)
    mass = arr * mw[None, :]
    denom = mass.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(mass, denom, out=np.zeros_like(mass), where=denom != 0.0)


def _pv(model: Any, state: Any) -> np.ndarray:
    operators = _get(model, "operators", None)
    pv = _get(operators, "pv", None)
    if pv is None and isinstance(operators, dict):
        pv = operators.get("pv")
    if pv is None:
        pv = _get(model, "pv", None)
    if pv is None:
        pv = np.ones(np.asarray(_state_field(state, "pressure")).size)
    return np.asarray(pv, dtype=float).reshape(-1)


def _phase_indices(model: Any, state: Any) -> tuple[int, int]:
    water = int(bool(_get(model, "water", False)))
    oil_ix = water
    gas_ix = water + 1
    s = np.asarray(_state_field(state, "s", "saturation", "saturations"), dtype=float)
    if s.shape[1] <= gas_ix:
        gas_ix = min(s.shape[1] - 1, oil_ix)
    return oil_ix, gas_ix


def _rho(state: Any, phase_ix: int, fallback: float = 1.0) -> np.ndarray:
    rho = _get(state, "rho", None)
    if rho is None:
        pressure = np.asarray(_state_field(state, "pressure"), dtype=float).reshape(-1)
        return np.full(pressure.size, fallback, dtype=float)
    rho = np.asarray(rho, dtype=float)
    if rho.ndim == 1:
        return rho
    return rho[:, phase_ix]


def _well_solutions(state: Any) -> list[Any]:
    ws = _get(state, "wellSol", None)
    if ws is None:
        ws = _get(state, "well_sol", None)
    if ws is None:
        return []
    if isinstance(ws, np.ndarray):
        ws = ws.tolist()
    return list(ws) if isinstance(ws, (list, tuple)) else [ws]


def _well_component_for_state(state: Any, comp_ix: int) -> np.ndarray:
    vals = []
    for well in _well_solutions(state):
        comps = _get(well, "components", None)
        if comps is None:
            vals.append(0.0)
            continue
        arr = np.asarray(comps, dtype=float)
        if arr.ndim == 1:
            vals.append(float(arr[comp_ix]))
        else:
            vals.append(float(arr[:, comp_ix].sum()))
    return np.asarray(vals, dtype=float)


def _step_values(schedule: Any, n: int) -> np.ndarray:
    if schedule is None:
        return np.ones(n, dtype=float)
    step = _get(schedule, "step", None)
    val = _get(step, "val", None)
    if val is None and isinstance(step, dict):
        val = step.get("val")
    if val is None:
        val = _get(schedule, "dt", None)
    if val is None:
        val = np.ones(n, dtype=float)
    val = np.asarray(val, dtype=float).reshape(-1)
    if val.size < n:
        val = np.pad(val, (0, n - val.size), mode="edge")
    return val[:n]


def _phase_component_mass(model: Any, state: Any, phase_fraction: np.ndarray, phase_ix: int, comp_ix: int) -> float:
    s = np.asarray(_state_field(state, "s", "saturation", "saturations"), dtype=float)
    pv = _pv(model, state)
    rho = _rho(state, phase_ix)
    return float(np.sum(phase_fraction[:, comp_ix] * s[:, phase_ix] * rho * pv))


def checkComponentMassBalance(model, state0, states, schedule=None, n: int | None = None, *, verbose: bool = True) -> dict[str, Any]:
    """Compute a component mass-balance report for a compositional run.

    The report mirrors the quantities printed by MRST: starting oil/gas mass,
    ending oil/gas mass, injected component mass, produced component mass, and
    aggregate totals. Well component streams are integrated over ``schedule``
    timesteps using ``wellSol[*].components``.
    """
    states = list(states)
    if n is None:
        n = len(states)
    states = states[:n]
    if not states:
        raise ValueError("states must contain at least one state")
    dt = _step_values(schedule, len(states))
    final = states[-1]
    x_final = _get(final, "xMass", None)
    if x_final is None:
        x_final = _mass_fraction(model, _state_field(final, "x"))
    y_final = _get(final, "yMass", None)
    if y_final is None:
        y_final = _mass_fraction(model, _state_field(final, "y"))
    x0 = _mass_fraction(model, _state_field(state0, "x"))
    y0 = _mass_fraction(model, _state_field(state0, "y"))
    oil_ix, gas_ix = _phase_indices(model, final)
    ncomp = x_final.shape[1]
    names = _component_names(model, ncomp)

    component_reports = []
    totals = {"oil_start": 0.0, "oil_end": 0.0, "gas_start": 0.0, "gas_end": 0.0, "injected": 0.0, "produced": 0.0}
    for comp_ix in range(ncomp):
        well_flux = np.vstack([_well_component_for_state(state, comp_ix) for state in states]) if states else np.zeros((0, 0))
        weighted = well_flux * dt[:, None] if well_flux.size else well_flux
        injected = float(weighted[weighted > 0.0].sum()) if weighted.size else 0.0
        produced = float(abs(weighted[weighted < 0.0].sum())) if weighted.size else 0.0
        oil_start = _phase_component_mass(model, state0, x0, oil_ix, comp_ix)
        oil_end = _phase_component_mass(model, final, x_final, oil_ix, comp_ix)
        gas_start = _phase_component_mass(model, state0, y0, gas_ix, comp_ix)
        gas_end = _phase_component_mass(model, final, y_final, gas_ix, comp_ix)
        row = {
            "name": names[comp_ix] if comp_ix < len(names) else f"Component {comp_ix + 1}",
            "oil_start": oil_start,
            "oil_end": oil_end,
            "gas_start": gas_start,
            "gas_end": gas_end,
            "injected": injected,
            "produced": produced,
            "start_total": oil_start + gas_start + injected,
            "end_total": oil_end + gas_end + produced,
        }
        component_reports.append(row)
        for key in totals:
            totals[key] += row[key]

    totals["start_total"] = totals["oil_start"] + totals["gas_start"] + totals["injected"]
    totals["end_total"] = totals["oil_end"] + totals["gas_end"] + totals["produced"]
    report = {"components": component_reports, "total": totals}
    if verbose:
        for row in component_reports:
            print(f"Component '{row['name']}':")
            print(f"* Oil mass: From {formatMassString(row['oil_start'], 1)} -> {formatMassString(row['oil_end'], 1)}")
            print(f"* Gas mass: From {formatMassString(row['gas_start'], 1)} -> {formatMassString(row['gas_end'], 1)}")
            print(f"* Injected: {formatMassString(row['injected'], 1)}")
            print(f"* Produced: {formatMassString(row['produced'], 1)}")
    return report


def check_component_mass_balance(model, state0, states, schedule=None, n: int | None = None, *, verbose: bool = True) -> dict[str, Any]:
    """Pythonic alias for :func:`checkComponentMassBalance`."""
    return checkComponentMassBalance(model, state0, states, schedule=schedule, n=n, verbose=verbose)


__all__ = ["checkComponentMassBalance", "check_component_mass_balance"]
