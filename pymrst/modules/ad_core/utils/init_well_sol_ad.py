"""Initialize AD well-solution rows from schedule wells.

Port of MRST ``initWellSolAD.m`` for black-oil and compositional models.
"""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any, Optional, Sequence

import numpy as np

from pymrst.utils.units import BARSA


def _get(container, name, default=None):
    if isinstance(container, dict):
        return container.get(name, default)
    return getattr(container, name, default)


def _set(container, name, value):
    if isinstance(container, dict):
        container[name] = value
    else:
        setattr(container, name, value)


def _phase_names(model) -> tuple[str, ...]:
    if hasattr(model, "getPhaseNames"):
        return tuple(str(x).upper()[0] for x in model.getPhaseNames())
    phases = getattr(getattr(model, "fluid", None), "phases", ())
    aliases = {"WATER": "W", "OIL": "O", "GAS": "G", "SOLVENT": "S"}
    return tuple(aliases.get(str(x).upper(), str(x).upper()[0]) for x in phases)


def _default_rows(wells, model, state):
    pressure = np.asarray(_get(state, "pressure"), dtype=float).ravel()
    phases = _phase_names(model)
    tiny = np.finfo(float).eps
    rows = []
    eos = getattr(model, "EOSModel", None)
    ncomp = int(eos.getNumberOfComponents()) if eos is not None and hasattr(eos, "getNumberOfComponents") else 0
    for well in wells:
        cells = np.asarray(well.cells, dtype=np.int64).ravel()
        if cells.size == 0:
            raise ValueError(f"well {well.name!r} has no connections")
        sign = int(well.sign)
        rates = {f"q{phase}s": sign * tiny for phase in phases}
        row = SimpleNamespace(
            name=well.name, status=bool(well.status), type=well.type,
            val=float(well.val), sign=sign,
            bhp=float(pressure[cells[0]] + 5.0 * sign * BARSA),
            mixs=np.asarray(well.compi, dtype=float).copy(),
            qs=np.full(len(phases), sign * tiny),
            cstatus=(
                np.ones(cells.size, dtype=bool) if well.cstatus is None
                else np.asarray(well.cstatus, dtype=bool).copy()
            ),
            cdp=np.zeros(cells.size),
            cqs=np.zeros((cells.size, len(phases))),
            flux=np.zeros(cells.size),
            pressure=float(pressure[cells[0]] + 5.0 * sign * BARSA),
            **rates,
        )
        if ncomp:
            row.components = np.zeros(ncomp)
        rows.append(row)
    return rows


def _assign_schedule(wells, model, rows):
    phases = _phase_names(model)
    for well, row in zip(wells, rows):
        for name, value in (
            ("status", bool(well.status)), ("type", well.type),
            ("val", float(well.val)), ("sign", int(well.sign)),
            ("cstatus", np.ones(len(well.cells), dtype=bool)
             if well.cstatus is None else np.asarray(well.cstatus, bool).copy()),
        ):
            _set(row, name, value)
        control = str(well.type).lower()
        value = float(well.val)
        if control == "bhp":
            _set(row, "bhp", value)
            _set(row, "pressure", value)
        elif control == "rate":
            compi = np.asarray(well.compi, dtype=float).ravel()
            for index, phase in enumerate(phases):
                if index < compi.size:
                    _set(row, f"q{phase}s", value * float(compi[index]))
        elif control in {"orat", "wrat", "grat"}:
            _set(row, f"q{control[0].upper()}s", value)
    return rows


def init_well_sol_ad(
    wells: Sequence,
    model: Any,
    state0: Any,
    well_sol_init: Optional[Sequence[Any]] = None,
):
    """Create or update one MRST-style well-solution row per schedule well."""
    wells = list(wells or [])
    existing = well_sol_init
    if existing is None:
        existing = _get(state0, "wellSol", None)
    if existing is None or len(list(existing)) == 0:
        rows = _default_rows(wells, model, state0)
    else:
        rows = deepcopy(list(existing))
        if len(rows) != len(wells):
            raise ValueError("well solution count must match schedule wells")
    return _assign_schedule(wells, model, rows)


initWellSolAD = init_well_sol_ad

__all__ = ["init_well_sol_ad", "initWellSolAD"]
