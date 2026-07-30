"""Convert incompressible well solutions to ad-core summary rows."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np


def _get(obj, name):
    return obj[name] if isinstance(obj, dict) else getattr(obj, name)


def convert_incomp_well_sols(wells, states, incomp_fluid):
    """Port MRST ``convertIncompWellSols`` for two-phase W/O fluids."""
    wells = list(wells)
    output = []
    for state in states:
        mu = np.asarray(incomp_fluid.properties(state), dtype=float)
        saturation = incomp_fluid.saturation(state)
        kr = np.asarray(incomp_fluid.relperm(saturation, state), dtype=float)
        mobility = kr / mu
        total = mobility.sum(axis=1)
        rows = []
        old_rows = list(_get(state, "wellSol"))
        for well, old in zip(wells, old_rows):
            flux = np.asarray(_get(old, "flux"), dtype=float).ravel()
            influx, outflux = np.minimum(flux, 0.0), np.maximum(flux, 0.0)
            cells = np.asarray(well.cells, dtype=np.int64).ravel()
            fractional = mobility[cells] / total[cells, None]
            rates = [
                float(np.sum(influx * fractional[:, phase])
                      + np.sum(outflux * float(well.compi[phase])))
                for phase in range(2)
            ]
            rows.append(SimpleNamespace(
                name=well.name, qWs=rates[0], qOs=rates[1],
                bhp=float(_get(old, "pressure")),
            ))
        output.append(rows)
    return output


convertIncompWellSols = convert_incomp_well_sols

__all__ = ["convert_incomp_well_sols", "convertIncompWellSols"]
