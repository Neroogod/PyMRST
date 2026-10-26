"""MRST ``ad-core/simulators/sim_runner/getPackedSimulatorOutput.m``.

Read a packed problem's stored output: `(wellSols, states, reports, model)`,
where `model` is validated against the first control's driving forces. It
reports whether the output is complete, partial or missing. If well solutions
were never written separately, they are taken from the states. With a
`read*FromDisk` option off, that output is returned as its handler instead.
`get_packed_simulator_output_ad` (in `packed_output.py`) is PyMRST's earlier
version, kept for its callers.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional

import numpy as np


def get_packed_simulator_output(problem: Any, *, readFromDisk: bool = True,
                                readWellSolsFromDisk: Optional[bool] = None,
                                readStatesFromDisk: Optional[bool] = None,
                                readReportsFromDisk: Optional[bool] = None):
    """MRST `getPackedSimulatorOutput`."""
    setup = problem.SimulatorSetup
    forces = setup.model.get_driving_forces(setup.schedule.controls[0])
    model = setup.model.validate_model(forces)
    wells_from_disk = readFromDisk if readWellSolsFromDisk is None else readWellSolsFromDisk
    reports_from_disk = readFromDisk if readReportsFromDisk is None else readReportsFromDisk
    states_from_disk = readFromDisk if readStatesFromDisk is None else readStatesFromDisk
    dt = np.asarray(setup.schedule.step_val, dtype=float)
    nstep = dt.size
    sh, wh, rh = (problem.OutputHandlers.states, problem.OutputHandlers.wellSols,
                  problem.OutputHandlers.reports)
    ndata = sh.numel_data()
    ws: list = [None]*wh.numel_data()
    states: list = [None]*ndata
    reports: list = [None]*rh.numel_data()
    want_wells = any(getattr(c, "W", None) for c in setup.schedule.controls)
    sn = f"{problem.BaseName} ({problem.Name})"
    if nstep == ndata:
        print(f"Found complete data for {sn}: {ndata} steps present")
    elif ndata > nstep:
        endstate = sh[ndata]
        if "time" in endstate and endstate["time"] > dt.sum():
            warnings.warn(f"Found too much data for {sn}: {ndata} of {nstep} steps "
                          "present. Case may have been redefined!")
        else:
            print(f"Found complete data for {sn}: {ndata} steps present")
    elif ndata > 0:
        print(f"Found partial data for {sn}: {ndata} of {nstep} steps present")
    else:
        print(f"Did not find data for {sn}")
    well_output_missing = want_wells and wh.numel_data() == 0
    if well_output_missing:
        ws = [None]*ndata
    for i in range(ndata):
        if states_from_disk:
            states[i] = sh[i + 1]
        if want_wells and wells_from_disk and well_output_missing:
            ws[i] = (sh[i + 1] if states[i] is None else states[i])["wellSol"]
    for i in range(wh.numel_data()):
        if want_wells and wells_from_disk and not well_output_missing:
            try:
                ws[i] = wh[i + 1]
            except (KeyError, FileNotFoundError):
                ws[i] = states[i]["wellSol"]
    for i in range(rh.numel_data()):
        if reports_from_disk:
            try:
                reports[i] = rh[i + 1]
            except (KeyError, FileNotFoundError):
                reports[i] = None
    if not states_from_disk:
        states = sh
    if not reports_from_disk:
        reports = rh
    if not wells_from_disk:
        ws = wh
    return ws, states, reports, model


getPackedSimulatorOutput = get_packed_simulator_output

__all__ = ["getPackedSimulatorOutput", "get_packed_simulator_output"]
