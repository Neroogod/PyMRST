"""MRST ``getWellPolymer.m`` well/perforation polymer mapping."""

import numpy as np


def get_well_polymer(wells):
    """Return injector concentrations, per-connection values, and water injectors.

    Returned perforation indices are zero-based. A missing or empty ``cp``
    field denotes zero injected polymer, matching MRST.
    """
    wells = list(wells or [])
    if not wells:
        return np.zeros(0), np.zeros(0), np.zeros(0, dtype=np.int64)
    injectors = [well for well in wells if int(getattr(well, "sign", 0)) == 1]
    concentrations = np.zeros(len(injectors))
    for index, well in enumerate(injectors):
        cp = getattr(well, "cp", None)
        if cp is not None and np.asarray(cp).size:
            concentrations[index] = float(np.asarray(cp).ravel()[0])
    connection_concentrations = np.repeat(
        concentrations, [len(np.asarray(w.cells).ravel()) for w in injectors]
    )

    water_injection = []
    offset = 0
    for well in wells:
        count = len(np.asarray(well.cells).ravel())
        compi = np.asarray(well.compi, dtype=float).ravel()
        if int(getattr(well, "sign", 0)) == 1 and compi.size and compi[0] == 1:
            water_injection.extend(range(offset, offset+count))
        offset += count
    return concentrations, connection_concentrations, np.asarray(
        water_injection, dtype=np.int64
    )


getWellPolymer = get_well_polymer

__all__ = ["get_well_polymer", "getWellPolymer"]
