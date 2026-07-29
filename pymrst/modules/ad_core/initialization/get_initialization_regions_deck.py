"""MRST ``getInitializationRegionsDeck`` for parsed deck dictionaries."""

from __future__ import annotations

import numpy as np

from pymrst.modules.ad_core.initialization.regions import (
    get_initialization_regions_base,
)


def _section(mapping, name):
    for key in (name, name.upper(), name.lower()):
        if isinstance(mapping, dict) and key in mapping:
            return mapping[key]
    return {}


def _flag(model, name):
    return bool(getattr(model, name, False))


def _phase_densities(model, pvt_region):
    rho = np.asarray(
        getattr(model.fluid, "rho_s", getattr(model.fluid, "rhoS", [])),
        dtype=float,
    ).ravel()
    if rho.size == 0:
        raise ValueError("model fluid must expose surface densities")
    return [float(value) for value in rho[: model.n_phases]]


def get_initialization_regions_deck(model, deck):
    """Build one equilibrium subregion per EQLNUM/SATNUM/PVTNUM combination."""
    solution = _section(deck, "SOLUTION")
    regions_data = _section(deck, "REGIONS")
    equil = np.atleast_2d(np.asarray(_section(solution, "EQUIL"), dtype=float))
    if equil.size == 0:
        raise ValueError("deck SOLUTION.EQUIL is required")
    nc = model.G.cells.num
    index_map = np.asarray(
        getattr(model.G.cells, "indexMap", np.arange(nc)), dtype=np.int64
    ).ravel()

    def region_array(name, default):
        raw = _section(regions_data, name)
        if isinstance(raw, dict) and not raw:
            return np.full(nc, default, dtype=np.int64)
        values = np.asarray(raw, dtype=np.int64).ravel()
        return values[index_map]

    eqlnum = region_array("EQLNUM", 1)
    satnum = region_array("SATNUM", 1)
    pvtnum = region_array("PVTNUM", 1)
    output = []
    for eql_index, record in enumerate(equil, 1):
        cells = np.flatnonzero(eqlnum == eql_index)
        if not cells.size:
            continue
        pairs = np.unique(np.column_stack([satnum[cells], pvtnum[cells]]), axis=0)
        for sat_region, pvt_region in pairs:
            subcells = cells[
                (satnum[cells] == sat_region) & (pvtnum[cells] == pvt_region)
            ]
            owc = float(record[2])
            goc = float(record[4]) if record.size > 4 else owc
            pc_ow = float(record[3]) if record.size > 3 else 0.0
            pc_go = float(record[5]) if record.size > 5 else 0.0
            contacts = [goc, owc] if model.n_phases > 2 else [owc]
            contacts_pc = [pc_go, pc_ow] if model.n_phases > 2 else [pc_ow]
            region = get_initialization_regions_base(
                model,
                _phase_densities(model, int(pvt_region)),
                contacts,
                cells=subcells,
                datum_depth=float(record[0]),
                datum_pressure=float(record[1]),
                contacts_pc=contacts_pc,
            )
            region.satnum = int(sat_region)
            region.pvtnum = int(pvt_region)
            region.eqlnum = eql_index
            output.append(region)
    return output


getInitializationRegionsDeck = get_initialization_regions_deck

__all__ = ["get_initialization_regions_deck", "getInitializationRegionsDeck"]
