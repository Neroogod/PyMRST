"""Read the official SPE3 wet-gas benchmark deck.

The importer covers every keyword used by ``BENCH_SPE3.DATA``, converts FIELD
data to SI, translates SWFN/SGFN/SOF3 to PyMRST relperm tables, and preserves
the mid-schedule injector control change.
"""

from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import numpy as np

from pymrst.modules.deckformat.spe1.read_spe1_deck import (
    LB_PER_FT3, MSCF, STB, _expand, _read_numbers, _read_records, _tokenize,
)
from pymrst.utils.units import CENTI, DARCY, DAY, FT, MILLI, POISE, PSIA

TABLE_KEYWORDS = {"SWFN", "SGFN", "SOF3", "PVDO"}
VECTOR_KEYWORDS = {
    "PORO", "PERMX", "PERMY", "PERMZ", "PRESSURE", "SWAT", "SGAS",
    "RV", "COORD", "ZCORN", "ACTNUM",
}
RECORD_KEYWORDS = {
    "PVTW", "ROCK", "DENSITY", "DIMENS", "SPECGRID", "WELSPECS",
    "COMPDAT", "WCONPROD", "WCONINJE", "TSTEP",
}


def read_spe3_deck(path) -> Dict:
    """Parse ``BENCH_SPE3.DATA`` into a simulation-ready SI mapping."""
    path = Path(path)
    tokens = _tokenize(path)
    raw: Dict[str, object] = {}
    step_count = 0
    injection_events = []
    index = 0
    while index < len(tokens):
        keyword = tokens[index].upper()
        index += 1
        if keyword in VECTOR_KEYWORDS:
            raw[keyword], index = _read_numbers(tokens, index)
        elif keyword == "PVTG":
            raw[keyword], index = _read_pvtg(tokens, index)
        elif keyword == "TSTEP":
            record = []
            while tokens[index] != "/":
                record.extend(_expand(tokens[index]))
                index += 1
            index += 1
            raw.setdefault(keyword, []).append(record)
            step_count += len(record)
        elif keyword in TABLE_KEYWORDS or keyword in RECORD_KEYWORDS:
            records, index = _read_records(tokens, index)
            raw.setdefault(keyword, []).extend(records)
            if keyword == "WCONINJE":
                injection_events.extend((step_count, record) for record in records)
    return _convert(raw, injection_events)


def _read_pvtg(tokens: List[str], index: int):
    branches = []
    while tokens[index] != "/":
        numbers, index = _read_numbers(tokens, index)
        if (numbers.size - 1) % 3:
            raise ValueError("PVTG requires pressure plus Rv/Bg/mu triples")
        branches.append((numbers[0], numbers[1:].reshape(-1, 3)))
    return branches, index + 1


def _numbers(raw, keyword):
    return np.asarray(raw[keyword], dtype=float).reshape(-1)


def _convert(raw: Dict, injection_events) -> Dict:
    out: Dict = {}
    dim_record = raw.get("DIMENS", raw.get("SPECGRID"))[0]
    out["cartDims"] = tuple(int(float(v)) for v in dim_record[:3])
    _convert_grid(raw, out)
    _convert_rock(raw, out)
    _convert_fluid(raw, out)
    _convert_initial(raw, out)
    _convert_schedule(raw, injection_events, out)
    return out


def _convert_grid(raw: Dict, out: Dict) -> None:
    nx, ny, nz = out["cartDims"]
    coord = raw["COORD"].reshape(-1, 6)
    xs = np.unique(coord[:, 0])
    ys = np.unique(coord[:, 1])
    zs = np.unique(raw["ZCORN"])
    if (xs.size, ys.size, zs.size) != (nx + 1, ny + 1, nz + 1):
        raise NotImplementedError("SPE3 importer requires its regular corner-point grid")
    out.update(x=xs * FT, y=ys * FT, z=zs * FT)
    out["actnum"] = np.asarray(raw.get("ACTNUM", np.ones(nx * ny * nz)), bool)


def _convert_rock(raw: Dict, out: Dict) -> None:
    out["poro"] = raw["PORO"]
    out["perm"] = np.column_stack(
        [raw["PERMX"], raw["PERMY"], raw["PERMZ"]]
    ) * MILLI * DARCY
    rock = _numbers(raw, "ROCK")
    out["rock_cp"] = {"p_ref": rock[0] * PSIA, "c": rock[1] / PSIA}


def _convert_fluid(raw: Dict, out: Dict) -> None:
    swfn = _numbers(raw, "SWFN").reshape(-1, 3)
    sgfn = _numbers(raw, "SGFN").reshape(-1, 3)
    sof3 = _numbers(raw, "SOF3").reshape(-1, 3)
    swcon = swfn[0, 0]
    krow = np.interp(1.0 - swfn[:, 0], sof3[:, 0], sof3[:, 1])
    krog = np.interp(1.0 - swcon - sgfn[:, 0], sof3[:, 0], sof3[:, 2])
    out["swof"] = np.column_stack([swfn[:, :2], krow, swfn[:, 2] * PSIA])
    out["sgof"] = np.column_stack([sgfn[:, :2], krog, sgfn[:, 2] * PSIA])
    pvtw = _numbers(raw, "PVTW")
    out["pvtw"] = {
        "p_ref": pvtw[0] * PSIA, "b_ref": pvtw[1], "c": pvtw[2] / PSIA,
        "mu_ref": pvtw[3] * CENTI * POISE,
        "viscosibility": pvtw[4] / PSIA,
    }
    density = _numbers(raw, "DENSITY") * LB_PER_FT3
    out["rhoS"] = np.array([density[1], density[0], density[2]])
    pvdo = _numbers(raw, "PVDO").reshape(-1, 3)
    pvdo[:, 0] *= PSIA
    pvdo[:, 2] *= CENTI * POISE
    out["pvdo"] = pvdo
    out["pvtg"] = []
    for pressure, rows in raw["PVTG"]:
        rows = rows.copy()
        rows[:, 0] *= STB / MSCF
        rows[:, 1] *= STB / MSCF
        rows[:, 2] *= CENTI * POISE
        out["pvtg"].append((pressure * PSIA, rows))


def _convert_initial(raw: Dict, out: Dict) -> None:
    out["pressure"] = raw["PRESSURE"] * PSIA
    out["swat"], out["sgas"] = raw["SWAT"], raw["SGAS"]
    out["rv"] = raw["RV"] * STB / MSCF
    out["rs"] = np.zeros_like(out["rv"])


def _default(record, index, fallback=None):
    if index >= len(record) or record[index].endswith("*"):
        return fallback
    return float(record[index])


def _convert_schedule(raw: Dict, injection_events, out: Dict) -> None:
    nx, ny, _ = out["cartDims"]
    specs = {r[0].strip("'\""): r for r in raw["WELSPECS"]}
    completions = {r[0].strip("'\""): r for r in raw["COMPDAT"]}
    base = {}
    for name, rec in completions.items():
        i, j = int(float(rec[1])) - 1, int(float(rec[2])) - 1
        k1, k2 = int(float(rec[3])) - 1, int(float(rec[4])) - 1
        base[name] = {
            "name": name,
            "cells": np.array([i + nx*j + nx*ny*k for k in range(k1, k2 + 1)]),
            "ref_depth": float(specs[name][4]) * FT,
            "radius": float(rec[8]) * FT / 2.0,
            "compi": np.array([0.0, 0.0, 1.0]),
        }
    prod = raw["WCONPROD"][0]
    pname = prod[0].strip("'\"")
    base[pname].update(type="grat", val=-_default(prod, 5, 0.0) * MSCF / DAY,
                       sign=-1, lims={"bhp": _default(prod, 8, 0.0) * PSIA})
    controls, switch_steps = [], []
    for step, rec in injection_events:
        wells = deepcopy(base)
        name = rec[0].strip("'\"")
        wells[name].update(type="rate", val=_default(rec, 4, 0.0) * MSCF / DAY,
                           sign=1, lims={"bhp": _default(rec, 6, np.inf) * PSIA})
        controls.append(wells)
        switch_steps.append(step)
    out["wells"] = controls[0]
    out["controls"] = controls
    out["control_switch_steps"] = np.asarray(switch_steps, dtype=int)
    out["tsteps"] = np.array(
        [float(v) for record in raw["TSTEP"] for v in record], dtype=float
    ) * DAY
    out["step_control"] = np.searchsorted(
        out["control_switch_steps"], np.arange(out["tsteps"].size), side="right"
    ) - 1


if __name__ == "__main__":
    root = Path(__file__).parents[4]
    deck = read_spe3_deck(
        root.parent / "mrst-2024a-Geomech/examples/data/SPE3/BENCH_SPE3.DATA"
    )
    assert deck["cartDims"] == (9, 9, 4)
    assert deck["pressure"].size == 324
    assert len(deck["pvtg"]) == 8
    assert deck["tsteps"].size == 25
    assert np.array_equal(deck["control_switch_steps"], [0, 15])
    assert deck["controls"][1]["INJECTOR"]["val"] == 0
    print("SPE3 deck parsed successfully")
