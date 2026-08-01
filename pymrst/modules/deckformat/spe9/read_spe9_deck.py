"""Importer for the official SPE9 black-oil benchmark deck."""

from copy import deepcopy
from pathlib import Path
from typing import Dict, List
import re

import numpy as np

from pymrst.modules.deckformat.spe1.read_spe1_deck import (
    LB_PER_FT3, MSCF, STB, _expand, _read_numbers, _read_pvto,
    _read_records,
)
from pymrst.utils.units import CENTI, DARCY, DAY, FT, MILLI, POISE, PSIA

VECTORS = {"COORD", "ZCORN", "ACTNUM", "PORO", "PERMX", "PRESSURE",
           "SWAT", "SGAS", "RS"}
TABLES = {"SWOF", "SGOF", "PVDG"}
RECORDS = {"DIMENS", "SPECGRID", "PVTW", "ROCK", "DENSITY", "WELSPECS",
           "COMPDAT", "WCONPROD", "WCONINJE", "VFPINJ", "VFPPROD"}


def read_spe9_deck(path) -> Dict:
    """Read SPE9 grid, properties, initial state, wells, and control periods."""
    tokens = _tokenize_spe9(Path(path))
    raw: Dict[str, object] = {}
    events, step_count, index = [], 0, 0
    while index < len(tokens):
        key = tokens[index].upper()
        index += 1
        if key in VECTORS:
            raw[key], index = _read_numbers(tokens, index)
        elif key == "PVTO":
            raw[key], index = _read_pvto(tokens, index)
        elif key == "TSTEP":
            values = []
            while tokens[index] != "/":
                values.extend(_expand(tokens[index])); index += 1
            index += 1
            raw.setdefault(key, []).append(values)
            step_count += len(values)
        elif key in TABLES or key in RECORDS:
            records, index = _read_records(tokens, index)
            records = [[_clean_token(v) for v in row] for row in records]
            raw.setdefault(key, []).extend(records)
            if key in {"WCONPROD", "WCONINJE", "VFPINJ", "VFPPROD"}:
                events.append((step_count, key, records))
    return _convert(raw, events)


def _clean_token(value):
    text = str(value)
    return text[1:-1].strip() if len(text) >= 2 and text[0] == text[-1] == "'" else text


def _tokenize_spe9(path: Path):
    """Tokenize while preserving quoted well names that contain padding."""
    tokens = []
    for line in path.read_text().splitlines():
        line = line.split("--", 1)[0]
        tokens.extend(re.findall(r"'(?:[^']*)'|/|[^\s/]+", line))
    out, index = [], 0
    while index < len(tokens):
        if tokens[index].upper() == "INCLUDE":
            filename = tokens[index + 1].strip("'\"")
            out.extend(_tokenize_spe9(path.parent/filename))
            index += 2
            if index < len(tokens) and tokens[index] == "/":
                index += 1
        else:
            out.append(tokens[index]); index += 1
    return out


def _flat(raw, key):
    return np.asarray(raw[key], dtype=float).reshape(-1)


def _convert(raw, events):
    out = {}
    dims = raw.get("DIMENS", raw.get("SPECGRID"))[0]
    out["cartDims"] = tuple(int(float(v)) for v in dims[:3])
    out["grdecl"] = {
        "cartDims": out["cartDims"], "COORD": raw["COORD"] * FT,
        "ZCORN": raw["ZCORN"] * FT,
        "ACTNUM": np.asarray(raw.get("ACTNUM", np.ones(np.prod(out["cartDims"]))), int),
    }
    out["poro"] = raw["PORO"]
    kx = raw["PERMX"] * MILLI * DARCY
    out["perm"] = np.column_stack([kx, kx, 0.01*kx])
    rock = _flat(raw, "ROCK")
    out["rock_cp"] = {"p_ref": rock[0]*PSIA, "c": rock[1]/PSIA}
    _fluid(raw, out)
    out["pressure"] = raw["PRESSURE"]*PSIA
    out["swat"], out["sgas"] = raw["SWAT"], raw["SGAS"]
    out["rs"] = raw["RS"]*MSCF/STB
    out["rv"] = np.zeros_like(out["rs"])
    _schedule(raw, events, out)
    return out


def _fluid(raw, out):
    swof = _flat(raw, "SWOF").reshape(-1, 4)
    sgof = _flat(raw, "SGOF").reshape(-1, 4)
    swof[:, 3] *= PSIA; sgof[:, 3] *= PSIA
    out["swof"], out["sgof"] = swof, sgof
    pvtw = _flat(raw, "PVTW")
    out["pvtw"] = {"p_ref": pvtw[0]*PSIA, "b_ref": pvtw[1],
                    "c": pvtw[2]/PSIA, "mu_ref": pvtw[3]*CENTI*POISE,
                    "viscosibility": pvtw[4]/PSIA}
    rho = _flat(raw, "DENSITY")*LB_PER_FT3
    out["rhoS"] = np.array([rho[1], rho[0], rho[2]])
    pvdg = _flat(raw, "PVDG").reshape(-1, 3)
    out["pvdg"] = np.column_stack(
        [pvdg[:, 0]*PSIA, pvdg[:, 1]*STB/MSCF, pvdg[:, 2]*CENTI*POISE]
    )
    out["pvto"] = []
    for rs, rows in raw["PVTO"]:
        rows = rows.copy(); rows[:, 0] *= PSIA; rows[:, 2] *= CENTI*POISE
        out["pvto"].append((rs*MSCF/STB, rows))


def _convert_geometry_records(raw):
    specs = deepcopy(raw["WELSPECS"])
    compdat = deepcopy(raw["COMPDAT"])
    for record in specs:
        record[4] = str(float(record[4])*FT)
    wi_unit = (STB/DAY)*(CENTI*POISE)/PSIA
    for record in compdat:
        # Connection factor is column 8 in ECLIPSE's COMPDAT record.
        if len(record) > 7 and not record[7].endswith("*"):
            record[7] = str(float(record[7])*wi_unit)
    return specs, compdat


def _expand_producer_wildcard(records, names):
    out = []
    for record in records:
        name = record[0].strip("'\"")
        if name.endswith("*"):
            prefix = name[:-1]
            for candidate in names:
                if candidate.startswith(prefix):
                    row = list(record); row[0] = candidate; out.append(row)
        else:
            out.append(record)
    return out


def _schedule(raw, events, out):
    specs, compdat = _convert_geometry_records(raw)
    names = [r[0].strip("'\"") for r in specs]
    current = {"WELSPECS": specs, "COMPDAT": compdat}
    controls, switches = [], []
    grouped = {}
    for step, key, records in events:
        grouped.setdefault(step, {}).setdefault(key, []).extend(records)
    for step in sorted(grouped):
        for key, records in grouped[step].items():
            if key == "WCONPROD":
                records = _expand_producer_wildcard(records, names)
                for row in records:
                    for index, unit in ((3, STB/DAY), (4, STB/DAY),
                                        (5, MSCF/DAY), (6, STB/DAY),
                                        (7, STB/DAY), (8, PSIA)):
                        if index < len(row) and not row[index].endswith("*"):
                            row[index] = str(float(row[index])*unit)
            elif key == "WCONINJE":
                for row in records:
                    phase = row[1].strip("'\"").upper()
                    rate_unit = MSCF/DAY if phase == "GAS" else STB/DAY
                    for index, unit in ((4, rate_unit), (5, STB/DAY), (6, PSIA)):
                        if index < len(row) and not row[index].endswith("*"):
                            row[index] = str(float(row[index])*unit)
            current[key] = deepcopy(records)
        controls.append(deepcopy(current)); switches.append(step)
    tsteps = np.array([float(v) for row in raw["TSTEP"] for v in row])*DAY
    out["tsteps"] = tsteps
    out["controls"] = controls
    out["control_switch_steps"] = np.asarray(switches, int)
    out["step_control"] = np.searchsorted(switches, np.arange(tsteps.size), side="right") - 1
    out["schedule"] = {"step": {"val": tsteps, "control": out["step_control"]},
                       "control": controls}


if __name__ == "__main__":
    root = Path(__file__).parents[4]
    d = read_spe9_deck(root.parent/"mrst-2024a-Geomech/examples/data/SPE9/BENCH_SPE9.DATA")
    assert d["cartDims"] == (24, 25, 15) and d["pressure"].size == 9000
    assert d["tsteps"].size == 35 and len(d["controls"]) == 3
    assert len(d["controls"][0]["WCONPROD"]) == 25
    print("SPE9 deck parsed successfully")
