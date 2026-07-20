"""Minimal ECLIPSE deck reader for the SPE 1 benchmark."""

from pathlib import Path
from typing import Dict, List

import numpy as np

from pymrst.utils.units import CENTI, DARCY, DAY, FT, MILLI, POISE, POUND, PSIA

STB: float = 0.158987294928
MSCF: float = 1000.0 * FT**3
LB_PER_FT3: float = POUND / FT**3

TABLE_KEYWORDS = {"SWOF", "SGOF", "PVDG"}
VECTOR_KEYWORDS = {"PORO", "PERMX", "PERMY", "PERMZ", "PRESSURE", "SWAT", "SGAS", "RS", "COORD", "ZCORN"}
RECORD_KEYWORDS = {"PVTW", "ROCK", "DENSITY", "DIMENS", "SPECGRID", "WELSPECS", "COMPDAT", "WCONPROD", "WCONINJE", "TSTEP"}


def read_spe1_deck(path) -> Dict:
    """Parse the SPE1 deck into an SI-unit dictionary."""
    path = Path(path)
    tokens = _tokenize(path)
    raw: Dict[str, object] = {}
    index = 0
    while index < len(tokens):
        keyword = tokens[index].upper()
        index += 1
        if keyword in VECTOR_KEYWORDS:
            raw[keyword], index = _read_numbers(tokens, index)
        elif keyword == "PVTO":
            raw[keyword], index = _read_pvto(tokens, index)
        elif keyword in TABLE_KEYWORDS or keyword in RECORD_KEYWORDS:
            records, index = _read_records(tokens, index)
            raw.setdefault(keyword, []).extend(records)
    return _convert(raw)


def _tokenize(path: Path) -> List[str]:
    """Return whitespace tokens with comments stripped and INCLUDEs inlined."""
    tokens: List[str] = []
    for raw in path.read_text().splitlines():
        line = raw.split("--")[0].strip()
        if not line:
            continue
        parts = line.split("'")
        for index in range(0, len(parts), 2):
            parts[index] = parts[index].replace("/", " / ")
        tokens.extend("'".join(parts).split())
    return _inline_includes(path, tokens)


def _inline_includes(path: Path, tokens: List[str]) -> List[str]:
    out: List[str] = []
    index = 0
    while index < len(tokens):
        if tokens[index].upper() == "INCLUDE":
            filename = tokens[index + 1].strip("'\"")
            out.extend(_tokenize(path.parent / filename))
            index += 2
            while index < len(tokens) and tokens[index] != "/":
                index += 1
            index += 1
        else:
            out.append(tokens[index])
            index += 1
    return out


def _expand(token: str) -> List[str]:
    """Expand `n*value` repeat syntax; bare `n*` yields defaults."""
    if "*" in token and not token.startswith("'"):
        count, _, value = token.partition("*")
        if count.isdigit():
            return [value if value else "1*"] * int(count)
    return [token]


def _read_numbers(tokens: List[str], index: int) -> tuple[np.ndarray, int]:
    values: List[float] = []
    while tokens[index] != "/":
        for token in _expand(tokens[index]):
            values.append(float(token))
        index += 1
    return np.array(values), index + 1


def _read_records(tokens: List[str], index: int) -> tuple[List[List[str]], int]:
    records: List[List[str]] = []
    current: List[str] = []
    while True:
        token = tokens[index]
        if token == "/":
            if not current:
                return records, index + 1
            records.append(current)
            current = []
            index += 1
            next_token = tokens[index] if index < len(tokens) else "/"
            if next_token == "/" or not _looks_like_data(next_token):
                if next_token == "/":
                    index += 1
                return records, index
        else:
            current.extend(_expand(token))
            index += 1


def _looks_like_data(token: str) -> bool:
    if token.startswith("'"):
        return True
    try:
        float(token.split("*")[0] or "1")
        return True
    except ValueError:
        return False


def _read_pvto(tokens: List[str], index: int) -> tuple[list, int]:
    branches = []
    while tokens[index] != "/":
        numbers, index = _read_numbers(tokens, index)
        rs, rows = numbers[0], numbers[1:].reshape(-1, 3)
        branches.append((rs, rows))
    return branches, index + 1


def _convert(raw: Dict) -> Dict:
    converted: Dict = {}
    dims = [int(float(value)) for value in raw["DIMENS"][0]]
    converted["cartDims"] = tuple(dims[:3])
    _convert_grid(raw, converted)
    _convert_rock(raw, converted)
    _convert_fluid_tables(raw, converted)
    _convert_initial_state(raw, converted)
    _convert_wells(raw, converted)
    converted["tsteps"] = np.array([float(value) for record in raw["TSTEP"] for value in record]) * DAY
    return converted


def _convert_grid(raw: Dict, converted: Dict) -> None:
    nx, ny, nz = converted["cartDims"]
    coord = raw["COORD"].reshape(-1, 6)
    xs = np.unique(coord[:, 0])
    ys = np.unique(coord[:, 1])
    zs = np.unique(raw["ZCORN"])
    assert xs.size == nx + 1 and ys.size == ny + 1 and zs.size == nz + 1, (
        "SPE1 reader supports regular corner-point geometry only"
    )
    converted["x"], converted["y"], converted["z"] = xs * FT, ys * FT, zs * FT


def _convert_rock(raw: Dict, converted: Dict) -> None:
    converted["poro"] = raw["PORO"]
    converted["perm"] = np.column_stack([raw["PERMX"], raw["PERMY"], raw["PERMZ"]]) * MILLI * DARCY
    rock = np.asarray(raw["ROCK"], dtype=float).ravel()
    converted["rock_cp"] = {"p_ref": rock[0] * PSIA, "c": rock[1] / PSIA}


def _convert_fluid_tables(raw: Dict, converted: Dict) -> None:
    swof = np.asarray(raw["SWOF"], dtype=float).reshape(-1, 4)
    sgof = np.asarray(raw["SGOF"], dtype=float).reshape(-1, 4)
    swof[:, 3] *= PSIA
    sgof[:, 3] *= PSIA
    converted["swof"], converted["sgof"] = swof, sgof
    pvtw = np.asarray(raw["PVTW"], dtype=float).ravel()
    converted["pvtw"] = {
        "p_ref": pvtw[0] * PSIA,
        "b_ref": pvtw[1],
        "c": pvtw[2] / PSIA,
        "mu_ref": pvtw[3] * CENTI * POISE,
        "viscosibility": pvtw[4] / PSIA,
    }
    dens = np.asarray(raw["DENSITY"], dtype=float).ravel() * LB_PER_FT3
    converted["rhoS"] = np.array([dens[1], dens[0], dens[2]])
    pvdg = np.asarray(raw["PVDG"], dtype=float).reshape(-1, 3)
    converted["pvdg"] = np.column_stack([pvdg[:, 0] * PSIA, pvdg[:, 1] * STB / MSCF, pvdg[:, 2] * CENTI * POISE])
    converted["pvto"] = _convert_pvto(raw["PVTO"])


def _convert_pvto(raw_pvto) -> list:
    branches = []
    for rs, rows in raw_pvto:
        rows = rows.copy()
        rows[:, 0] *= PSIA
        rows[:, 2] *= CENTI * POISE
        branches.append((rs * MSCF / STB, rows))
    return branches


def _convert_initial_state(raw: Dict, converted: Dict) -> None:
    converted["pressure"] = raw["PRESSURE"] * PSIA
    converted["swat"] = raw["SWAT"]
    converted["sgas"] = raw["SGAS"]
    converted["rs"] = raw["RS"] * MSCF / STB


def _convert_wells(raw: Dict, converted: Dict) -> None:
    nx, ny, _ = converted["cartDims"]
    wspec = {record[0].strip("'\""): record for record in raw["WELSPECS"]}
    compdat = {record[0].strip("'\""): record for record in raw["COMPDAT"]}
    wells = {}
    for name, rec in compdat.items():
        wells[name] = _well_from_compdat(rec, wspec[name], nx, ny)
    converted["wells"] = wells
    for rec in raw.get("WCONPROD", []):
        name = rec[0].strip("'\"")
        assert rec[2].strip("'\"").upper() == "ORAT", "expected ORAT control"
        wells[name].update(type="orat", val=-float(rec[3]) * STB / DAY, sign=-1, compi=np.array([0.0, 1.0, 0.0]))
    for rec in raw.get("WCONINJE", []):
        name = rec[0].strip("'\"")
        assert rec[1].strip("'\"").upper() == "GAS"
        assert rec[3].strip("'\"").upper() == "RATE"
        wells[name].update(type="rate", val=float(rec[4]) * MSCF / DAY, sign=1, compi=np.array([0.0, 0.0, 1.0]))


def _well_from_compdat(rec: List[str], wspec: List[str], nx: int, ny: int) -> Dict:
    i0, j0 = int(float(rec[1])) - 1, int(float(rec[2])) - 1
    k1, k2 = int(float(rec[3])) - 1, int(float(rec[4])) - 1
    cells = np.array([i0 + nx * j0 + nx * ny * layer for layer in range(k1, k2 + 1)])
    return {
        "name": rec[0].strip("'\""),
        "cells": cells,
        "ref_depth": float(wspec[4]) * FT,
        "radius": float(rec[8]) * FT / 2.0,
    }


if __name__ == "__main__":
    root = Path(__file__).parents[5]
    deck_path = root / "mrst-2024a-Geomech/examples/data/SPE1/BENCH_SPE1.DATA"

    def test_parse_spe1() -> None:
        deck = read_spe1_deck(deck_path)
        assert deck["cartDims"] == (10, 10, 3)
        assert np.isclose(deck["x"][-1], 10000 * FT)
        assert np.allclose(np.diff(deck["z"]) / FT, [20, 30, 50])
        assert deck["poro"].size == 300
        assert np.allclose(deck["poro"], 0.3)
        assert np.isclose(deck["perm"][0, 0], 500 * MILLI * DARCY)
        assert deck["swof"].shape == (14, 4)
        assert deck["sgof"].shape == (15, 4)
        assert len(deck["pvto"]) == 9
        producer = deck["wells"]["PRODUCER"]
        injector = deck["wells"]["INJECTOR"]
        assert producer["cells"][0] == 99 + 200
        assert producer["type"] == "orat"
        assert np.isclose(producer["val"], -20000 * STB / DAY)
        assert injector["cells"][0] == 0
        assert injector["type"] == "rate"
        assert np.isclose(injector["val"], 100000 * MSCF / DAY)
        assert np.isclose(deck["tsteps"].sum() / DAY, 1216.0, atol=20)
        print("ok test_parse_spe1")

    test_parse_spe1()
    print("\nAll tests passed!")

