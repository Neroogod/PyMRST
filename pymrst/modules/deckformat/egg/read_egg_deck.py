"""Dataset loader and METRIC deck reader for the EGG benchmark."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import re
import urllib.request
import zipfile

import numpy as np

from pymrst.modules.deckformat.spe1.read_spe1_deck import _expand, _read_numbers, _read_records
from pymrst.utils.units import BARSA, CENTI, DARCY, DAY, MILLI, POISE

ARTICLE = "0e53ddd5-33b7-44f2-9ca4-921880825e47"
FILE = "5867bc9e-4bf6-4e07-827c-4a6669a539ce"
DOWNLOAD_URL = f"https://data.4tu.nl/file/{ARTICLE}/{FILE}"
MD5 = "50d270d9f769f2ca4bb1a8c10b49852e"


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[4]


def egg_data_directory(root=None) -> Path:
    """Return the installed dataset directory, accepting both archive layouts."""
    base = Path(root) if root else _workspace_root().parent/"mrst-2024a-Geomech/examples/data/Egg"
    candidates = (base, base/"Egg_Model_Data_Files_v2")
    for candidate in candidates:
        if (candidate/"MRST/Egg_Model_ECL.DATA").is_file():
            return candidate
    raise FileNotFoundError(
        f"EGG dataset is missing under {base}. Call fetch_egg_dataset() or "
        "extract the official data.zip there."
    )


def fetch_egg_dataset(destination=None, *, overwrite=False) -> Path:
    """Download and verify the official public 4TU EGG archive."""
    destination = Path(destination) if destination else (
        _workspace_root().parent/"mrst-2024a-Geomech/examples/data/Egg"
    )
    if not overwrite:
        try:
            return egg_data_directory(destination)
        except FileNotFoundError:
            pass
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination/"data.zip"
    urllib.request.urlretrieve(DOWNLOAD_URL, archive)
    if hashlib.md5(archive.read_bytes()).hexdigest() != MD5:
        raise IOError("downloaded EGG archive failed its official MD5 checksum")
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(destination)
    archive.unlink()
    return egg_data_directory(destination)


def _tokens(path: Path):
    tokens = []
    for line in path.read_text(errors="ignore").splitlines():
        line = line.split("--", 1)[0].replace(",", " ")
        tokens.extend(re.findall(r"'(?:[^']*)'|/|[^\s/]+", line))
    out, index = [], 0
    while index < len(tokens):
        if tokens[index].upper() == "INCLUDE":
            filename = tokens[index + 1].strip("'\"")
            out.extend(_tokens(path.parent/filename))
            index += 2
            if index < len(tokens) and tokens[index] == "/": index += 1
        else:
            out.append(tokens[index]); index += 1
    return out


def _clean(records):
    return [[v[1:-1].strip() if len(v) > 1 and v[0] == v[-1] == "'" else v
             for v in row] for row in records]


def read_egg_deck(path=None, *, realization=0, data_root=None):
    """Read base realization 0 or one of the 100 permeability realizations."""
    if not 0 <= int(realization) <= 100:
        raise ValueError("EGG realization must be between 0 and 100")
    data = egg_data_directory(data_root)
    path = Path(path) if path else data/"MRST/Egg_Model_ECL.DATA"
    tokens, raw, index = _tokens(path), {}, 0
    vectors = {"ACTNUM", "DX", "DY", "DZ", "TOPS", "PERMX", "PERMY",
               "PERMZ", "PORO"}
    records_kw = {"DIMENS", "SPECGRID", "PVCDO", "PVTW", "ROCK", "DENSITY",
                  "SWOF", "WELSPECS", "COMPDAT", "WCONPROD", "WCONINJE"}
    while index < len(tokens):
        key = tokens[index].upper(); index += 1
        if key in vectors:
            raw[key], index = _read_numbers(tokens, index)
        elif key == "TSTEP":
            vals = []
            while tokens[index] != "/": vals.extend(_expand(tokens[index])); index += 1
            index += 1; raw.setdefault(key, []).extend(vals)
        elif key in records_kw:
            values, index = _read_records(tokens, index)
            raw.setdefault(key, []).extend(_clean(values))
    if realization:
        perm_tokens = _tokens(data/f"Permeability_Realizations/PERM{int(realization)}_ECL.INC")
        p_raw, i = {}, 0
        while i < len(perm_tokens):
            key = perm_tokens[i].upper(); i += 1
            if key in {"PERMX", "PERMY", "PERMZ"}:
                p_raw[key], i = _read_numbers(perm_tokens, i)
        raw.update(p_raw)
    return _convert(raw, int(realization), data)


def _convert(raw, realization, data):
    dims = tuple(int(float(v)) for v in raw.get("DIMENS", raw["SPECGRID"])[0][:3])
    n = int(np.prod(dims))
    kx = raw["PERMX"]
    ky = raw.get("PERMY", kx)
    kz = raw.get("PERMZ", 0.1*kx)
    pvcd = np.asarray(raw["PVCDO"], float).ravel()
    pvtw = np.asarray(raw["PVTW"], float).ravel()
    density = np.asarray(raw["DENSITY"], float).ravel()
    swof = np.asarray(raw["SWOF"], float).reshape(-1, 4)
    specs = [deepcopy(row) for row in raw["WELSPECS"] if len(row) >= 6]
    compdat = [deepcopy(row) for row in raw["COMPDAT"] if len(row) >= 6]
    # Defaults in COMPDAT intentionally leave I/J to WELSPECS; all numeric
    # quantities are already METRIC except pressures (bar -> Pa).
    controls = [{"WELSPECS": specs, "COMPDAT": compdat,
                 "WCONPROD": deepcopy(raw["WCONPROD"]),
                 "WCONINJE": deepcopy(raw["WCONINJE"])}]
    for row in controls[0]["WCONPROD"]:
        if len(row) > 8 and not row[8].endswith("*"): row[8] = str(float(row[8])*BARSA)
    for row in controls[0]["WCONINJE"]:
        if len(row) > 4 and not row[4].endswith("*"): row[4] = str(float(row[4])/DAY)
        if len(row) > 6 and not row[6].endswith("*"): row[6] = str(float(row[6])*BARSA)
    steps = np.asarray(raw["TSTEP"], float)*DAY
    return {
        "cartDims": dims, "actnum": np.asarray(raw["ACTNUM"], bool),
        "dx": raw["DX"], "dy": raw["DY"], "dz": raw["DZ"], "tops": raw["TOPS"],
        "poro": raw["PORO"],
        "perm": np.column_stack([kx, ky, kz])*MILLI*DARCY,
        "pvcd": {"p_ref": pvcd[0]*BARSA, "b_ref": pvcd[1], "c": pvcd[2]/BARSA,
                 "mu_ref": pvcd[3]*CENTI*POISE, "viscosibility": pvcd[4]/BARSA},
        "pvtw": {"p_ref": pvtw[0]*BARSA, "b_ref": pvtw[1], "c": pvtw[2]/BARSA,
                 "mu_ref": pvtw[3]*CENTI*POISE, "viscosibility": pvtw[4]/BARSA},
        "rhoS": np.array([density[1], density[0]]), "swof": swof,
        "tsteps": steps, "controls": controls,
        "schedule": {"step": {"val": steps, "control": np.zeros(steps.size, int)},
                     "control": controls},
        "realization": realization, "data_root": data,
    }
