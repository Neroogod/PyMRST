"""Complete Norne field archive discovery and structural deck parsing."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import hashlib
import re
import urllib.request
import zipfile

import numpy as np

NORNE_FIELD_URL = (
    "https://www.sintef.no/globalassets/project/mrst/"
    "norne_field_model_example_files.zip"
)
NORNE_FIELD_SHA256 = "a9bd18cc726b8565dc352917ed541202d1f8332b756f5d8c6c28913e87df8b0e"
CASE_NAME = "NORNE_ATW2013"

_CONTROL_KEYWORDS = {
    "WELSPECS", "COMPDAT", "WCONPROD", "WCONINJE", "WCONINJH",
    "WCONHIST", "WELOPEN", "WELTARG", "WPIMULT", "WGRUPCON",
    "GCONPROD", "GCONINJE", "GRUPTREE", "DATES", "TSTEP",
}
_STRUCTURAL_KEYWORDS = _CONTROL_KEYWORDS | {
    "FAULTS", "MULTFLT", "MULTREGT", "EQUIL", "SPECGRID", "DIMENS",
    "COORD", "ZCORN", "ACTNUM", "PORO", "PERMX", "PERMY", "PERMZ",
    "NTG", "SATNUM", "PVTNUM", "EQLNUM", "FIPNUM", "FLUXNUM",
    "SWOF", "SGOF", "SWFN", "SGFN", "SOF3", "PVTO", "PVTG", "PVTW",
    "VFPINJ", "VFPPROD", "ROCK", "DENSITY", "RSVD", "RVVD",
    "START",
}
_SECTIONS = {"RUNSPEC", "GRID", "EDIT", "PROPS", "REGIONS", "SOLUTION",
             "SUMMARY", "SCHEDULE", "END"}
_MONTHS = {name: index for index, name in enumerate(
    ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG",
     "SEP", "OCT", "NOV", "DEC"), 1)}


@dataclass
class NorneScheduleMetadata:
    dates: np.ndarray
    wells: tuple[str, ...]
    records: dict[str, list[list[str]]]
    events: list[tuple[str, list[str]]]

    @property
    def control_change_count(self) -> int:
        return sum(len(self.records.get(key, ())) for key in _CONTROL_KEYWORDS)


@dataclass
class NorneFieldDeck:
    root: Path
    data_file: Path
    expanded_text: str
    keyword_counts: Counter
    schedule: NorneScheduleMetadata

    @property
    def faults(self) -> list[list[str]]:
        return self.schedule.records.get("FAULTS", [])

    @property
    def fault_multipliers(self) -> list[list[str]]:
        return self.schedule.records.get("MULTFLT", [])

    def schedule_deck(self) -> dict:
        """Return dated delta controls accepted by schedule conversion.

        Each interval ends at a DATES record and uses every well change since
        the preceding date. The converter carries prior wells forward, so the
        resulting controls preserve the historical schedule without copying
        thousands of raw records into every period.
        """
        start_records = self.schedule.records.get("START", [])
        if not start_records:
            raise ValueError("Norne deck has no START date")
        start = _date(start_records[0])
        previous = start
        pending: dict[str, list[list[str]]] = {}
        values, controls = [], []
        relevant = {
            "WELSPECS", "COMPDAT", "WCONPROD", "WCONHIST", "WCONINJE",
            "WCONINJH", "WELOPEN", "WELTARG", "WPIMULT", "WGRUPCON",
            "GCONPROD", "GCONINJE", "GRUPTREE",
        }
        for keyword, record in self.schedule.events:
            if keyword == "DATES":
                current = _date(record)
                dt = (current-previous).days
                if dt <= 0:
                    raise ValueError(f"non-increasing Norne date {current}")
                values.append(float(dt)*86400.0)
                controls.append(pending)
                pending = {}
                previous = current
            elif keyword in relevant:
                pending.setdefault(keyword, []).append(
                    _metric_schedule_record(keyword, record)
                )
        return {
            "step": {"val": np.asarray(values),
                     "control": np.arange(len(values), dtype=int)},
            "control": controls,
            "start": np.datetime64(start.date()),
        }

    def vfp_tables(self) -> tuple[dict[int, dict], dict[int, dict]]:
        """Decode Norne's metric VFP records into MRST-like table mappings.

        Returns ``(injector, producer)`` dictionaries keyed by the one-based
        table numbers used by ``WCONINJE``/``WCONPROD``. Pressure and rate
        values are converted to SI so the mappings can be passed directly to
        :class:`~pymrst.modules.ad_core.models.facilities.vfp.VFPTable`.
        """
        return (
            _decode_vfpinj(self.schedule.records.get("VFPINJ", [])),
            _decode_vfpprod(self.schedule.records.get("VFPPROD", [])),
        )


def _default_root() -> Path:
    return Path(__file__).resolve().parents[4].parent/"mrst-2024a-Geomech/examples/data/norne_field_model"


def _date(record: list[str]) -> datetime:
    if len(record) < 3:
        raise ValueError(f"invalid Norne date record: {record}")
    month = record[1].upper()[:3]
    if month not in _MONTHS:
        raise ValueError(f"invalid Norne month: {record[1]}")
    return datetime(int(record[2]), _MONTHS[month], int(record[0]))


def _numbers(record):
    return np.asarray([float(value) for value in record], dtype=float)


def _metric_schedule_record(keyword: str, record: list[str]) -> list:
    """Convert a raw Norne METRIC schedule row to PyMRST SI units."""
    out = list(record)

    def scale(indices, factor):
        for index in indices:
            if index >= len(out):
                continue
            token = str(out[index])
            if "*" in token:
                continue
            try:
                out[index] = float(token)*factor
            except ValueError:
                pass

    if keyword == "COMPDAT":
        # Connection factor in cP*m^3/(day*bar).
        scale((7,), 1e-3/(86400.0*1e5))
        # KH is stated in mD*m and is retained for metadata; explicit WI is
        # present throughout Norne and controls the actual completion flux.
        scale((9,), 9.869233e-16)
    elif keyword in {"WCONPROD", "WCONHIST"}:
        scale((3, 4, 5, 6, 7), 1.0/86400.0)
        scale((8, 9), 1e5)
    elif keyword in {"WCONINJE", "WCONINJ"}:
        scale((4, 5), 1.0/86400.0)
        scale((6, 7), 1e5)
    elif keyword == "WCONINJH":
        scale((3,), 1.0/86400.0)
        scale((6,), 1e5)
    elif keyword in {"WELTARG", "WELLTARG"} and len(out) >= 3:
        mode = str(out[1]).upper()
        scale((2,), 1e5 if mode in {"BHP", "THP"} else 1.0/86400.0)
    return out


def _decode_vfpinj(records: list[list[str]]) -> dict[int, dict]:
    """Decode METRIC VFPINJ tables (flow-major BHP arrays)."""
    tables: dict[int, dict] = {}
    index = 0
    while index < len(records):
        header = records[index]
        if len(header) != 3 or not header[0].isdigit():
            index += 1
            continue
        number, depth, flow_type = int(header[0]), float(header[1]), header[2]
        if index + 2 >= len(records):
            raise ValueError(f"incomplete VFPINJ table {number}")
        flow = _numbers(records[index + 1])/86400.0
        thp = _numbers(records[index + 2])*1e5
        index += 3
        rows = []
        while index < len(records):
            row = records[index]
            if len(row) == 3 and row[0].isdigit() and row[2].upper() in {"WAT", "GAS", "OIL"}:
                break
            rows.append(row); index += 1
        bhp = np.full((flow.size, thp.size), np.nan)
        for row in rows:
            ti = int(row[0]) - 1
            values = _numbers(row[1:])*1e5
            if values.size != flow.size or not 0 <= ti < thp.size:
                raise ValueError(f"invalid row in VFPINJ table {number}")
            bhp[:, ti] = values
        if np.isnan(bhp).any():
            raise ValueError(f"incomplete BHP array in VFPINJ table {number}")
        tables[number] = {
            "FLO": flow, "THP": thp, "BHP": bhp, "FLOID": flow_type,
            "depth": depth,
        }
    return tables


def _decode_vfpprod(records: list[list[str]]) -> dict[int, dict]:
    """Decode METRIC VFPPROD tables (flow, THP, WFR, GFR, ALQ)."""
    tables: dict[int, dict] = {}
    index = 0
    flow_ids = {"LIQ", "OIL", "GAS"}
    while index < len(records):
        header = records[index]
        if len(header) < 5 or not header[0].isdigit() or header[2].upper() not in flow_ids:
            index += 1
            continue
        number, depth = int(header[0]), float(header[1])
        if index + 5 >= len(records):
            raise ValueError(f"incomplete VFPPROD table {number}")
        flow = _numbers(records[index + 1])/86400.0
        thp = _numbers(records[index + 2])*1e5
        wfr = _numbers(records[index + 3])
        gfr = _numbers(records[index + 4])
        alq = _numbers(records[index + 5])
        index += 6
        shape = (flow.size, thp.size, wfr.size, gfr.size, alq.size)
        bhp = np.full(shape, np.nan)
        while index < len(records):
            row = records[index]
            if len(row) >= 5 and row[0].isdigit() and row[2].upper() in flow_ids:
                break
            if len(row) < 4:
                raise ValueError(f"invalid row in VFPPROD table {number}")
            ti, wi, gi, ai = (int(value) - 1 for value in row[:4])
            values = _numbers(row[4:])*1e5
            if values.size != flow.size:
                raise ValueError(f"invalid flow row in VFPPROD table {number}")
            bhp[:, ti, wi, gi, ai] = values
            index += 1
        if np.isnan(bhp).any():
            raise ValueError(f"incomplete BHP array in VFPPROD table {number}")
        tables[number] = {
            "FLO": flow, "THP": thp, "WFR": wfr, "GFR": gfr, "ALQ": alq,
            "Q": bhp, "FLOID": header[2], "WFRID": header[3],
            "GFRID": header[4], "ALQID": header[5] if len(header) > 5 else "",
            "depth": depth,
        }
    return tables


def norne_field_directory(root=None) -> Path:
    base = Path(root) if root is not None else _default_root()
    candidates = (base, base/"norne_field_model")
    for candidate in candidates:
        if (candidate/"opm-data/norne"/(CASE_NAME+".DATA")).is_file():
            return candidate
    raise FileNotFoundError(
        f"Norne field archive missing below {base}. Call fetch_norne_field_model()."
    )


def fetch_norne_field_model(destination=None, *, overwrite=False) -> Path:
    destination = Path(destination) if destination else _default_root()
    if not overwrite:
        try:
            return norne_field_directory(destination)
        except FileNotFoundError:
            pass
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination/"norne_field_model.zip"
    urllib.request.urlretrieve(NORNE_FIELD_URL, archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != NORNE_FIELD_SHA256:
        raise IOError(f"Norne field archive checksum mismatch: {digest}")
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(destination)
    archive.unlink()
    return norne_field_directory(destination)


def _expand(path: Path, stack=()) -> str:
    path = path.resolve()
    if path in stack:
        raise ValueError(f"Norne INCLUDE cycle through {path}")
    lines, output, index = path.read_text(errors="ignore").splitlines(), [], 0
    while index < len(lines):
        clean = lines[index].split("--", 1)[0].strip()
        if clean.upper() != "INCLUDE":
            output.append(lines[index]); index += 1; continue
        index += 1
        name_parts = []
        while index < len(lines):
            part = lines[index].split("--", 1)[0].strip()
            name_parts.append(part)
            index += 1
            if "/" in part:
                break
        match = re.search(r"['\"]([^'\"]+)['\"]", " ".join(name_parts))
        if match is None:
            raise ValueError(f"malformed INCLUDE in {path}")
        child = (path.parent/match.group(1)).resolve()
        if not child.is_file():
            raise FileNotFoundError(child)
        output.append(f"-- BEGIN INCLUDE {child}")
        output.append(_expand(child, stack+(path,)))
        output.append(f"-- END INCLUDE {child}")
    return "\n".join(output)


def _records(text: str):
    found: dict[str, list[list[str]]] = {key: [] for key in _STRUCTURAL_KEYWORDS}
    counts = Counter()
    events: list[tuple[str, list[str]]] = []
    active = None
    current: list[str] = []
    token_pattern = re.compile(r"'(?:[^']*)'|/|[^\s/]+")
    for raw in text.splitlines():
        line = raw.split("--", 1)[0].strip()
        if not line:
            continue
        first = line.split(None, 1)[0].upper()
        words = line.split()
        is_keyword_line = (
            len(words) == 1 and
            re.fullmatch(r"[A-Z][A-Z0-9_-]{1,11}", first) is not None
        )
        if first in _STRUCTURAL_KEYWORDS or first in _SECTIONS or is_keyword_line:
            if active is not None and current:
                found[active].append(current); events.append((active, current)); current = []
            counts[first] += 1
            active = first if first in _STRUCTURAL_KEYWORDS else None
            remainder = line[len(line.split(None, 1)[0]):]
        elif active is not None:
            remainder = line
        else:
            continue
        if active is None:
            continue
        for token in token_pattern.findall(remainder):
            if token == "/":
                if current:
                    found[active].append(current); events.append((active, current)); current = []
            else:
                current.append(token.strip("'"))
    if active is not None and current:
        found[active].append(current); events.append((active, current))
    return found, counts, events


def read_norne_field_deck(root=None) -> NorneFieldDeck:
    root = norne_field_directory(root)
    data_file = root/"opm-data/norne"/(CASE_NAME+".DATA")
    text = _expand(data_file)
    records, counts, events = _records(text)
    dates = []
    for record in records["DATES"]:
        if len(record) >= 3 and record[1].upper()[:3] in _MONTHS:
            dates.append(datetime(int(record[2]), _MONTHS[record[1].upper()[:3]],
                                  int(record[0])))
    well_names = set()
    for keyword in ("WELSPECS", "COMPDAT", "WCONPROD", "WCONINJE",
                    "WCONHIST", "WCONINJH", "WELOPEN", "WELTARG"):
        for record in records[keyword]:
            if record:
                well_names.add(record[0])
    schedule = NorneScheduleMetadata(
        np.asarray(dates, dtype="datetime64[D]"), tuple(sorted(well_names)),
        records, events,
    )
    return NorneFieldDeck(root, data_file, text, counts, schedule)


def norne_output_prefix(root=None, *, simulator="opm") -> Path:
    root = norne_field_directory(root)
    base = root/"opm-tests/norne"
    if simulator.lower() == "eclipse":
        return base/"ECL.2014.2"/CASE_NAME
    return base/"opm-simulation-reference/flow_legacy"/CASE_NAME


__all__ = [
    "CASE_NAME", "NorneFieldDeck", "NorneScheduleMetadata",
    "fetch_norne_field_model", "norne_field_directory", "norne_output_prefix",
    "read_norne_field_deck",
]
