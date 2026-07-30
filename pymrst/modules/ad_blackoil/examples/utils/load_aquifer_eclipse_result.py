"""Load ECLIPSE grid, restart states, and aquifer summary histories.

Ports MRST ``loadAquiferEclipseResult.m`` and extends it to unified restart
files and SMSPEC/UNSMRY aquifer vectors. ECLIPSE restart pressure is stored in
bar; PyMRST states use Pa, so conversion happens at the importer boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from pymrst.modules.ad_core.utils.read_summary_local import _fields, read_summary_local
from pymrst.utils.units import BARSA


@dataclass
class EclipseResultGrid:
    cart_dims: tuple[int, int, int] | None
    active: np.ndarray
    coord: np.ndarray | None = None
    zcorn: np.ndarray | None = None

    @property
    def num_active(self) -> int:
        return int(self.active.size)

    def to_pymrst_grid(self, *, compute_geometry: bool = True):
        """Reconstruct the active corner-point grid from EGRID geometry."""
        if self.cart_dims is None or self.coord is None or self.zcorn is None:
            raise ValueError("EGRID lacks GRIDHEAD, COORD, or ZCORN")
        from pymrst.gridprocessing import compute_geometry as compute, process_grdecl

        nc = int(np.prod(self.cart_dims))
        actnum = np.zeros(nc, dtype=np.int64)
        actnum[self.active] = 1
        grid = process_grdecl({
            "cartDims": self.cart_dims,
            "COORD": self.coord,
            "ZCORN": self.zcorn,
            "ACTNUM": actnum,
        })
        return compute(grid) if compute_geometry else grid


@dataclass
class EclipseAquiferState:
    pressure: np.ndarray
    saturation: np.ndarray
    report_step: int
    raw: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def s(self) -> np.ndarray:
        return self.saturation


@dataclass
class EclipseAquiferSummary:
    time: np.ndarray
    vectors: dict[tuple[str, str], np.ndarray]

    def get(self, keyword: str, name: str | None = None) -> np.ndarray:
        matches = [value for (vector_name, key), value in self.vectors.items()
                   if key == keyword and (name is None or vector_name == name)]
        if not matches:
            raise KeyError((name, keyword))
        if len(matches) > 1 and name is None:
            return np.vstack(matches)
        return matches[0]

    @property
    def aquifer_vectors(self) -> dict[tuple[str, str], np.ndarray]:
        return {key: value for key, value in self.vectors.items()
                if key[1].upper().startswith(("AAQ", "AQP", "AQR"))}


@dataclass
class AquiferEclipseResult:
    states: list[EclipseAquiferState]
    grid: EclipseResultGrid
    summary: EclipseAquiferSummary | None
    prefix: Path

    def __iter__(self):
        """Backward-compatible ``states, grid = result`` unpacking."""
        yield self.states
        yield self.grid

    def compare_states(self, pymrst_states: Iterable) -> dict[str, np.ndarray]:
        """Cellwise relative pressure and absolute water-saturation errors."""
        py = list(pymrst_states)
        if len(py) != len(self.states):
            raise ValueError(
                f"state count differs: ECLIPSE={len(self.states)}, PyMRST={len(py)}"
            )
        pressure, sw = [], []
        for eclipse, state in zip(self.states, py):
            p = np.asarray(state["pressure"], dtype=float).ravel()
            water = np.asarray(state["sW"], dtype=float).ravel()
            if p.size != eclipse.pressure.size:
                raise ValueError("active-cell count differs between ECLIPSE and PyMRST")
            pressure.append((p - eclipse.pressure) /
                            np.maximum(np.abs(eclipse.pressure), 1e-300))
            sw.append(water - eclipse.saturation[:, 0])
        return {"pressure_relative": np.vstack(pressure),
                "water_saturation": np.vstack(sw)}

    def compare_aquifer_history(
        self, pymrst_history: dict[str, np.ndarray], *,
        pressure_keyword: str = "AAQP", volume_keyword: str = "AAQT",
        name: str | None = None,
    ) -> dict[str, np.ndarray]:
        """Compare summary aquifer pressure/cumulative influx with PyMRST."""
        if self.summary is None:
            raise ValueError("no SMSPEC/UNSMRY summary pair was loaded")
        out = {}
        if "aquifer_p" in pymrst_history:
            ecl = np.ravel(self.summary.get(pressure_keyword, name)) * BARSA
            py = np.asarray(pymrst_history["aquifer_p"], dtype=float).reshape(-1)
            out["aquifer_pressure_relative"] = (
                py - ecl[-py.size:]
            ) / np.maximum(np.abs(ecl[-py.size:]), 1e-300)
        if "aquifer_V" in pymrst_history:
            ecl = np.ravel(self.summary.get(volume_keyword, name))
            py = np.asarray(pymrst_history["aquifer_V"], dtype=float).reshape(-1)
            out["aquifer_volume"] = py - ecl[-py.size:]
        return out


def _collect(path: Path) -> dict[str, list[np.ndarray]]:
    output: dict[str, list[np.ndarray]] = {}
    for name, _, values in _fields(path):
        output.setdefault(name, []).append(np.asarray(values))
    return output


def _last(fields, name, default=None):
    values = fields.get(name)
    return default if not values else values[-1]


def _grid(prefix: Path, init_fields) -> EclipseResultGrid:
    egrid = _collect(prefix.with_suffix(".EGRID"))
    head = _last(egrid, "GRIDHEAD")
    dims = None
    if head is not None and np.asarray(head).size >= 4:
        dims = tuple(int(v) for v in np.asarray(head).ravel()[1:4])
    actnum = _last(egrid, "ACTNUM", _last(init_fields, "ACTNUM"))
    if actnum is not None:
        active = np.flatnonzero(np.asarray(actnum).ravel() != 0)
    else:
        porv = _last(init_fields, "PORV")
        active = (np.flatnonzero(np.asarray(porv).ravel() > 0)
                  if porv is not None else np.zeros(0, dtype=np.int64))
    return EclipseResultGrid(
        dims, active.astype(np.int64),
        coord=None if _last(egrid, "COORD") is None else np.asarray(_last(egrid, "COORD"), float),
        zcorn=None if _last(egrid, "ZCORN") is None else np.asarray(_last(egrid, "ZCORN"), float),
    )


def _state(fields, report_step: int, pressure_scale: float) -> EclipseAquiferState:
    pressure = _last(fields, "PRESSURE")
    sw = _last(fields, "SWAT")
    if pressure is None or sw is None:
        raise ValueError(f"restart step {report_step} lacks PRESSURE or SWAT")
    pressure = np.asarray(pressure, dtype=float).ravel() * pressure_scale
    sw = np.asarray(sw, dtype=float).ravel()
    if pressure.size != sw.size:
        raise ValueError(f"restart step {report_step} PRESSURE/SWAT sizes differ")
    raw = {name: np.asarray(values[-1]).copy() for name, values in fields.items()}
    return EclipseAquiferState(
        pressure, np.column_stack([sw, 1.0 - sw]), report_step, raw,
    )


def _restart_states(prefix: Path, pressure_scale: float):
    split = sorted(prefix.parent.glob(prefix.name + ".X[0-9][0-9][0-9][0-9]"))
    if split:
        return [_state(_collect(path), int(path.suffix[2:]), pressure_scale)
                for path in split]
    unrst = prefix.with_suffix(".UNRST")
    if not unrst.exists():
        raise FileNotFoundError(
            f"no split X0001 restart files or unified restart {unrst}"
        )
    groups, current, step = [], {}, 0
    for name, _, values in _fields(unrst):
        if name == "SEQNUM":
            if current:
                groups.append(_state(current, step, pressure_scale))
            step = int(np.asarray(values).ravel()[0])
            current = {}
        else:
            current.setdefault(name, []).append(np.asarray(values))
    if current:
        groups.append(_state(current, step, pressure_scale))
    return groups


def _summary(prefix: Path) -> EclipseAquiferSummary | None:
    if not prefix.with_suffix(".SMSPEC").exists() or not prefix.with_suffix(".UNSMRY").exists():
        return None
    summary, _ = read_summary_local(prefix)
    vectors = {}
    for row, (ni, ki) in enumerate(zip(summary.nInx, summary.kInx)):
        vectors[(summary.WGNAMES[int(ni)], summary.KEYWORDS[int(ki)])] = summary.data[row].copy()
    time_rows = [value for (name, keyword), value in vectors.items()
                 if keyword == "TIME"]
    time = np.asarray(time_rows[0] if time_rows else np.arange(summary.data.shape[1]), float)
    return EclipseAquiferSummary(time, vectors)


def load_aquifer_eclipse_result(
    directory_or_prefix: str | Path,
    fnroot: str | None = None,
    *, pressure_scale: float = BARSA,
) -> AquiferEclipseResult:
    """Load EGRID/INIT, restart states, and optional aquifer summary vectors."""
    first = Path(directory_or_prefix)
    prefix = first / fnroot if fnroot is not None else first
    if prefix.suffix.upper() in {".DATA", ".INIT", ".EGRID", ".UNRST"}:
        prefix = prefix.with_suffix("")
    init_path = prefix.with_suffix(".INIT")
    egrid_path = prefix.with_suffix(".EGRID")
    if not init_path.exists():
        raise FileNotFoundError(init_path)
    if not egrid_path.exists():
        raise FileNotFoundError(egrid_path)
    init_fields = _collect(init_path)
    states = _restart_states(prefix, float(pressure_scale))
    grid = _grid(prefix, init_fields)
    if grid.active.size == 0 and states:
        grid.active = np.arange(states[0].pressure.size, dtype=np.int64)
    if states and grid.num_active != states[0].pressure.size:
        raise ValueError(
            f"grid has {grid.num_active} active cells but restart has "
            f"{states[0].pressure.size} values"
        )
    return AquiferEclipseResult(states, grid, _summary(prefix), prefix)


loadAquiferEclipseResult = load_aquifer_eclipse_result

__all__ = [
    "AquiferEclipseResult", "EclipseAquiferState", "EclipseAquiferSummary",
    "EclipseResultGrid", "load_aquifer_eclipse_result",
    "loadAquiferEclipseResult",
]
