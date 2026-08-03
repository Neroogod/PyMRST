"""Write a PyMRST case as ECLIPSE output (M3.8, ECLIPSE half).

Every other reader in this package goes one way. This is the first writer, and
writing is not the mirror image of reading: a reader may ignore a keyword it
does not support and still produce a usable case, while a writer that omits
one produces a file the other tool either rejects or - the case that matters -
reads as a *different* model without complaint.

So this package is defined by what it refuses:

* a **declared supported set**, and a hard failure outside it. There is no
  partial export. A file that is missing something is worse than no file,
  because it opens.
* **no invented values.** A case with no rock produces no INIT, not an INIT
  full of plausible defaults.
* **no unit conversion**, for the same reason the readers do none: the values
  are written as PyMRST holds them, and the unit system is the caller's to
  declare in the deck that accompanies them.

Acceptance is the round trip closing through `read_eclipse_case`, which is a
weaker claim than an external simulator opening the file and is stated as
such. CMG is a separate package (M2.10) and nothing here attempts it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from pymrst.research_studio.eclipse.egrid import write_egrid
from pymrst.research_studio.eclipse.records import write_records
from pymrst.research_studio.evidence.manifest import ReproducibilityManifest

# What a state field is called on the way out. Anything outside this map is a
# field ECLIPSE has no keyword for, and is refused rather than dropped.
_STATE_KEYWORDS = {
    "pressure": "PRESSURE",
    "sW": "SWAT",
    "sG": "SGAS",
    "sO": "SOIL",
}

_SUPPORTED = {
    "grid": ("corner-point via EGRID: COORD, ZCORN, ACTNUM",),
    "rock": ("PORO", "PERMX", "PERMY", "PERMZ"),
    "state": tuple(sorted(_STATE_KEYWORDS)),
    "refused": (
        "a state field with no ECLIPSE keyword",
        "a rock without both porosity and permeability",
        "an array whose length is not the cell count",
    ),
    "not_written": ("wells and completions", "schedule and controls",
                    "PVT and relative permeability", "NNC", "LGR",
                    "unit-system declaration"),
    "runnable_deck_subset": (
        "single-phase water", "METRIC or FIELD units", "corner-point grid",
        "isotropic/diagonal rock", "uniform initial pressure", "TSTEP schedule",
        "no wells, aquifers, faults, NNC, LGR or regions",
    ),
}


class UnsupportedForExport(ValueError):
    """The case uses something this writer does not represent."""


@dataclass(frozen=True, slots=True)
class ExportedCase:
    prefix: Path
    files: tuple[Path, ...]

    def __str__(self) -> str:
        return f"{self.prefix.name}: {', '.join(p.name for p in self.files)}"


@dataclass(frozen=True, slots=True)
class RunnableWaterDeck:
    """The deliberately narrow ECLIPSE deck subset this writer certifies."""

    units: str
    initial_pressure: float
    timesteps: tuple[float, ...]
    water_pvt: tuple[float, float, float, float, float]
    title: str = "PyMRST single-phase water export"

    def __post_init__(self):
        units = self.units.upper()
        if units not in {"METRIC", "FIELD"}:
            raise UnsupportedForExport("water deck units must be METRIC or FIELD")
        if not np.isfinite(self.initial_pressure) or self.initial_pressure <= 0:
            raise UnsupportedForExport("initial pressure must be finite and positive")
        steps = tuple(float(value) for value in self.timesteps)
        if not steps or any(not np.isfinite(value) or value <= 0 for value in steps):
            raise UnsupportedForExport("TSTEP values must be finite and positive")
        if len(self.water_pvt) != 5 or any(not np.isfinite(value)
                                           for value in self.water_pvt):
            raise UnsupportedForExport("PVTW requires five finite values")
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "timesteps", steps)
        object.__setattr__(self, "water_pvt", tuple(map(float, self.water_pvt)))


def export_support() -> dict[str, tuple[str, ...]]:
    """What can be written, and - the important half - what cannot.

    `not_written` is the list that decides whether an exported case is
    *runnable* elsewhere or merely *loadable*. It is not: a deck needs wells,
    a schedule and fluid properties, none of which this writes.
    """
    return {key: tuple(value) for key, value in _SUPPORTED.items()}


def _check_length(values, expected: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).ravel()
    if array.size != expected:
        raise UnsupportedForExport(
            f"{name} has {array.size} values for {expected} cells; a partial "
            f"array cannot be written correctly and will not be guessed at"
        )
    return array


def write_eclipse_case(
    prefix: str | Path,
    grid: Any,
    *,
    grdecl: Mapping[str, Any],
    rock: Any = None,
    states: Sequence[Mapping[str, Any]] = (),
    times: Sequence[float] | None = None,
    manifest: ReproducibilityManifest | None = None,
) -> ExportedCase:
    """Write `<prefix>.EGRID`, and `.INIT`/`.UNRST` when there is data for them.

    `grdecl` is required and is the corner-point description the grid was
    built from - COORD, ZCORN, ACTNUM and cartDims. It is asked for rather
    than reconstructed from the processed grid, because that reconstruction is
    lossy for faulted geometry and a silently approximated fault is exactly
    the kind of error this package refuses to make.
    """
    stem = Path(prefix)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for required in ("cartDims", "COORD", "ZCORN"):
        if required not in grdecl:
            raise UnsupportedForExport(
                f"grdecl has no {required}; the corner-point description is "
                f"required and is not reconstructed from the processed grid"
            )

    written = [write_egrid(stem.with_suffix(".EGRID"), grdecl)]
    cells = int(grid.cells.num)

    if rock is not None:
        poro = getattr(rock, "poro", None)
        perm = getattr(rock, "perm", None)
        if poro is None or perm is None:
            raise UnsupportedForExport(
                "rock must carry both porosity and permeability; an INIT with "
                "one of them is not a rock any simulator can use"
            )
        perm = np.atleast_2d(np.asarray(perm, dtype=float))
        if perm.shape[0] != cells:
            raise UnsupportedForExport(
                f"permeability has {perm.shape[0]} rows for {cells} cells"
            )
        records = [
            ("INTEHEAD", "INTE", np.array([1, cells] + [0]*93, dtype=np.int64)),
            ("PORO", "REAL", _check_length(poro, cells, "porosity")),
        ]
        for index, keyword in enumerate(("PERMX", "PERMY", "PERMZ")):
            column = perm[:, index] if perm.shape[1] > index else perm[:, -1]
            records.append((keyword, "REAL", np.asarray(column, dtype=float)))
        written.append(write_records(stem.with_suffix(".INIT"), records))

    if states:
        if times is not None and len(times) != len(states):
            raise UnsupportedForExport(
                f"{len(times)} times for {len(states)} states"
            )
        records = []
        for index, state in enumerate(states):
            unknown = [k for k in state if k not in _STATE_KEYWORDS]
            if unknown:
                raise UnsupportedForExport(
                    f"state {index} carries {sorted(unknown)}, which ECLIPSE "
                    f"has no keyword for here; writing the rest would produce "
                    f"a file that silently describes a different model"
                )
            records.append(("SEQNUM", "INTE", np.array([index], dtype=np.int64)))
            for field, keyword in _STATE_KEYWORDS.items():
                if field in state:
                    records.append((
                        keyword, "REAL",
                        _check_length(state[field], cells, f"{field} at step {index}"),
                    ))
        written.append(write_records(stem.with_suffix(".UNRST"), records))

    if manifest is not None:
        manifest_path = stem.with_suffix(".pymrst-manifest.json")
        manifest_path.write_text(manifest.to_json()+"\n", encoding="utf-8")
        written.append(manifest_path)

    return ExportedCase(stem, tuple(written))


def _keyword(name: str, values) -> str:
    flat = np.asarray(values).reshape(-1)
    body = "\n".join(" ".join(f"{value:.16g}" for value in flat[i:i+6])
                     for i in range(0, flat.size, 6))
    return f"{name}\n{body} /\n"


def write_runnable_water_deck(
    path: str | Path, grid: Any, rock: Any, *, grdecl: Mapping[str, Any],
    specification: RunnableWaterDeck,
    manifest: ReproducibilityManifest,
    unsupported_features: Sequence[str] = (),
) -> Path:
    """Write the certified no-well single-phase-water ECLIPSE deck subset.

    This is called runnable because it contains RUNSPEC through SCHEDULE, not
    because an external ECLIPSE executable has certified it. Independent
    external-simulator acceptance remains separate in the implementation ledger.
    """
    if unsupported_features:
        raise UnsupportedForExport(
            f"runnable water subset cannot represent {sorted(unsupported_features)}")
    for required in ("cartDims", "COORD", "ZCORN"):
        if required not in grdecl:
            raise UnsupportedForExport(f"runnable deck requires grdecl.{required}")
    cells = int(grid.cells.num)
    poro = getattr(rock, "poro", None)
    perm = getattr(rock, "perm", None)
    if poro is None or perm is None:
        raise UnsupportedForExport("runnable deck requires porosity and permeability")
    poro = _check_length(poro, cells, "porosity")
    perm = np.atleast_2d(np.asarray(perm, dtype=float))
    if perm.shape[0] != cells:
        raise UnsupportedForExport("permeability row count differs from active cells")
    dims = tuple(int(value) for value in np.asarray(grdecl["cartDims"]).ravel())
    actnum = np.asarray(grdecl.get("ACTNUM", np.ones(np.prod(dims))), dtype=int).ravel()
    if int(np.count_nonzero(actnum)) != cells:
        raise UnsupportedForExport("ACTNUM active count differs from processed grid")
    pressure = np.full(cells, specification.initial_pressure)
    pvtw = " ".join(f"{value:.16g}" for value in specification.water_pvt)
    text = (
        "RUNSPEC\n"
        f"TITLE\n{specification.title}\n"
        f"{specification.units}\nWATER\n"
        f"DIMENS\n{dims[0]} {dims[1]} {dims[2]} /\n"
        "GRID\n" + _keyword("COORD", grdecl["COORD"])
        + _keyword("ZCORN", grdecl["ZCORN"])
        + _keyword("ACTNUM", actnum) + "PROPS\n"
        + _keyword("PORO", poro) + _keyword("PERMX", perm[:, 0])
        + _keyword("PERMY", perm[:, min(1, perm.shape[1]-1)])
        + _keyword("PERMZ", perm[:, min(2, perm.shape[1]-1)])
        + f"PVTW\n{pvtw} /\n"
        + "SOLUTION\n" + _keyword("PRESSURE", pressure)
        + "SUMMARY\nFPR\n/\nSCHEDULE\n"
        + _keyword("TSTEP", specification.timesteps) + "END\n"
    )
    target = Path(path)
    if target.suffix.upper() != ".DATA":
        target = target.with_suffix(".DATA")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="ascii")
    target.with_suffix(".pymrst-manifest.json").write_text(
        manifest.to_json()+"\n", encoding="utf-8")
    return target


__all__ = ["ExportedCase", "RunnableWaterDeck", "UnsupportedForExport",
           "export_support", "write_eclipse_case", "write_runnable_water_deck"]
