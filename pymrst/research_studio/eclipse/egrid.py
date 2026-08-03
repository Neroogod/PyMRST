"""EGRID corner-point geometry (M2.6).

An EGRID carries the same corner-point description a GRDECL deck does -
`COORD` pillars, `ZCORN` corner depths, `ACTNUM` activity - in ECLIPSE's
unformatted record form rather than as text. PyMRST already reads the text
form through `deckformat`, and that path is validated against MATLAB.

That is what makes this testable at all. There is no ECLIPSE binary output in
this repository or its dataset cache, so a reader here cannot be checked
against a real file. It can be checked against the text path: take a deck
PyMRST already reads, write it as an EGRID, read it back, and require the
arrays to match what the text reader produced. Round-trip alone would only
show the writer and reader share an understanding - correct or otherwise -
which is the trap the transport Jacobian's finite-difference check turned out
to contain.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pymrst.research_studio.eclipse.records import read_records, write_records

EGRID_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class CornerPointGrid:
    """The corner-point arrays, in the shapes `process_grdecl` expects."""

    cartDims: tuple[int, int, int]
    coord: np.ndarray          # 6 per pillar
    zcorn: np.ndarray          # 8 per cell
    actnum: np.ndarray         # 1 per cell

    def to_grdecl(self) -> dict:
        """The mapping `pymrst.gridprocessing.process_grdecl` takes.

        Returned rather than a Grid so the caller decides whether to process
        it, and so the comparison against the text path is array-to-array.
        """
        return {
            "cartDims": np.asarray(self.cartDims, dtype=np.int64),
            "COORD": np.asarray(self.coord, dtype=float),
            "ZCORN": np.asarray(self.zcorn, dtype=float),
            "ACTNUM": np.asarray(self.actnum, dtype=np.int64),
        }


def read_egrid(path: str | Path) -> CornerPointGrid:
    """Read `COORD`, `ZCORN` and `ACTNUM` from an EGRID file."""
    found: dict[str, Any] = {}
    for name, _, values in read_records(path):
        found.setdefault(name, values)

    if "GRIDHEAD" not in found:
        raise ValueError(f"{path} has no GRIDHEAD; is it an EGRID?")
    head = np.asarray(found["GRIDHEAD"], dtype=np.int64).ravel()
    if head.size < 4:
        raise ValueError("GRIDHEAD is too short to carry the grid dimensions")
    # GRIDHEAD[0] is the grid type (1 = corner point); dimensions follow.
    dims = tuple(int(v) for v in head[1:4])
    cells = dims[0]*dims[1]*dims[2]

    for required in ("COORD", "ZCORN"):
        if required not in found:
            raise ValueError(f"{path} has no {required}")

    coord = np.asarray(found["COORD"], dtype=float).ravel()
    zcorn = np.asarray(found["ZCORN"], dtype=float).ravel()
    expected_pillars = (dims[0] + 1)*(dims[1] + 1)*6
    if coord.size != expected_pillars:
        raise ValueError(
            f"COORD has {coord.size} values, expected {expected_pillars} for "
            f"a {dims[0]}x{dims[1]}x{dims[2]} grid"
        )
    if zcorn.size != 8*cells:
        raise ValueError(
            f"ZCORN has {zcorn.size} values, expected {8*cells}"
        )

    actnum = (np.asarray(found["ACTNUM"], dtype=np.int64).ravel()
              if "ACTNUM" in found else np.ones(cells, dtype=np.int64))
    if actnum.size != cells:
        raise ValueError(f"ACTNUM has {actnum.size} values, expected {cells}")

    return CornerPointGrid(dims, coord, zcorn, actnum)


def write_egrid(path: str | Path, grdecl: dict) -> Path:
    """Write a GRDECL-shaped mapping as an EGRID.

    Exists so `read_egrid` can be verified against the text path, and so a
    PyMRST case can be handed to tooling that reads EGRID. `FILEHEAD` and
    `GRIDHEAD` carry the minimum a reader needs; this does not attempt to be
    a complete ECLIPSE grid file.
    """
    dims = np.asarray(grdecl["cartDims"], dtype=np.int64).ravel()
    if dims.size != 3:
        raise ValueError("cartDims must have three entries")
    cells = int(dims[0]*dims[1]*dims[2])

    filehead = np.zeros(100, dtype=np.int64)
    filehead[0] = EGRID_SCHEMA_VERSION
    gridhead = np.zeros(100, dtype=np.int64)
    gridhead[0] = 1                       # corner-point
    gridhead[1:4] = dims

    actnum = grdecl.get("ACTNUM")
    records = [
        ("FILEHEAD", "INTE", filehead),
        ("GRIDHEAD", "INTE", gridhead),
        ("COORD", "REAL", np.asarray(grdecl["COORD"], dtype=float).ravel()),
        ("ZCORN", "REAL", np.asarray(grdecl["ZCORN"], dtype=float).ravel()),
    ]
    if actnum is not None:
        records.append(
            ("ACTNUM", "INTE", np.asarray(actnum, dtype=np.int64).ravel())
        )
    else:
        records.append(("ACTNUM", "INTE", np.ones(cells, dtype=np.int64)))
    records.append(("ENDGRID", "MESS", []))
    return write_records(path, records)


__all__ = ["CornerPointGrid", "EGRID_SCHEMA_VERSION", "read_egrid",
           "write_egrid"]
