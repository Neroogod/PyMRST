"""Non-neighbour connections from an ECLIPSE case (M2.6).

An NNC is a flow path between two cells that do not share a face - across a
fault throw, between a well's cells and a fractured block, or between the parts
of a grid an LGR splits. They are not decoration: on a faulted model a large
share of the cross-fault flow goes through them, and a case imported without
them is a *sealed* version of the model somebody built.

**These are read into their own object and are not attached to the grid.** That
is the standing rule of this repository - new capability is an add-on to
PyMRST, never a change to it - and `Grid` is MRST-shaped, with no place for a
connection that is not a face. Extending it would put research-studio state
into the replica layer and break the covenant that makes the MRST parity worth
anything.

The consequence is a real limitation and is stated rather than buried: an
imported case's NNCs **do not flow into the standard solvers**. What they
support is inspection - seeing that they exist, how many, which cells they
join, and what transmissibility they carry - which is the difference between
knowing a model is sealed and wondering why it is.

**On the index convention.** ECLIPSE writes `NNC1`/`NNC2` as one-based indices
into the *global* Cartesian box. That is what this assumes, and there is no
ECLIPSE-written file in this repository to confirm it against, so the
assumption is checked rather than trusted: every index must land on an active
cell, and anything that does not is refused with the reason. `indices="active"`
selects the other convention explicitly. Nothing here guesses between them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pymrst.research_studio.eclipse.egrid import read_egrid
from pymrst.research_studio.eclipse.records import read_records


@dataclass(frozen=True, slots=True)
class NonNeighbourConnections:
    """Cell pairs joined by something other than a shared face."""

    cells: np.ndarray                       # (n, 2), zero-based active indices
    transmissibility: np.ndarray | None     # (n,) or None when INIT has none
    source: dict[str, str]

    def __len__(self) -> int:
        return int(self.cells.shape[0])

    @property
    def sealed(self) -> bool:
        """True when every connection carries zero transmissibility.

        Worth asking separately from "are there any": a case can list hundreds
        of NNCs that all transmit nothing, which looks like a connected model
        and behaves like a sealed one.
        """
        if self.transmissibility is None:
            return False
        return bool(np.all(self.transmissibility == 0.0))

    def __str__(self) -> str:
        if not len(self):
            return "no non-neighbour connections"
        detail = "no transmissibilities"
        if self.transmissibility is not None:
            detail = (f"T in [{self.transmissibility.min():.3e}, "
                      f"{self.transmissibility.max():.3e}]")
        return f"{len(self)} non-neighbour connection(s), {detail}"


def _active_lookup(actnum: np.ndarray) -> np.ndarray:
    """Global cell index -> active index, or -1 where the cell is inactive."""
    active = np.asarray(actnum).ravel() > 0
    lookup = np.full(active.size, -1, dtype=np.int64)
    lookup[active] = np.arange(int(active.sum()), dtype=np.int64)
    return lookup


def _collect(path: Path, keyword: str) -> np.ndarray | None:
    """Every record of one keyword, concatenated in file order.

    Concatenated rather than first-wins: a large model splits NNC1 across
    several records, and keeping only the first would silently drop most of the
    connections while looking like it worked.
    """
    found = [np.asarray(values).ravel() for name, value_type, values
             in read_records(path)
             if name.upper() == keyword and value_type != "MESS"]
    return np.concatenate(found) if found else None


def read_nnc(prefix: str | Path, *, indices: str = "global",
             ) -> NonNeighbourConnections:
    """Read `NNC1`/`NNC2` from the EGRID and `TRANNNC` from the INIT.

    `indices` selects how `NNC1`/`NNC2` are interpreted: `"global"` (the
    ECLIPSE convention, one-based into the Cartesian box) or `"active"`
    (one-based into the active cells). There is no automatic detection, because
    on a grid with no inactive cells the two are indistinguishable and a reader
    that guessed would be right on test models and wrong on real ones.
    """
    if indices not in ("global", "active"):
        raise ValueError(f"indices must be 'global' or 'active', got {indices!r}")

    stem = Path(prefix)
    egrid = stem.with_suffix(".EGRID")
    if not egrid.exists():
        raise FileNotFoundError(f"{egrid} not found; the grid is required")

    corner = read_egrid(egrid)
    actnum = np.asarray(corner.actnum, dtype=np.int64).ravel()
    n_active = int((actnum > 0).sum())
    source: dict[str, str] = {"grid": str(egrid)}

    first, second = _collect(egrid, "NNC1"), _collect(egrid, "NNC2")
    if first is None or second is None:
        source["nnc"] = f"{egrid} has no NNC1/NNC2 records"
        return NonNeighbourConnections(np.zeros((0, 2), dtype=np.int64),
                                       None, source)
    if first.size != second.size:
        raise ValueError(
            f"NNC1 has {first.size} entries and NNC2 has {second.size}; a "
            f"connection needs both ends and these cannot be paired")

    pairs = np.column_stack([first, second]).astype(np.int64) - 1   # to 0-based
    if indices == "global":
        if pairs.size and pairs.max() >= actnum.size:
            raise ValueError(
                f"an NNC references global cell {int(pairs.max()) + 1} in a "
                f"grid of {actnum.size}; these are not global indices - pass "
                f"indices='active' if the file uses active numbering")
        lookup = _active_lookup(actnum)
        mapped = lookup[pairs]
        dead = np.flatnonzero(np.any(mapped < 0, axis=1))
        if dead.size:
            raise ValueError(
                f"{dead.size} non-neighbour connection(s) reference inactive "
                f"cells, the first being global pair "
                f"{tuple(int(v) + 1 for v in pairs[dead[0]])}; refusing to drop "
                f"them silently")
    else:
        if pairs.size and pairs.max() >= n_active:
            raise ValueError(
                f"an NNC references active cell {int(pairs.max()) + 1} of "
                f"{n_active}")
        mapped = pairs
    if pairs.size and pairs.min() < 0:
        raise ValueError("NNC indices are one-based; found a value below 1")

    source["nnc"] = str(egrid)
    transmissibility = None
    init = stem.with_suffix(".INIT")
    if init.exists():
        raw = _collect(init, "TRANNNC")
        if raw is not None:
            if raw.size != mapped.shape[0]:
                raise ValueError(
                    f"TRANNNC has {raw.size} values for {mapped.shape[0]} "
                    f"connections; refusing to pair them by position")
            transmissibility = np.asarray(raw, dtype=float)
            source["transmissibility"] = str(init)

    return NonNeighbourConnections(mapped, transmissibility, source)


def write_nnc(path: str | Path, pairs, transmissibility=None, *,
              actnum=None) -> list[tuple[str, str, Any]]:
    """The records an EGRID/INIT would carry for these connections.

    Returned rather than written, so the caller decides which file they belong
    in. Exists so `read_nnc` can be checked against something other than
    itself; `pairs` are zero-based active indices, converted here to the
    one-based global form ECLIPSE writes.
    """
    pairs = np.atleast_2d(np.asarray(pairs, dtype=np.int64))
    if pairs.shape[1] != 2:
        raise ValueError("pairs must have two columns")
    if actnum is None:
        globalised = pairs
    else:
        active = np.flatnonzero(np.asarray(actnum).ravel() > 0)
        globalised = active[pairs]
    records = [("NNC1", "INTE", globalised[:, 0] + 1),
               ("NNC2", "INTE", globalised[:, 1] + 1)]
    if transmissibility is not None:
        records.append(("TRANNNC", "REAL",
                        np.asarray(transmissibility, dtype=float)))
    return records


__all__ = ["NonNeighbourConnections", "read_nnc", "write_nnc"]
