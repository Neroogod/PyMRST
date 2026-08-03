"""Assemble an ECLIPSE output set into a PyMRST case (M2.6).

`read_egrid`, `read_init` and `read_restart` return arrays. This turns them
into the objects PyMRST actually runs and plots: a processed `Grid`, a `Rock`,
and the states from each report step.

**Active cells are the subtlety.** ECLIPSE writes INIT and UNRST arrays over
*active* cells only, while EGRID's ACTNUM is over the full Cartesian box. A
reader that assumes both are the same length works on a grid with no inactive
cells and silently misplaces every property on a real field model - where the
mismatch is usually large and always in the interesting parts. So the lengths
are checked and mapped explicitly, and anything that does not fit either
convention is refused rather than reshaped.

Nothing here converts units. The values are as ECLIPSE stored them, in
whatever system its RUNSPEC declared, which INIT does not carry.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pymrst.research_studio.eclipse.egrid import read_egrid
from pymrst.research_studio.eclipse.solution import read_init, read_restart

# INIT keyword -> the permeability column it fills. Ordered, because Rock takes
# the diagonal in this order.
_PERM_KEYWORDS = ("PERMX", "PERMY", "PERMZ")


@dataclass(frozen=True, slots=True)
class EclipseCase:
    """A case another simulator ran, in PyMRST's own objects."""

    grid: Any
    rock: Any | None
    states: tuple[dict, ...]
    active: np.ndarray
    source: dict[str, str]

    def __str__(self) -> str:
        return (f"EclipseCase({self.grid.cells.num} active cells, "
                f"{len(self.states)} report step(s), "
                f"rock={'yes' if self.rock is not None else 'no'})")


def _map_to_active(values: np.ndarray, active: np.ndarray, name: str) -> np.ndarray:
    """An INIT/UNRST array placed onto the grid's active cells.

    Accepts either convention - a value per active cell, or one per Cartesian
    cell - and refuses anything else. Guessing between them is what puts
    porosity in the wrong layer.
    """
    values = np.asarray(values).ravel()
    n_active = int(active.sum())
    if values.size == n_active:
        return values
    if values.size == active.size:
        return values[active]
    raise ValueError(
        f"{name} has {values.size} values, which is neither the active cell "
        f"count ({n_active}) nor the Cartesian cell count ({active.size}); "
        f"refusing to guess how it maps onto the grid"
    )


def read_eclipse_case(prefix: str | Path, *, restart: bool = True) -> EclipseCase:
    """Read `<prefix>.EGRID`, `.INIT` and `.UNRST` into a PyMRST case.

    Only the EGRID is required. A case with no INIT has no rock, and one with
    no UNRST has no states; both are reported in `source` rather than
    substituted with defaults, because a silently invented porosity is worse
    than a missing one.
    """
    from pymrst.gridprocessing import compute_geometry, process_grdecl
    from pymrst.params.rock import Rock

    stem = Path(prefix)
    egrid = stem.with_suffix(".EGRID")
    if not egrid.exists():
        raise FileNotFoundError(f"{egrid} not found; the grid is required")

    corner = read_egrid(egrid)
    # `split_disconnected=False` on purpose. PyMRST's default returns a *list*
    # of components for a disconnected grid, which is a convenience; ECLIPSE's
    # grid is one grid that happens to have disconnected parts, and a case
    # reader silently keeping one component would misrepresent the model.
    # `inspection.validate` reports the disconnection as a warning, which is
    # where that belongs.
    grid = compute_geometry(
        process_grdecl(corner.to_grdecl(), split_disconnected=False)
    )
    active = np.asarray(corner.actnum, dtype=np.int64).ravel() > 0
    source = {"grid": str(egrid)}

    rock = None
    init_path = stem.with_suffix(".INIT")
    if init_path.exists():
        init = read_init(init_path)
        poro = init.get("PORO")
        columns = [init.get(k) for k in _PERM_KEYWORDS]
        present = [c for c in columns if c is not None]
        if poro is not None and present:
            perm = np.column_stack([
                _map_to_active(c, active, k)
                for k, c in zip(_PERM_KEYWORDS, columns) if c is not None
            ])
            rock = Rock(perm=perm, poro=_map_to_active(poro, active, "PORO"))
            source["rock"] = str(init_path)
        else:
            source["rock"] = (
                f"{init_path} has no usable PORO/PERM pair "
                f"(PORO={'yes' if poro is not None else 'no'}, "
                f"PERM={[k for k, c in zip(_PERM_KEYWORDS, columns) if c is not None]})"
            )

    states: list[dict] = []
    restart_path = stem.with_suffix(".UNRST")
    if restart and restart_path.exists():
        for step in read_restart(restart_path):
            recorded = {}
            for keyword, target in (("PRESSURE", "pressure"), ("SWAT", "sW"),
                                    ("SGAS", "sG"), ("SOIL", "sO")):
                values = step.get(keyword)
                if values is not None:
                    recorded[target] = _map_to_active(values, active, keyword)
            if recorded:
                states.append(recorded)
        source["states"] = str(restart_path)

    return EclipseCase(grid, rock, tuple(states), active, source)


def keyword_support() -> dict[str, tuple[str, ...]]:
    """What this reader understands, ignores, and refuses.

    The package asks for this explicitly, and it is not decoration: a reader
    that silently skips a keyword it does not understand produces a case that
    is quietly not the one on disk. Anything absent from `read` is carried
    through as a raw array by `read_init`/`read_restart` and used by nothing.
    """
    return {
        "grid": ("GRIDHEAD", "COORD", "ZCORN", "ACTNUM"),
        "rock": _PERM_KEYWORDS + ("PORO",),
        "state": ("PRESSURE", "SWAT", "SGAS", "SOIL"),
        # SEQNUM is *not* unused: read_restart splits report steps on it, and
        # a reader that ignored it would merge a whole simulation into one
        # step. It carries no case field, which is why it sat here, but
        # calling it unused misreports the one keyword the restart reader most
        # depends on.
        "structural": ("SEQNUM",),
        "read_but_unused": ("INTEHEAD", "FILEHEAD"),
        "refused": (
            "arrays whose length is neither the active nor the Cartesian cell "
            "count",
        ),
        "not_implemented": ("NNC", "LGR", "unit-system conversion"),
    }


__all__ = ["EclipseCase", "keyword_support", "read_eclipse_case"]
