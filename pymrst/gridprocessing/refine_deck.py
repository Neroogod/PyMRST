"""Refine a full ECLIPSE deck (MRST ``refineDeck``)."""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping, Sequence

import numpy as np

from pymrst.gridprocessing.refine_grdecl import refine_grdecl


_CELL_FIELDS = {
    "SOLUTION": ("EQLNUM", "PRESSURE", "SWAT", "SOIL", "SGAS", "RS", "RV"),
    "REGIONS": ("SATNUM", "FIPNUM", "PVTNUM", "ROCKNUM", "IMBNUM", "EQLNUM"),
    "PROPS": (
        "SWL", "SWCR", "SGU", "SGL", "SGCR", "SOWCR", "SOGCR", "SWU",
        "ISWCR", "ISGU", "ISWL", "ISWU", "ISGL", "ISOGCR", "ISOWCR",
        "ISGCR", "SWATINIT",
    ),
}


def _as_int(value) -> int:
    if isinstance(value, str):
        return int(float(value.strip().strip("'\"")))
    return int(value)


def _positive(value) -> bool:
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False


def _expand(values, old_dims, factors):
    array = np.asarray(values)
    if array.size != int(np.prod(old_dims)):
        raise ValueError(
            f"cell field has {array.size} values, expected {np.prod(old_dims)}"
        )
    parent = array.reshape(tuple(old_dims), order="F")
    ix = [np.repeat(np.arange(old_dims[d]), factors[d]) for d in range(3)]
    return parent[np.ix_(*ix)].ravel(order="F")


def _refine_controls(schedule, factors, default_well: bool):
    controls = schedule.get("control", schedule.get("CONTROL"))
    if controls is None:
        return
    if isinstance(controls, Mapping):
        control_sequence = [controls]
        single = True
    else:
        control_sequence = list(controls)
        single = False

    for control in control_sequence:
        wkey = next(
            (key for key in control if str(key).upper() == "WELSPECS"), None
        )
        if wkey is None:
            continue
        ckey = next(
            (key for key in control if str(key).upper() == "COMPDAT"), None
        )
        if ckey is None:
            raise ValueError("control with WELSPECS must also contain COMPDAT")
        welspecs = np.atleast_2d(np.asarray(control[wkey], dtype=object)).copy()
        compdat = np.atleast_2d(np.asarray(control[ckey], dtype=object)).copy()
        if welspecs.shape[1] < 4 or compdat.shape[1] < 13:
            raise ValueError("WELSPECS/COMPDAT records are shorter than MRST format")

        for axis in range(2):
            welspecs[:, 2 + axis] = [
                factors[axis] * (_as_int(value) - 1)
                + int(np.ceil(factors[axis] / 2))
                for value in welspecs[:, 2 + axis]
            ]
            compdat[:, 1 + axis] = [
                factors[axis] * (_as_int(value) - 1)
                + int(np.ceil(factors[axis] / 2))
                for value in compdat[:, 1 + axis]
            ]

        direction = np.array([
            str(value).strip().strip("'\"").upper()
            for value in compdat[:, 12]
        ])
        vertical = direction == "Z"
        for row in range(compdat.shape[0]):
            k1, k2 = _as_int(compdat[row, 3]), _as_int(compdat[row, 4])
            if vertical[row]:
                compdat[row, 3] = factors[2] * (k1 - 1) + 1
                compdat[row, 4] = factors[2] * k2
            else:
                center = int(np.ceil(factors[2] / 2))
                compdat[row, 3] = factors[2] * (k1 - 1) + center
                compdat[row, 4] = factors[2] * (k2 - 1) + center

        refined_rows = []
        direction_axis = {"X": 0, "Y": 1, "Z": 2}
        for row in compdat:
            name = str(row[12]).strip().strip("'\"").upper()
            if name not in direction_axis:
                raise ValueError(f"unsupported COMPDAT direction {row[12]!r}")
            axis = direction_axis[name]
            copies = factors[axis] if axis != 2 else 1
            rows = np.tile(row, (copies, 1))
            if axis != 2:
                rows[:, 1 + axis] = [
                    _as_int(row[1 + axis]) + offset
                    for offset in range(copies)
                ]
            if default_well:
                rows[:, 7] = "-1"
                rows[:, 9] = "-1"
            else:
                for column in (7, 9):
                    rows[:, column] = [
                        float(value) / factors[axis] if _positive(value) else value
                        for value in rows[:, column]
                    ]
            refined_rows.extend(rows.tolist())

        control[wkey] = welspecs
        control[ckey] = np.asarray(refined_rows, dtype=object)

    key = "control" if "control" in schedule else "CONTROL"
    schedule[key] = control_sequence[0] if single else control_sequence


def refine_deck(
    deck_in: Mapping,
    dim: Sequence[int],
    *,
    default_well: bool = False,
) -> dict:
    """Refine grid, cell fields, and schedule completions by integer factors."""
    factors = np.asarray(dim, dtype=np.int64).ravel()
    if factors.shape != (3,) or np.any(factors <= 0):
        raise ValueError("dim must contain three positive integers")
    if "RUNSPEC" not in deck_in or "GRID" not in deck_in:
        raise KeyError("refine_deck requires RUNSPEC and GRID")

    deck = deepcopy(dict(deck_in))
    old_dims = np.asarray(deck_in["RUNSPEC"]["DIMENS"], dtype=np.int64).ravel()
    if old_dims.shape != (3,):
        raise ValueError("RUNSPEC.DIMENS must contain three entries")
    new_dims = old_dims * factors
    deck["RUNSPEC"]["DIMENS"] = new_dims.copy()
    deck["RUNSPEC"]["cartDims"] = new_dims.copy()

    for section, fields in _CELL_FIELDS.items():
        if section not in deck:
            continue
        for field in fields:
            if field in deck[section]:
                deck[section][field] = _expand(
                    deck[section][field], old_dims, factors
                )

    deck["GRID"] = refine_grdecl(deck_in["GRID"], factors)
    if "SCHEDULE" in deck:
        _refine_controls(deck["SCHEDULE"], factors, default_well)
    return deck


refineDeck = refine_deck

__all__ = ["refine_deck", "refineDeck"]
