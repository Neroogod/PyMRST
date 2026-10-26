"""Read an unformatted ECLIPSE output file (MRST
``deckformat/resultinput/readEclipseOutputFileUnFmt.m``).

EGRID, INIT, restart (`.UNRST`/`.X####`) and summary files all share one
format: a stream of records, each a 24-byte header (8-character keyword,
element count, 4-character type) followed by the data in blocks, every block
wrapped in Fortran record-length markers.

The reader for that format already existed in this port, as `_fields` inside
`ad_core/utils/read_summary_local.py` - private, and named after summary files.
So "PyMRST has no ECLIPSE output reader" was true of the *name* and false of
the capability, and the Norne field-model example was left unported on that
basis. This is the same reader under MRST's name and in MRST's directory,
returning MRST's shape: a dict of `{keyword: EclipseField}`.

Repeated keywords are concatenated rather than overwritten. A restart file
writes `SEQNUM` once per report step and INIT writes some arrays in several
blocks; keeping only the last occurrence silently truncates them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class EclipseField:
    """One keyword's values, with the type ECLIPSE stored them under."""

    name: str
    type: str
    values: Any

    def __len__(self) -> int:
        return int(np.size(self.values))

    def __array__(self, dtype=None):
        array = np.asarray(self.values)
        return array if dtype is None else array.astype(dtype)


def read_eclipse_output_file_un_fmt(path, *, cellOutput: bool = False,
                                    maxCellSize: float = np.inf) -> Dict[str, EclipseField]:
    """Every keyword in ``path``, in file order.

    With `cellOutput` (MRST's `readEclipseOutputStreamCellOutput`), each
    keyword's `values` is a list with one entry per occurrence of the file's
    first keyword (one per restart step; `None` where the keyword is absent),
    names made MATLAB-valid as MRST makes them, LGR sections left out, and at
    most `maxCellSize` steps read.
    """
    if cellOutput:
        return _read_cell_output(path, maxCellSize)
    out: Dict[str, EclipseField] = {}
    for name, kind, values in iter_eclipse_output_records(path):
        if name in out:
            out[name] = EclipseField(name, kind, _join(out[name].values, values))
        else:
            out[name] = EclipseField(name, kind, values)
    return out


def iter_eclipse_output_records(path) -> Iterator[Tuple[str, str, Any]]:
    """Yield `(keyword, type, values)` for each record, without merging.

    A restart file's records must be read in order to be split on `SEQNUM`,
    which merging by keyword would destroy.
    """
    from pymrst.modules.ad_core.utils.read_summary_local import _fields
    yield from _fields(Path(path))


readEclipseOutputFileUnFmt = read_eclipse_output_file_un_fmt


def _read_cell_output(path, max_cell_size: float) -> Dict[str, EclipseField]:
    out: Dict[str, EclipseField] = {}
    in_lgr = False
    step_count = 0
    first_name = None
    for name, kind, values in iter_eclipse_output_records(path):
        if step_count == 0:
            first_name = name
        if name == first_name:
            step_count += 1
            if step_count > max_cell_size:
                return out
        if values is None or np.size(values) == 0:
            continue
        name = fix_var_name(name)
        if name == "LGR":
            in_lgr = True
        if name == "NNCHEAD" and int(np.ravel(values)[1]) > 0:
            in_lgr = True
        if name in ("ENDLGR", "LGRSGONE"):
            in_lgr = False
        if in_lgr:
            continue
        if name not in out:
            out[name] = EclipseField(name, kind, [])
        cells = out[name].values
        assert out[name].type == kind, "Data type of new field data does not match existing."
        while len(cells) < step_count - 1:
            cells.append(None)
        cells.append(values if isinstance(values, list) else np.asarray(values).ravel())
    return out


def fix_var_name(name: str) -> str:
    """MRST's `fixVarName`: `+` -> `p`, `-` -> `n`, other non-word -> `_`,
    for a name that is not a valid MATLAB variable name."""
    import keyword
    import re
    if re.fullmatch(r"[A-Za-z]\w{0,62}", name):
        return name
    name = re.sub(r"\W", "_", name.replace("+", "p").replace("-", "n"))
    if not re.match(r"[A-Za-z]", name):
        name = "x" + name
    return name if not keyword.iskeyword(name) else name + "_"


def _join(first: Any, second: Any):
    if isinstance(first, list) or isinstance(second, list):
        return list(first) + list(second)
    return np.concatenate([np.asarray(first).ravel(),
                           np.asarray(second).ravel()])


__all__ = ["EclipseField", "fix_var_name", "iter_eclipse_output_records",
           "readEclipseOutputFileUnFmt", "read_eclipse_output_file_un_fmt"]


if __name__ == "__main__":
    NORNE = Path("/Users/Shared/PyMRST_workspace/"
                 "mrst-2024a-Geomech/examples/data/norne_field_model/"
                 "opm-tests/norne/ECL.2014.2/NORNE_ATW2013")

    if not NORNE.with_suffix(".EGRID").exists():
        print("skip (Norne output files not installed)")
        raise SystemExit(0)

    EGRID = read_eclipse_output_file_un_fmt(NORNE.with_suffix(".EGRID"))
    INIT = read_eclipse_output_file_un_fmt(NORNE.with_suffix(".INIT"))

    def test_the_egrid_carries_the_corner_point_geometry():
        for name in ("GRIDHEAD", "COORD", "ZCORN", "ACTNUM"):
            assert name in EGRID, sorted(EGRID)
        dims = np.asarray(EGRID["GRIDHEAD"].values, dtype=np.int64)[1:4]
        assert tuple(dims) == (46, 112, 22), dims
        assert np.size(EGRID["ACTNUM"].values) == int(np.prod(dims))
        print(f"ok test_the_egrid_carries_the_corner_point_geometry ({tuple(dims)})")

    def test_the_active_cell_count_is_nornes():
        active = np.asarray(EGRID["ACTNUM"].values, dtype=np.int64)
        assert int(active.sum()) == 44431, int(active.sum())
        print("ok test_the_active_cell_count_is_nornes")

    def test_the_init_carries_rock_and_transmissibilities():
        for name in ("PORO", "PERMX", "PERMY", "PERMZ", "PORV",
                     "TRANX", "TRANY", "TRANZ", "DEPTH", "INTEHEAD"):
            assert name in INIT, sorted(INIT)[:20]
        assert np.size(INIT["PERMX"].values) == 44431
        assert np.size(INIT["PORV"].values) == 46*112*22
        print("ok test_the_init_carries_rock_and_transmissibilities")

    def test_porv_is_per_cartesian_cell_and_perm_per_active_cell():
        """The asymmetry that decides every index into these arrays. `PORV` is
        written for the whole box - it is how ECLIPSE says which cells it
        actually kept - while `PERMX` is written only for the active ones."""
        assert np.size(INIT["PORV"].values) > np.size(INIT["PERMX"].values)
        print("ok test_porv_is_per_cartesian_cell_and_perm_per_active_cell")

    def test_the_unit_system_is_readable():
        """`INTEHEAD[2]` is 1 metric, 2 field, 3 lab. Guessing it turns a
        permeability in millidarcy into one in square metres."""
        unit = int(np.asarray(INIT["INTEHEAD"].values, dtype=np.int64)[2])
        assert unit in (1, 2, 3), unit
        print(f"ok test_the_unit_system_is_readable ({unit})")

    def test_repeated_keywords_are_concatenated_not_overwritten():
        """INIT writes some arrays in several blocks; keeping the last
        occurrence silently truncates them."""
        counts = {}
        for name, _kind, values in iter_eclipse_output_records(
                NORNE.with_suffix(".INIT")):
            counts[name] = counts.get(name, 0) + int(np.size(values))
        for name, total in counts.items():
            if name in INIT and not isinstance(INIT[name].values, list):
                assert np.size(INIT[name].values) == total, name
        print("ok test_repeated_keywords_are_concatenated_not_overwritten")

    def test_nnc_pairs_are_present():
        """Norne is faulted; its non-neighbour connections are what carry flow
        across the faults, and dropping them disconnects the model."""
        assert "NNC1" in EGRID and "NNC2" in EGRID
        assert np.size(EGRID["NNC1"].values) == np.size(EGRID["NNC2"].values)
        print(f"ok test_nnc_pairs_are_present "
              f"({np.size(EGRID['NNC1'].values)} connections)")

    test_the_egrid_carries_the_corner_point_geometry()
    test_the_active_cell_count_is_nornes()
    test_the_init_carries_rock_and_transmissibilities()
    test_porv_is_per_cartesian_cell_and_perm_per_active_cell()
    test_the_unit_system_is_readable()
    test_repeated_keywords_are_concatenated_not_overwritten()
    test_nnc_pairs_are_present()
    print("\nAll tests passed!")
