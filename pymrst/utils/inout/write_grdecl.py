"""Write a corner-point deck (MRST ``utils/inout/writeGRDECL.m``).

Emits the GRDECL keywords PyMRST can read back: `SPECGRID`, `COORD`, `ZCORN`,
and whichever of `ACTNUM`, `SATNUM`, `PORO`, `PERMX/Y/Z` are present. Round
trips against `read_grdecl`, and a test below checks exactly that rather than
comparing the text to a fixture - the file is a means, and what matters is that
reading it back gives the same grid.

**Only those keywords are written.** Anything else in the dict is dropped
silently by MRST; here it is reported through the return value, because a deck
that quietly loses `MULTX` or a fault definition is worse than one that says
so. `write_grdecl` returns the list of keys it skipped.

Values are written in **SI**, exactly as held in memory. MRST does the same,
which means the file has no `GRIDUNIT` or `MAPUNITS` record and a reader
applying METRIC or FIELD conventions to it will be wrong by the corresponding
factor. `convert_input_units` is the inverse operation and the reason this
matters; the two are easy to use in the wrong order.
"""

from pathlib import Path
from typing import Dict, List, Sequence, Union

import numpy as np

# Keyword -> (values per line, format). Order is the order written.
LAYOUT = (
    ("COORD", 6, "{: 12f}"),
    ("ZCORN", 8, "{: 12f}"),
    ("ACTNUM", 20, "{:d}"),
    ("SATNUM", 20, "{:d}"),
    ("PORO", 4, "{: 18.5e}"),
    ("PERMX", 4, "{: 18.5e}"),
    ("PERMY", 4, "{: 18.5e}"),
    ("PERMZ", 4, "{: 18.5e}"),
)
WRITTEN = {"cartDims"} | {name for name, _, _ in LAYOUT}


def _records(values: Sequence, per_line: int, fmt: str) -> str:
    flat = np.asarray(values).ravel(order="F")
    if fmt.endswith("d}"):
        flat = flat.astype(np.int64)
    lines = []
    for start in range(0, flat.size, per_line):
        lines.append(" ".join(fmt.format(v) for v in flat[start:start + per_line]))
    return "\n".join(lines)


def write_grdecl(grdecl: Dict, filename: Union[str, Path]) -> List[str]:
    """Write ``grdecl`` to ``filename``; return the keys that were skipped."""
    path = Path(filename)
    parts = ["-- Written by PyMRST.\n"]

    dims = grdecl.get("cartDims")
    if dims is not None:
        nx, ny, nz = (int(v) for v in np.asarray(dims).ravel())
        parts.append(f"SPECGRID\n{nx} {ny} {nz} 1 F\n/\n")

    for name, per_line, fmt in LAYOUT:
        if name not in grdecl or grdecl[name] is None:
            continue
        parts.append(f"\n{name}\n{_records(grdecl[name], per_line, fmt)}\n/\n")

    path.write_text("".join(parts))
    return sorted(k for k in grdecl if k not in WRITTEN and grdecl[k] is not None)


writeGRDECL = write_grdecl


if __name__ == "__main__":
    import tempfile

    from pymrst.gridprocessing import compute_geometry, process_grdecl
    from pymrst.utils.inout.eclipse.deckinput_simple.read_grdecl import read_grdecl

    def box(nx=3, ny=2, nz=4):
        xyz = np.zeros((6, nx + 1, ny + 1))
        for i in range(nx + 1):
            for j in range(ny + 1):
                xyz[:, i, j] = [i, j, 0.0, i, j, float(nz)]
        zc = np.zeros((2 * nx, 2 * ny, 2 * nz))
        for k in range(nz):
            zc[:, :, 2 * k] = k
            zc[:, :, 2 * k + 1] = k + 1
        return {"cartDims": np.array([nx, ny, nz]),
                "COORD": xyz.ravel(order="F"), "ZCORN": zc.ravel(order="F")}

    def written(deck):
        handle = tempfile.NamedTemporaryFile("w", suffix=".GRDECL", delete=False)
        handle.close()
        skipped = write_grdecl(deck, handle.name)
        return Path(handle.name), skipped

    def test_it_round_trips_through_read_grdecl() -> None:
        """The property that matters; the text itself is not the point."""
        deck = box()
        path, _ = written(deck)
        back = read_grdecl(path)
        assert tuple(back["cartDims"]) == tuple(deck["cartDims"])
        assert np.allclose(np.asarray(back["COORD"], float),
                           np.asarray(deck["COORD"], float))
        assert np.allclose(np.asarray(back["ZCORN"], float),
                           np.asarray(deck["ZCORN"], float))
        print("ok test_it_round_trips_through_read_grdecl")

    def test_the_round_tripped_deck_builds_the_same_grid() -> None:
        deck = box()
        path, _ = written(deck)
        before = compute_geometry(process_grdecl(deck))
        after = compute_geometry(process_grdecl(read_grdecl(path)))
        assert before.cells.num == after.cells.num
        assert np.isclose(np.sum(before.cells.volumes),
                          np.sum(after.cells.volumes))
        print("ok test_the_round_tripped_deck_builds_the_same_grid")

    def test_optional_arrays_survive() -> None:
        deck = box()
        n = int(np.prod(deck["cartDims"]))
        deck["PORO"] = np.linspace(0.1, 0.3, n)
        deck["ACTNUM"] = np.ones(n, dtype=np.int32)
        path, skipped = written(deck)
        back = read_grdecl(path)
        assert np.allclose(np.asarray(back["PORO"], float), deck["PORO"],
                           atol=1e-5)
        assert np.array_equal(np.asarray(back["ACTNUM"], np.int64),
                              deck["ACTNUM"])
        assert skipped == []
        print("ok test_optional_arrays_survive")

    def test_absent_keywords_are_simply_omitted() -> None:
        path, _ = written(box())
        text = path.read_text()
        assert "PORO" not in text and "ACTNUM" not in text
        assert "COORD" in text and "ZCORN" in text
        print("ok test_absent_keywords_are_simply_omitted")

    def test_unwritable_keywords_are_reported() -> None:
        """MRST drops them silently; see the module docstring."""
        deck = box()
        deck["MULTX"] = np.ones(24)
        deck["FAULTS"] = [("F1", 1, 1, 1, 2, 1, 4, "X")]
        _, skipped = written(deck)
        assert skipped == ["FAULTS", "MULTX"], skipped
        print("ok test_unwritable_keywords_are_reported")

    def test_the_specgrid_record_matches_the_dimensions() -> None:
        path, _ = written(box(5, 4, 3))
        assert "SPECGRID\n5 4 3 1 F\n/" in path.read_text()
        print("ok test_the_specgrid_record_matches_the_dimensions")

    def test_every_record_is_terminated() -> None:
        deck = box()
        deck["PORO"] = np.full(24, 0.2)
        path, _ = written(deck)
        text = path.read_text()
        # One '/' per keyword record: SPECGRID, COORD, ZCORN, PORO.
        assert text.count("/\n") == 4, text.count("/\n")
        print("ok test_every_record_is_terminated")

    def test_integers_are_written_without_a_decimal_point() -> None:
        deck = box()
        deck["ACTNUM"] = np.ones(24, dtype=np.int32)
        path, _ = written(deck)
        block = path.read_text().split("ACTNUM\n")[1].split("/")[0]
        assert "." not in block, block[:60]
        print("ok test_integers_are_written_without_a_decimal_point")

    test_it_round_trips_through_read_grdecl()
    test_the_round_tripped_deck_builds_the_same_grid()
    test_optional_arrays_survive()
    test_absent_keywords_are_simply_omitted()
    test_unwritable_keywords_are_reported()
    test_the_specgrid_record_matches_the_dimensions()
    test_every_record_is_terminated()
    test_integers_are_written_without_a_decimal_point()
    print("\nAll tests passed!")
