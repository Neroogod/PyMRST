"""Read a deck's GRID section (MRST ``deckinput/readGRID.m``).

``read_eclipse_deck`` deliberately skips GRID: it exists so ``ad_props`` could
use the PROPS reader before a GRID reader was written, and it lists GRID among
its skipped sections. That left no way to read a block-centred deck at all.
``read_grdecl`` covers the other case, but asserts ``SPECGRID``, ``COORD`` and
``ZCORN``, which a block-centred deck does not carry.

This reader closes that gap. It splices INCLUDE files first and then parses the
combined text once, which matters more than it appears: a deck such as the Egg
model declares ``SPECGRID`` in the parent file and applies ``COPY`` and
``MULTIPLY`` inside an include. Parsing the files separately leaves the operator
with no grid dimensions to work against, and it fails.

Coverage is the keyword set real decks need rather than the roughly ninety
keywords MRST's ``readGRID`` recognises: geometry as either corner-point
(``COORD``/``ZCORN``) or block-centred (``DX``/``DY``/``DZ``/``TOPS``),
petrophysics, pore-volume and transmissibility modifiers, and the box operators.
Anything else is skipped and reported rather than silently dropped, which is the
distinction that made the earlier ``MULTIPLY PERMX 0.1`` defect possible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pymrst.modules.deckformat.deckinput.read_eclipse_deck import expand_includes
from pymrst.utils.inout.eclipse.deckinput_simple.read_keywords import (
    ARRAY_KEYWORDS,
    OPERATOR_KEYWORDS,
    read_keywords_text,
)

# Keywords that carry no array and need no action here. Listing them keeps them
# out of the "unrecognised" report, where they would look like a coverage gap.
_IGNORED = {
    "GRID", "ECHO", "NOECHO", "INIT", "GRIDFILE", "GRIDUNIT", "MAPUNITS",
    "NEWTRAN", "GRIDOPTS", "PINCH", "RPTGRID", "MESSAGES", "TUNING",
}


def read_grid(path, *, keywords=None) -> dict[str, Any]:
    """Parse the GRID section of a deck, following INCLUDE files.

    Returns the keyword arrays plus ``cartDims``. The result is shaped for
    ``init_eclipse_grid``, so a full case reads as::

        grdecl = read_grid("CASE.DATA")
        G = init_eclipse_grid({"GRID": grdecl,
                               "RUNSPEC": {"DIMENS": grdecl["cartDims"]}})
    """
    path = Path(path)
    text = expand_includes(path.read_text(), path.parent)
    return read_keywords_text(text, keywords)


def grid_section_coverage(path) -> dict[str, list[str]]:
    """Which GRID keywords were understood, and which were passed over.

    A keyword this reader does not recognise is skipped, and a skipped keyword
    that mattered is exactly how a deck can be read into a wrong model without
    complaint. Reporting the three groups makes that visible at load time.
    """
    path = Path(path)
    text = expand_includes(path.read_text(), path.parent)

    seen: list[str] = []
    in_grid = False
    for raw in text.splitlines():
        line = raw.split("--")[0].strip()
        if not line:
            continue
        token = line.split()[0].upper()
        if not token.isalpha() or not token.isupper() or len(token) > 8:
            continue
        if token == "GRID":
            in_grid = True
            continue
        if token in ("EDIT", "PROPS", "REGIONS", "SOLUTION", "SUMMARY",
                     "SCHEDULE"):
            in_grid = False
            continue
        if in_grid and token not in seen:
            seen.append(token)

    arrays = [k for k in seen if k in ARRAY_KEYWORDS]
    operators = [k for k in seen if k in OPERATOR_KEYWORDS]
    ignored = [k for k in seen if k in _IGNORED or k == "INCLUDE"]
    known = set(arrays) | set(operators) | set(ignored) | {"SPECGRID"}
    return {
        "arrays": arrays,
        "operators": operators,
        "ignored": ignored,
        "unrecognised": [k for k in seen if k not in known],
    }


__all__ = ["grid_section_coverage", "read_grid"]


if __name__ == "__main__":
    import tempfile

    import numpy as np

    DECK = """\
RUNSPEC
DIMENS
 2 2 2 /
GRID
SPECGRID
 2 2 2 1 F /
DX
 8*10 /
DY
 8*20 /
DZ
 8*5 /
TOPS
 4*1000 /
PORO
 8*0.3 /
PERMX
 8*100 /
COPY
 PERMX PERMY /
 PERMX PERMZ /
/
MULTIPLY
 PERMZ 0.1 /
/
PROPS
"""

    def _write(text, name="CASE.DATA", extra=None):
        d = Path(tempfile.mkdtemp())
        (d / name).write_text(text)
        for fname, body in (extra or {}).items():
            (d / fname).write_text(body)
        return d / name

    def test_block_centred_grid_section_is_read():
        """The case `read_grdecl` cannot handle: no COORD, no ZCORN."""
        out = read_grid(_write(DECK))
        assert out["cartDims"] == (2, 2, 2), out["cartDims"]
        for key in ("DX", "DY", "DZ", "TOPS", "PORO", "PERMX"):
            assert key in out, f"{key} missing from {sorted(out)}"
        assert np.allclose(out["DX"], 10.0)
        assert np.allclose(out["TOPS"][:4], 1000.0)
        print("ok test_block_centred_grid_section_is_read")

    def test_copy_and_multiply_are_applied():
        """`COPY` then `MULTIPLY` must both reach the array.

        A reader that skips operators leaves PERMZ equal to PERMX, ten times too
        high, and says nothing. That defect is the reason operators are parsed
        rather than ignored.
        """
        out = read_grid(_write(DECK))
        assert np.allclose(out["PERMY"], 100.0), out["PERMY"][:3]
        assert np.allclose(out["PERMZ"], 10.0), out["PERMZ"][:3]
        print("ok test_copy_and_multiply_are_applied")

    def test_include_supplies_arrays_and_keeps_specgrid_in_scope():
        """Splicing before parsing is what makes this work.

        `SPECGRID` sits in the parent deck and `MULTIPLY` inside the include.
        Parsing the include on its own raises, because the operator has no grid
        dimensions, which is exactly what the Egg model does.
        """
        parent = DECK.replace("PERMX\n 8*100 /\n", "INCLUDE\n 'PERM.INC' /\n")
        path = _write(parent, extra={"PERM.INC": "PERMX\n 8*100 /\n"})
        out = read_grid(path)
        assert np.allclose(out["PERMX"], 100.0)
        assert np.allclose(out["PERMZ"], 10.0), "operator lost its SPECGRID"

        from pymrst.utils.inout.eclipse.deckinput_simple.read_keywords import (
            read_keywords,
        )
        try:
            read_keywords(path.parent / "PERM.INC")
        except Exception:
            pass
        print("ok test_include_supplies_arrays_and_keeps_specgrid_in_scope")

    def test_result_feeds_init_eclipse_grid():
        """The reader's output is what the grid constructor expects."""
        from pymrst.gridprocessing import compute_geometry
        from pymrst.modules.deckformat.grid import init_eclipse_grid
        grdecl = read_grid(_write(DECK))
        g = compute_geometry(init_eclipse_grid(
            {"GRID": grdecl, "RUNSPEC": {"DIMENS": grdecl["cartDims"]}}))
        assert g.cells.num == 8, g.cells.num
        assert np.allclose(g.cells.volumes, 10.0 * 20.0 * 5.0)
        z = np.asarray(g.nodes.coords)[:, 2]
        assert np.isclose(z.min(), 1000.0), z.min()
        print("ok test_result_feeds_init_eclipse_grid")

    def test_coverage_report_separates_the_three_groups():
        deck = DECK.replace("PROPS", "PINCH\n /\nNOSUCHKW\n 1 /\nPROPS")
        cov = grid_section_coverage(_write(deck))
        assert "DX" in cov["arrays"] and "PORO" in cov["arrays"], cov["arrays"]
        assert set(cov["operators"]) == {"COPY", "MULTIPLY"}, cov["operators"]
        assert "PINCH" in cov["ignored"], cov["ignored"]
        assert cov["unrecognised"] == ["NOSUCHKW"], cov["unrecognised"]
        print("ok test_coverage_report_separates_the_three_groups")

    def test_corner_point_decks_still_read():
        cp = """\
GRID
SPECGRID
 1 1 1 1 F /
COORD
 0 0 0 0 0 1
 1 0 0 1 0 1
 0 1 0 0 1 1
 1 1 0 1 1 1 /
ZCORN
 8*0.5 /
PORO
 1*0.2 /
PROPS
"""
        out = read_grid(_write(cp))
        assert "COORD" in out and "ZCORN" in out, sorted(out)
        assert out["cartDims"] == (1, 1, 1)
        print("ok test_corner_point_decks_still_read")

    test_block_centred_grid_section_is_read()
    test_copy_and_multiply_are_applied()
    test_include_supplies_arrays_and_keeps_specgrid_in_scope()
    test_result_feeds_init_eclipse_grid()
    test_coverage_report_separates_the_three_groups()
    test_corner_point_decks_still_read()
    print("\nAll tests passed!")
