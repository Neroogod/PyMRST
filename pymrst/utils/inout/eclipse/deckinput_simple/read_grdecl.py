"""Standalone GRDECL reader, ported from MRST `readGRDECL` subset."""

import pathlib
from pathlib import Path
from typing import Dict

from pymrst.utils.inout.eclipse.deckinput_simple.read_keywords import read_keywords


def read_grdecl(path) -> Dict:
    """Parse a self-contained GRDECL file into raw arrays plus `cartDims`."""
    out = read_keywords(path)
    assert "cartDims" in out and "COORD" in out and "ZCORN" in out, (
        "GRDECL file must contain SPECGRID, COORD, and ZCORN"
    )
    nx, ny, nz = out["cartDims"]
    assert out["COORD"].size == 6 * (nx + 1) * (ny + 1), "bad COORD size"
    assert out["ZCORN"].size == 8 * nx * ny * nz, "bad ZCORN size"
    return out


def _mrst_data_dir():
    """Find MRST's example data by searching upward, not by counting levels.

    These files moved from `modules/deckformat/grdecl/` to
    `utils/inout/eclipse/deckinput_simple/`, which changed their depth and
    silently broke a hard-coded `parents[5]`. Searching for the sibling tree
    survives the next move too.
    """
    for parent in pathlib.Path(__file__).resolve().parents:
        candidate = parent / "mrst-2024a-Geomech" / "examples" / "data"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "could not find mrst-2024a-Geomech/examples/data above "
        + str(pathlib.Path(__file__).resolve()))


if __name__ == "__main__":
    deck_dir = _mrst_data_dir()

    def test_read_spe1_grdecl() -> None:
        deck = read_grdecl(deck_dir / "SPE1/SPE1.GRDECL")
        assert deck["cartDims"] == (10, 10, 3)
        assert deck["COORD"].size == 6 * 11 * 11
        assert deck["ZCORN"].size == 8 * 300
        assert deck["ZCORN"].min() == 8325.0
        assert deck["ZCORN"].max() == 8425.0
        print("ok test_read_spe1_grdecl")

    test_read_spe1_grdecl()
    print("\nAll tests passed!")

