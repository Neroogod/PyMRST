"""Standalone GRDECL reader, ported from MRST `readGRDECL` subset."""

from pathlib import Path
from typing import Dict

from pymrst.modules.deckformat.grdecl.read_keywords import read_keywords


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


if __name__ == "__main__":
    root = Path(__file__).parents[5]
    deck_dir = root / "mrst-2024a-Geomech/examples/data"

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

