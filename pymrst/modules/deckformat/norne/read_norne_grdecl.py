"""Assemble the Norne corner-point description."""

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from pymrst.gridprocessing.cut_grdecl import cut_grdecl
from pymrst.modules.deckformat.grdecl import read_keywords
from pymrst.modules.deckformat.norne.constants import (
    CART_DIMS,
    FILES,
    PERMZ_MULTIPLIERS,
    SUBSETS,
)
from pymrst.modules.deckformat.norne.fetch_norne import fetch_norne
from pymrst.utils.units import DARCY, MILLI


def read_norne_grdecl(
    directory: Optional[Path] = None,
    subset: Optional[str] = "full",
    all_active: bool = True,
) -> Dict:
    """Return the Norne corner-point description in strict SI units."""
    directory = Path(directory) if directory else fetch_norne(verbose=False)
    grdecl: Dict = {}
    for remote in FILES:
        grdecl.update(read_keywords(directory / Path(remote).name))
    if "cartDims" not in grdecl:
        grdecl["cartDims"] = CART_DIMS
    assert tuple(grdecl["cartDims"]) == CART_DIMS, grdecl["cartDims"]
    _copy_and_scale_permeability(grdecl)
    if all_active:
        grdecl["ACTNUM"] = np.ones(int(np.prod(CART_DIMS)))
    if subset is not None:
        grdecl = cut_grdecl(grdecl, SUBSETS[subset])
        if all_active:
            grdecl["ACTNUM"] = np.ones(int(np.prod(grdecl["cartDims"])))
    return grdecl


def _copy_and_scale_permeability(grdecl: Dict) -> None:
    nc = int(np.prod(CART_DIMS))
    permx = grdecl["PERMX"]
    assert permx.size == nc, f"PERMX has {permx.size} values, expected {nc}"
    grdecl["PERMY"] = permx.copy()
    permz = permx.copy().reshape(CART_DIMS, order="F")
    permz *= PERMZ_MULTIPLIERS[None, None, :]
    grdecl["PERMZ"] = permz.ravel(order="F")
    for keyword in ("PERMX", "PERMY", "PERMZ"):
        grdecl[keyword] = grdecl[keyword] * (MILLI * DARCY)


if __name__ == "__main__":
    from pymrst.modules.deckformat.norne.data_directory import data_directory
    from pymrst.modules.deckformat.norne.norne_rock import norne_rock

    def test_permz_multipliers_are_the_deck_s() -> None:
        assert PERMZ_MULTIPLIERS.size == CART_DIMS[2] == 22
        assert PERMZ_MULTIPLIERS[3] == 0.0
        assert PERMZ_MULTIPLIERS.max() == 1.0
        print("ok test_permz_multipliers_are_the_deck_s")

    def test_subsets_lie_inside_the_field() -> None:
        for name, box in SUBSETS.items():
            for (lo, hi), count in zip(box, CART_DIMS):
                assert 0 <= lo <= hi < count, (name, lo, hi, count)
        assert [hi - lo + 1 for lo, hi in SUBSETS["full"]] == [16, 21, 22]
        assert [hi - lo + 1 for lo, hi in SUBSETS["mini"]] == [11, 11, 5]
        print("ok test_subsets_lie_inside_the_field")

    def test_reads_the_field_if_it_is_downloaded() -> None:
        directory = data_directory()
        if not all((directory / Path(remote).name).exists() for remote in FILES):
            print("skip test_reads_the_field_if_it_is_downloaded")
            return
        from pymrst.gridprocessing import compute_geometry, process_grdecl

        grdecl = read_norne_grdecl(subset="mini")
        assert grdecl["cartDims"] == (11, 11, 5)
        grid = compute_geometry(process_grdecl(grdecl))
        rock = norne_rock(grid, grdecl)
        assert grid.cells.num == 11 * 11 * 5
        assert np.all(rock.poro >= 0.1)
        assert np.all(rock.perm >= 0.0)
        assert np.all(rock.perm[:, 2].reshape((11, 11, 5), order="F")[:, :, 3] == 0.0)
        print("ok test_reads_the_field_if_it_is_downloaded")

    test_permz_multipliers_are_the_deck_s()
    test_subsets_lie_inside_the_field()
    test_reads_the_field_if_it_is_downloaded()
    print("\nAll tests passed!")

