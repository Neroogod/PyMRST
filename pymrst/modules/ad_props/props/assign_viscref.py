"""Viscosity reference conditions from VISCREF (MRST ``props/assignVISCREF.m``).

VISCREF gives the reference pressure and gas-oil ratio at which the
solvent model's viscosities are quoted.

MRST guards this on ``numel(reg.SATNUM) == 1`` - the number of *cells*, not
the number of regions - and that guard is wrong in the common case. A deck
with no REGIONS section has ``reg.SATNUM = []``, so ``numel`` is 0, the test
fails, and MRST evaluates ``val([], :)``, which is an empty array. The
reference viscosity is then silently gone rather than defaulted.

This port takes the first record whenever the field is unpartitioned, which
is the evident intent, and so **differs from MRST** for any deck without a
SATNUM. Reproducing the empty result would hand the solvent model nothing.

The whole record is kept rather than split into named fields, because the
solvent model reads both columns together.
"""

from typing import Any, Dict

import numpy as np


def assign_viscref(f: Dict[str, Any], viscref, reg=None) -> Dict[str, Any]:
    """Store ``refVisc``, the reference pressure and gas-oil ratio."""
    values = np.asarray(viscref, dtype=float).ravel()
    record = np.atleast_2d(np.asarray(viscref, dtype=float))
    if reg is None or reg.SATNUM is None or np.size(reg.SATNUM) == 1:
        f["refVisc"] = record[0]
    else:
        f["refVisc"] = record[reg.SATNUM]
    return f


assignVISCREF = assign_viscref


if __name__ == "__main__":

    from pymrst.modules.ad_props.utils.handle_regions import Regions

    def test_a_single_record_is_stored_whole() -> None:
        f = assign_viscref({}, [[2.0e7, 50.0]], Regions())
        assert np.allclose(f["refVisc"], [2.0e7, 50.0])
        print("ok test_a_single_record_is_stored_whole")

    def test_both_columns_are_kept_together() -> None:
        """The solvent model reads pressure and GOR as a pair."""
        f = assign_viscref({}, [[2.0e7, 50.0]], Regions())
        assert np.size(f["refVisc"]) == 2
        print("ok test_both_columns_are_kept_together")

    def test_several_cells_index_by_satnum() -> None:
        reg = Regions(SATNUM=np.array([0, 1, 0]),
                      SATINX=[np.array([0, 2]), np.array([1])])
        f = assign_viscref({}, [[2.0e7, 50.0], [3.0e7, 80.0]], reg)
        assert f["refVisc"].shape == (3, 2), f["refVisc"].shape
        assert np.allclose(f["refVisc"][1], [3.0e7, 80.0])
        print("ok test_several_cells_index_by_satnum")

    def test_no_regions_falls_back_to_the_first_record() -> None:
        f = assign_viscref({}, [[2.0e7, 50.0]], None)
        assert np.allclose(f["refVisc"], [2.0e7, 50.0])
        print("ok test_no_regions_falls_back_to_the_first_record")

    test_a_single_record_is_stored_whole()
    test_both_columns_are_kept_together()
    test_several_cells_index_by_satnum()
    test_no_regions_falls_back_to_the_first_record()
    print("\nAll tests passed!")
