"""Constant-compressibility dead oil from PVCDO (MRST ``props/assignPVCDO.m``).

PVCDO is PVDO's closed-form counterpart: instead of a table, one record giving
a reference pressure, FVF, compressibility, viscosity and viscosibility.

Note the contrast with PVTW, which describes water with the *same five
quantities* but expands them to second order. PVCDO uses true exponentials.
The two are not interchangeable, and the difference is MRST's, not an
oversight here.
"""

from typing import Any, Callable, Dict, List

import numpy as np


def assign_pvcdo(f: Dict[str, Any], pvcdo, reg) -> Dict[str, Any]:
    """Store ``bO`` and ``muO`` as exponentials, one pair per PVT region."""
    pvcdo = np.atleast_2d(np.asarray(pvcdo, dtype=float))
    b_functions: List[Callable] = []
    mu_functions: List[Callable] = []

    for index in range(reg.pvt):
        p_ref, b_ref, compressibility, mu_ref, viscosibility = pvcdo[index, :5]
        b_functions.append(
            lambda p, *, c=compressibility, b=b_ref, pr=p_ref:
                np.exp(c * (p - pr)) / b)
        mu_functions.append(
            lambda p, *, m=mu_ref, v=viscosibility, pr=p_ref:
                m * np.exp(v * (p - pr)))

    f["bO"], f["muO"] = b_functions, mu_functions
    return f


assignPVCDO = assign_pvcdo


if __name__ == "__main__":
    from pymrst.modules.ad_props.props.assign_pvtw import b_w
    from pymrst.modules.ad_props.utils.handle_regions import Regions

    RECORD = [2.0e7, 1.20, 1.0e-9, 1.0e-3, 2.0e-10]

    def test_b_is_the_reciprocal_reference_fvf_at_p_ref() -> None:
        f = assign_pvcdo({}, [RECORD], Regions())
        assert np.isclose(f["bO"][0](RECORD[0]), 1.0 / RECORD[1])
        print("ok test_b_is_the_reciprocal_reference_fvf_at_p_ref")

    def test_viscosity_is_the_reference_at_p_ref() -> None:
        f = assign_pvcdo({}, [RECORD], Regions())
        assert np.isclose(f["muO"][0](RECORD[0]), RECORD[3])
        print("ok test_viscosity_is_the_reference_at_p_ref")

    def test_both_rise_with_pressure() -> None:
        f = assign_pvcdo({}, [RECORD], Regions())
        assert f["bO"][0](5.0e7) > f["bO"][0](1.0e7)
        assert f["muO"][0](5.0e7) > f["muO"][0](1.0e7)
        print("ok test_both_rise_with_pressure")

    def test_pvcdo_is_exponential_where_pvtw_is_a_series() -> None:
        """The same five fields, deliberately different formulas."""
        p_ref, b_ref, c = RECORD[0], RECORD[1], 5e-8
        far = 3.0e8
        exponential = assign_pvcdo({}, [[p_ref, b_ref, c, 1e-3, 0.0]],
                                   Regions())["bO"][0](far)
        series = b_w(far, c, b_ref, p_ref)
        assert not np.isclose(exponential, series, rtol=1e-3), \
            (exponential, series)
        print("ok test_pvcdo_is_exponential_where_pvtw_is_a_series")

    def test_one_pair_of_functions_per_pvt_region() -> None:
        f = assign_pvcdo({}, [RECORD, [1.0e7, 1.1, 2e-9, 2e-3, 0.0]],
                         Regions(pvt=2))
        assert len(f["bO"]) == 2
        assert np.isclose(f["muO"][1](1.0e7), 2e-3)
        print("ok test_one_pair_of_functions_per_pvt_region")

    test_b_is_the_reciprocal_reference_fvf_at_p_ref()
    test_viscosity_is_the_reference_at_p_ref()
    test_both_rise_with_pressure()
    test_pvcdo_is_exponential_where_pvtw_is_a_series()
    test_one_pair_of_functions_per_pvt_region()
    print("\nAll tests passed!")
