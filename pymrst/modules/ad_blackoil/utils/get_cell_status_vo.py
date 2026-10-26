"""MRST ``ad-blackoil/utils/getCellStatusVO.m``.

Which of the three meanings black-oil's switched variable `x` has in each
cell. MRST encodes it as `status = oilPresent + 2*gasPresent`:

- **1** - oil but no free gas: the oil is undersaturated and `x` is `rs`;
- **2** - gas but no free oil: the gas is undersaturated and `x` is `rv`;
- **3** - both phases present: `x` is `sG`.

The result is MRST's cell array `{st1, st2, st3}` of logical masks. `st1` is
only ever true for a model with dissolved gas and `st2` only for one with
vaporised oil; otherwise each is a scalar `False`, as in MRST.

A water-only cell counts as having *both* hydrocarbon phases. Nothing there
constrains `rs` or `rv`, so pinning them to their saturation curves (status 3,
`x = sG`) is the only well-posed choice.
"""

from typing import Any, List, Optional

import numpy as np

from pymrst.utils.adi import value

WATER_ONLY_TOLERANCE: float = 1.0 - np.sqrt(np.finfo(float).eps)


def get_cell_status_vo(s_o: Any, s_w: Any, s_g: Any, *,
                       status: Optional[Any] = None, vapoil: bool = False,
                       disgas: bool = False) -> List[Any]:
    """Return `[st1, st2, st3]`: undersaturated oil, undersaturated gas, both."""
    if status is None:
        s_o, s_w, s_g = (np.asarray(value(x), dtype=float) for x in (s_o, s_w, s_g))
        water_only = s_w > WATER_ONLY_TOLERANCE
        oil_present = np.logical_or(s_o > 0, water_only) if vapoil else True
        gas_present = np.logical_or(s_g > 0, water_only) if disgas else True
        status = np.asarray(oil_present, dtype=int) + 2*np.asarray(gas_present, dtype=int)
    status = np.asarray(status)
    st1 = status == 1 if disgas else np.False_
    st2 = status == 2 if vapoil else np.False_
    st3 = status == 3
    return [st1, st2, st3]


getCellStatusVO = get_cell_status_vo

__all__ = ["get_cell_status_vo", "getCellStatusVO"]


if __name__ == "__main__":
    S_O = np.array([0.8, 0.0, 0.5, 0.0])
    S_W = np.array([0.2, 0.7, 0.2, 1.0])
    S_G = np.array([0.0, 0.3, 0.3, 0.0])

    def test_live_oil_and_wet_gas_classify_every_cell():
        st1, st2, st3 = get_cell_status_vo(S_O, S_W, S_G, disgas=True, vapoil=True)
        np.testing.assert_array_equal(st1, [True, False, False, False])
        np.testing.assert_array_equal(st2, [False, True, False, False])
        # A water-only cell is declared to hold both hydrocarbon phases.
        np.testing.assert_array_equal(st3, [False, False, True, True])
        print("ok test_live_oil_and_wet_gas_classify_every_cell")

    def test_without_vaporised_oil_there_is_no_undersaturated_gas():
        st1, st2, st3 = get_cell_status_vo(S_O, S_W, S_G, disgas=True)
        assert st2 is np.False_
        np.testing.assert_array_equal(st1, [True, False, False, False])
        np.testing.assert_array_equal(st3, [False, True, True, True])
        print("ok test_without_vaporised_oil_there_is_no_undersaturated_gas")

    def test_dead_oil_dry_gas_is_three_phase_everywhere():
        st1, st2, st3 = get_cell_status_vo(S_O, S_W, S_G)
        assert st1 is np.False_ and st2 is np.False_
        assert np.all(st3)
        print("ok test_dead_oil_dry_gas_is_three_phase_everywhere")

    def test_a_given_status_overrides_the_saturations():
        st1, st2, st3 = get_cell_status_vo(S_O, S_W, S_G, status=[3, 3, 1, 2],
                                           disgas=True, vapoil=True)
        np.testing.assert_array_equal(st1, [False, False, True, False])
        np.testing.assert_array_equal(st2, [False, False, False, True])
        np.testing.assert_array_equal(st3, [True, True, False, False])
        print("ok test_a_given_status_overrides_the_saturations")

    def test_the_masks_partition_a_live_oil_wet_gas_domain():
        st1, st2, st3 = get_cell_status_vo(S_O, S_W, S_G, disgas=True, vapoil=True)
        np.testing.assert_array_equal(st1.astype(int) + st2 + st3, 1)
        print("ok test_the_masks_partition_a_live_oil_wet_gas_domain")

    test_live_oil_and_wet_gas_classify_every_cell()
    test_without_vaporised_oil_there_is_no_undersaturated_gas()
    test_dead_oil_dry_gas_is_three_phase_everywhere()
    test_a_given_status_overrides_the_saturations()
    test_the_masks_partition_a_live_oil_wet_gas_domain()
    print("\nAll tests passed!")
